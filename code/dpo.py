import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.distributions import Normal

from code.llm import *
from code.adapter import *
from code.preference_data import *
from code.actor import *
from code.evaluation import *
from code.prune import *

def get_pruning_action_from_actor(
    actor_model_eval, actor_input_text, tokenizer_actor,
    num_total_llm_layers, max_seq_len_actor, current_device
    ):
    actor_model_eval.eval() # Ensure actor is in eval mode
    tokenized = tokenizer_actor(
        actor_input_text, truncation=True, padding='max_length',
        max_length=max_seq_len_actor, return_tensors="pt"
    )
    input_ids_actor = tokenized.input_ids.to(current_device)
    attention_mask_actor = tokenized.attention_mask.to(current_device)

    with torch.no_grad():
        layers_log_probs, mu_ratio, _ = actor_model_eval(input_ids_actor, attention_mask_actor)

    mu_ratio_scalar = mu_ratio.squeeze().item()
    k_pruned = int(round(mu_ratio_scalar * num_total_llm_layers))
    k_pruned = max(0, min(k_pruned, num_total_llm_layers)) # Clamp k

    layer_scores = layers_log_probs.squeeze() # No need for exp if just taking topk
    
    if k_pruned > 0:
        _, top_k_indices = torch.topk(layer_scores, k_pruned)
        pruned_indices_list = top_k_indices.cpu().tolist()
    else:
        pruned_indices_list = []
        
    return pruned_indices_list, k_pruned, mu_ratio_scalar

def run_dpo_evaluation_epoch(
    actor_model_eval, model_llm_eval, tokenizer_llm_eval, pruner_llm,
    dataset_llm_eval, tokenizer_actor_eval, num_total_llm_layers,
    max_seq_len_actor_eval, current_device, epoch, batch_num_actor
    ):
    print(f"\n--- Running DPO Evaluation after Epoch {epoch}, Actor Batch {batch_num_actor} ---")
    actor_model_eval.eval()
    model_llm_eval.eval() # Ensure LLM is in eval mode

    total_eval_score = 0.0
    total_k_pruned = 0
    total_mu_ratio = 0.0
    num_eval_samples_processed = 0

    for i in tqdm(range(len(dataset_llm_eval)), desc="DPO Eval"):
        sample = dataset_llm_eval[i]
        
        # Actor input: Use same format as preference data
        # actor_input_text_eval = generate_prompt_str(sample, tokenizer_llm_eval) # Option 1: LLM prompt style
        actor_input_text_eval = sample['tools'] + '\n' + sample['query'] # Option 2: Preference data style

        pruned_layer_indices, k_val, mu_ratio_val = get_pruning_action_from_actor(
            actor_model_eval, actor_input_text_eval, tokenizer_actor_eval,
            num_total_llm_layers, max_seq_len_actor_eval, current_device
        )
        
        pruner_llm.prune_model(pruned_layer_indices)
        
        llm_generated_output = generate_llm_output(sample, model_llm_eval, tokenizer_llm_eval, current_device)
        
        pruner_llm._restore_original_layers() # Important to restore for next sample / training

        score = calc_score(llm_generated_output, sample['answers'])

        total_eval_score += score
        total_k_pruned += k_val
        total_mu_ratio += mu_ratio_val
        num_eval_samples_processed += 1

    avg_eval_score = total_eval_score / num_eval_samples_processed if num_eval_samples_processed > 0 else 0
    avg_k_pruned = total_k_pruned / num_eval_samples_processed if num_eval_samples_processed > 0 else 0
    avg_mu_ratio = total_mu_ratio / num_eval_samples_processed if num_eval_samples_processed > 0 else 0

    print(f"Avg Evaluation Score: {avg_eval_score:.4f}")
    print(f"Avg k Pruned: {avg_k_pruned:.2f}, Avg Pred. Mu Ratio: {avg_mu_ratio:.4f}")

    if wandb.run:
        wandb.log({
            "eval/avg_score": avg_eval_score,
            "eval/avg_k_pruned": avg_k_pruned,
            "eval/avg_mu_ratio": avg_mu_ratio,
            "dpo_epoch_eval": epoch, # Distinguish from training epoch
            "dpo_batch_eval": batch_num_actor
        })
    
    actor_model_eval.train() # Set actor back to train mode
    return avg_eval_score

def get_action_log_probs(
    pred_layers_log_probs, pred_mu_ratio, pred_log_std_ratio,
    target_layers_list, target_ratios, target_ks, current_device
    ):
    batch_size = pred_layers_log_probs.size(0)
    dist_ratio = Normal(pred_mu_ratio.squeeze(-1), torch.exp(pred_log_std_ratio.squeeze(-1)))
    log_prob_ratio = dist_ratio.log_prob(target_ratios)

    log_prob_layers_selection = torch.zeros(batch_size, device=current_device)
    for i in range(batch_size):
        k_val = target_ks[i].item()
        if k_val > 0:
            indices = target_layers_list[i].to(current_device)
            # Ensure indices are within bounds for pred_layers_log_probs[i]
            if indices.max() >= pred_layers_log_probs.size(1):
                # This case should ideally not happen if num_llm_layers is consistent
                print(f"Warning: Index out of bounds in get_action_log_probs. Max index: {indices.max()}, Log probs size: {pred_layers_log_probs.size(1)}")
                # Handle error, e.g. by assigning a very low probability or skipping
                log_prob_layers_selection[i] = -float('inf') # Or some large negative number
                continue

            gathered_log_probs = pred_layers_log_probs[i].gather(0, indices)
            log_prob_layers_selection[i] = gathered_log_probs.sum()
    return log_prob_ratio + log_prob_layers_selection

def train_dpo_actor(
    actor_model, train_pref_dataset, num_total_llm_layers_train,
    # For evaluation
    model_llm_train, tokenizer_llm_train, adapters_llm_train, eval_dataset_llm, tokenizer_actor_train,
    # Configs
    epochs_actor, batch_size_actor, lr_actor, current_device,
    max_seq_len_actor_train, eval_config_batches, gradient_clip
    ):
    actor_model.to(current_device)
    actor_model.train()
    optimizer_actor = AdamW(actor_model.parameters(), lr=lr_actor)
    
    # Ensure num_llm_layers used by dataset matches actor if not passed directly
    # For PruningPreferenceDataset instance, num_llm_layers is already baked in.
    # We need num_total_llm_layers_train for ratio calculation in dataset, if not pre-calculated.

    dataloader_actor = DataLoader(
        train_pref_dataset, batch_size=batch_size_actor, shuffle=True, collate_fn=collate_preference_data
    )

    pruner_for_eval = LLMPruner(model_llm_train, adapters_llm_train)
    best_eval_score = -float('inf')

    for epoch_idx in range(epochs_actor):
        actor_model.train()
        total_loss_epoch_actor = 0
        processed_batches_actor = 0

        for batch_idx, batch_data in enumerate(tqdm(dataloader_actor, desc=f"Epoch {epoch_idx+1}/{epochs_actor}")):
            input_ids_actor_b = batch_data["input_ids"].to(current_device)
            attention_mask_actor_b = batch_data["attention_mask"].to(current_device)
            winner_layers_list_b = batch_data["winner_layers"]
            winner_ratios_b = batch_data["winner_ratio"].to(current_device)
            winner_ks_b = batch_data["winner_k"].to(current_device)
            loser_layers_list_b = batch_data["loser_layers"]
            loser_ratios_b = batch_data["loser_ratio"].to(current_device)
            loser_ks_b = batch_data["loser_k"].to(current_device)

            optimizer_actor.zero_grad()
            
            pred_layers_log_probs_b, pred_mu_ratio_b, pred_log_std_ratio_b = actor_model(
                input_ids_actor_b, attention_mask_actor_b
            )

            log_probs_winner = get_action_log_probs(
                pred_layers_log_probs_b, pred_mu_ratio_b, pred_log_std_ratio_b,
                winner_layers_list_b, winner_ratios_b, winner_ks_b, current_device
            )
            log_probs_loser = get_action_log_probs(
                pred_layers_log_probs_b, pred_mu_ratio_b, pred_log_std_ratio_b,
                loser_layers_list_b, loser_ratios_b, loser_ks_b, current_device
            )

            # Check for NaNs/Infs before loss calculation
            if torch.isinf(log_probs_winner).any() or torch.isnan(log_probs_winner).any() or \
               torch.isinf(log_probs_loser).any() or torch.isnan(log_probs_loser).any():
                print(f"Warning: NaN/Inf in log_probs at epoch {epoch_idx+1}, batch {batch_idx+1}. Skipping batch.")
                wandb.log({"warning_nan_inf_log_probs": 1})
                continue

            preference_term = -F.logsigmoid(log_probs_winner - log_probs_loser)
            loss_actor = preference_term.mean()

            if torch.isnan(loss_actor) or torch.isinf(loss_actor):
                print(f"Warning: Loss is NaN/Inf at epoch {epoch_idx+1}, batch {batch_idx+1}. Skipping batch.")
                wandb.log({"warning_nan_inf_loss": 1})
                continue

            loss_actor.backward()
            if gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(actor_model.parameters(), max_norm=gradient_clip)
            optimizer_actor.step()

            total_loss_epoch_actor += loss_actor.item()
            processed_batches_actor +=1
            
            if wandb.run:
                wandb.log({
                    "train/dpo_loss_batch": loss_actor.item(),
                    "dpo_epoch_train": epoch_idx + 1,
                    "dpo_batch_train": batch_idx + 1
                })

            # --- Periodic Evaluation ---
            if eval_config_batches > 0 and (batch_idx + 1) % eval_config_batches == 0:
                current_eval_score = run_dpo_evaluation_epoch(
                    actor_model, model_llm_train, tokenizer_llm_train, pruner_for_eval,
                    eval_dataset_llm, tokenizer_actor_train, num_total_llm_layers_train,
                    max_seq_len_actor_train, current_device, epoch_idx + 1, batch_idx + 1
                )
                if current_eval_score > best_eval_score:
                    best_eval_score = current_eval_score
                    print(f"New best evaluation score: {best_eval_score:.4f}. Saving actor model.")
                    torch.save(actor_model.state_dict(), "best_dpo_actor_model.pth")
                    wandb.save("best_dpo_actor_model.pth") # Save to wandb
                actor_model.train() # Ensure actor is back in train mode

        avg_epoch_loss_actor = total_loss_epoch_actor / processed_batches_actor if processed_batches_actor > 0 else 0
        print(f"--- DPO Actor Epoch {epoch_idx+1} finished. Average Loss: {avg_epoch_loss_actor:.4f} ---")
        if wandb.run:
            wandb.log({"train/dpo_loss_epoch": avg_epoch_loss_actor, "dpo_epoch_train_end": epoch_idx + 1})
        
        # --- End of Epoch Evaluation (if not done per batch) ---
        if eval_config_batches == 0: # Eval at end of epoch
            current_eval_score = run_dpo_evaluation_epoch(
                actor_model, model_llm_train, tokenizer_llm_train, pruner_for_eval,
                eval_dataset_llm, tokenizer_actor_train, num_total_llm_layers_train,
                max_seq_len_actor_train, current_device, epoch_idx + 1, len(dataloader_actor)
            )
            if current_eval_score > best_eval_score:
                best_eval_score = current_eval_score
                print(f"New best evaluation score: {best_eval_score:.4f}. Saving actor model.")
                torch.save(actor_model.state_dict(), "best_dpo_actor_model.pth")
                wandb.save("best_dpo_actor_model.pth")
            actor_model.train()


    pruner_for_eval._restore_original_layers() # Final cleanup
    print("DPO Actor Training finished.")
    return actor_model
