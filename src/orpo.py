import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.distributions import Normal
from tqdm import tqdm
import wandb
import os

from src.llm import *
from src.adapter import *
from src.preference_data import *
from src.actor import *
from src.evaluation import *

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

def train_orpo_actor(
        actor_model, train_pref_dataset, num_total_llm_layers_train,
        # For evaluation
        model_llm_train, tokenizer_llm_train, adapters_llm_train, eval_dataset_llm, tokenizer_actor_train,
        # Configs
        epochs_actor, batch_size_actor, lr_actor, beta_orpo_actor, current_device,
        max_seq_len_actor_train, eval_config_batches, gradient_clip, checkpoints_path,
        # Logging
        verbose
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
                if wandb.run:
                    wandb.log({"warning_nan_inf_log_probs": 1})
                continue

            preference_term = -F.logsigmoid(log_probs_winner - log_probs_loser)
            sft_term = -log_probs_winner 
            loss_actor = (preference_term + beta_orpo_actor * sft_term).mean()

            if torch.isnan(loss_actor) or torch.isinf(loss_actor):
                print(f"Warning: Loss is NaN/Inf at epoch {epoch_idx+1}, batch {batch_idx+1}. Skipping batch.")
                if wandb.run:
                    wandb.log({"warning_nan_inf_loss": 1})
                continue

            loss_actor.backward()
            if gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(actor_model.parameters(), max_norm=gradient_clip)
            optimizer_actor.step()

            total_loss_epoch_actor += loss_actor.item()
            processed_batches_actor +=1
            
            if verbose and (batch_idx % 10 == 0): # Print every 10 batches or as needed
                print(f"  Batch {batch_idx+1} Training Details:")
                # Detailed logging for the first item in the batch as an example
                print(f"    Input Text (first in batch): {tokenizer_actor_train.decode(input_ids_actor_b[0], skip_special_tokens=True)[:100]}...")
                print(f"    Predicted mu_ratio (first in batch): {pred_mu_ratio_b[0].item():.4f}")
                # pred_layers_log_probs_b is (batch_size, num_layers)
                # layers_log_probs_winner_first = pred_layers_log_probs_b[0][winner_layers_list_b[0]].cpu().tolist() if winner_ks_b[0].item() > 0 else []
                # layers_log_probs_loser_first = pred_layers_log_probs_b[0][loser_layers_list_b[0]].cpu().tolist() if loser_ks_b[0].item() > 0 else []
                
                print(f"    Winner Details (first in batch):")
                print(f"      Target k: {winner_ks_b[0].item()}, Target Ratio: {winner_ratios_b[0].item():.4f}")
                print(f"      Target Layers: {winner_layers_list_b[0].cpu().tolist() if winner_ks_b[0].item() > 0 else '[]'}")
                print(f"      Log Prob Winner: {log_probs_winner[0].item():.4f}")
                # print(f"      Log Probs for Chosen Winner Layers: {layers_log_probs_winner_first}")

                print(f"    Loser Details (first in batch):")
                print(f"      Target k: {loser_ks_b[0].item()}, Target Ratio: {loser_ratios_b[0].item():.4f}")
                print(f"      Target Layers: {loser_layers_list_b[0].cpu().tolist() if loser_ks_b[0].item() > 0 else '[]'}")
                print(f"      Log Prob Loser: {log_probs_loser[0].item():.4f}")
                # print(f"      Log Probs for Chosen Loser Layers: {layers_log_probs_loser_first}")

                print(f"    Preference Term (batch mean): {preference_term.mean().item():.4f}")
                print(f"    SFT Term (batch mean): {sft_term.mean().item():.4f}")
                print(f"    Total Loss (batch mean): {loss_actor.item():.4f}")

            if wandb.run:
                wandb.log({
                    "train/orpo_loss_batch": loss_actor.item(),
                    "train/orpo_preference_term_batch": preference_term.mean().item(),
                    "train/orpo_sft_term_batch": sft_term.mean().item(),
                    "orpo_epoch_train": epoch_idx + 1,
                    "orpo_batch_train": batch_idx + 1
                })

            # --- Periodic Evaluation ---
            if eval_config_batches > 0 and (batch_idx + 1) % eval_config_batches == 0:
                current_eval_score = run_evaluation_epoch(
                    actor_model, model_llm_train, tokenizer_llm_train, pruner_for_eval,
                    eval_dataset_llm, tokenizer_actor_train, num_total_llm_layers_train,
                    max_seq_len_actor_train, current_device, epoch_idx + 1, batch_idx + 1, verbose
                )
                if current_eval_score > best_eval_score:
                    best_eval_score = current_eval_score
                    print(f"New best evaluation score: {best_eval_score:.4f}. Saving actor model.")
                    torch.save(actor_model.state_dict(), "best_orpo_actor_model.pth")
                actor_model.train()

        avg_epoch_loss_actor = total_loss_epoch_actor / processed_batches_actor if processed_batches_actor > 0 else 0
        print(f"--- ORPO Actor Epoch {epoch_idx+1} finished. Average Loss: {avg_epoch_loss_actor:.4f} ---")
        if wandb.run:
            wandb.log({"train/orpo_loss_epoch": avg_epoch_loss_actor, "orpo_epoch_train_end": epoch_idx + 1})
        
        # --- End of Epoch Evaluation (if not done per batch) ---
        if eval_config_batches == 0: # Eval at end of epoch
            current_eval_score = run_evaluation_epoch(
                actor_model, model_llm_train, tokenizer_llm_train, pruner_for_eval,
                eval_dataset_llm, tokenizer_actor_train, num_total_llm_layers_train,
                max_seq_len_actor_train, current_device, epoch_idx + 1, len(dataloader_actor), verbose
            )
            if current_eval_score > best_eval_score:
                best_eval_score = current_eval_score
                print(f"New best evaluation score: {best_eval_score:.4f}. Saving actor model.")
                os.makedirs(checkpoints_path, exist_ok=True)
                torch.save(actor_model.state_dict(), os.path.join(checkpoints_path, f"epoch_{epoch_idx+1}"))
            actor_model.train()


    pruner_for_eval._restore_original_layers() # Final cleanup
    print("ORPO Actor Training finished.")
    return actor_model
