import json
import torch
from tqdm import tqdm
import wandb
from src.llm import generate_llm_output
from src.actor import get_pruning_action_from_actor
import time # Added for timing


def calc_score(output_str, answer_str):
    try:
        output_json = json.loads(output_str)
        parsed_output = {}
        if not isinstance(output_json, list): # Handle if output is not a list of calls
            return -5 # Malformed output
        for func_call in output_json:
            if not isinstance(func_call, dict) or 'name' not in func_call or 'arguments' not in func_call:
                return -5 # Malformed function call
            if not isinstance(func_call['arguments'], dict): # Arguments must be a dict
                 return -5
            parsed_output[func_call['name']] = func_call['arguments']
    except json.JSONDecodeError:
        return -5 # Cannot parse output
    except TypeError: # Other parsing issues
        return -5

    answer_json = json.loads(answer_str)
    if len(answer_json) == 0:
        return 10 if len(parsed_output) == 0 else 0

    score = 0.0
    for expected_func_call in answer_json:
        single_call_max_score = 10.0 / len(answer_json)
        if expected_func_call['name'] not in parsed_output: # Function name mismatch or missing
            continue # No points for this expected call

        score += single_call_max_score / 2.0 # Half points for correct function name
        
        # Argument matching
        expected_args = expected_func_call['arguments']
        predicted_args = parsed_output[expected_func_call['name']]
        
        if not expected_args: # If no arguments expected and name matches
            if not predicted_args: # and no arguments predicted
                 score += single_call_max_score / 2.0
            continue


        num_expected_args = len(expected_args)
        if num_expected_args == 0: continue # Already handled if name matched

        arg_score_per_item = (single_call_max_score / 2.0) / num_expected_args
        
        for arg_name, arg_value in expected_args.items():
            if arg_name not in predicted_args:
                # score -= arg_score_per_item / 3.0 # Penalty for missing argument (optional, can be harsh)
                pass
            elif predicted_args[arg_name] == arg_value:
                score += arg_score_per_item # Full point for this arg
            # else: # Argument value mismatch (optional: partial credit or penalty)
                # score += arg_score_per_item / 3.0 # Partial for key match, value mismatch (optional)

    return max(0, score) # Ensure score is not negative from penalties if any


def run_evaluation_epoch(
    actor_model_eval, model_llm_eval, tokenizer_llm_eval, pruner_llm,
    dataset_llm_eval, tokenizer_actor_eval, num_total_llm_layers,
    max_seq_len_actor_eval, current_device, epoch, batch_num_actor, verbose
    ):
    print(f"\n--- Running Evaluation after Epoch {epoch}, Actor Batch {batch_num_actor} ---")
    actor_model_eval.eval()
    model_llm_eval.eval() # Ensure LLM is in eval mode

    total_eval_score = 0.0
    total_k_pruned = 0
    total_mu_ratio = 0.0
    num_eval_samples_processed = 0
    total_unpruned_score = 0.0
    unpruned_better_count = 0

    # Performance metrics accumulators
    total_latency_unpruned = 0.0
    total_tokens_generated_unpruned = 0
    total_latency_pruned = 0.0
    total_tokens_generated_pruned = 0


    for i in tqdm(range(len(dataset_llm_eval)), desc="Eval"):
        sample = dataset_llm_eval[i]
        
        # Actor input: Use same format as preference data
        # actor_input_text_eval = generate_prompt_str(sample, tokenizer_llm_eval) # Option 1: LLM prompt style
        actor_input_text_eval = sample['tools'] + '\n' + sample['query'] # Option 2: Preference data style

        # --- Evaluate Unpruned LLM (Baseline) ---
        # Ensure no layers are pruned before this
        pruner_llm._restore_original_layers() # Make sure we are starting from a clean slate
        
        time_start_unpruned = time.perf_counter()
        llm_generated_output_unpruned, tokens_generated_unpruned = generate_llm_output(sample, model_llm_eval, tokenizer_llm_eval, return_token_count=True)
        latency_unpruned = time.perf_counter() - time_start_unpruned
        total_latency_unpruned += latency_unpruned
        total_tokens_generated_unpruned += tokens_generated_unpruned
        
        score_unpruned = calc_score(llm_generated_output_unpruned, sample['answers'])
        total_unpruned_score += score_unpruned
        # --- End Unpruned LLM Evaluation ---

        pruned_layer_indices, k_val, mu_ratio_val = get_pruning_action_from_actor(
            actor_model_eval, actor_input_text_eval, tokenizer_actor_eval,
            num_total_llm_layers, max_seq_len_actor_eval, current_device
        )
        
        pruner_llm.prune_model(pruned_layer_indices)
        
        time_start_pruned = time.perf_counter()
        llm_generated_output_pruned, tokens_generated_pruned_sample = generate_llm_output(sample, model_llm_eval, tokenizer_llm_eval, return_token_count=True)
        latency_pruned = time.perf_counter() - time_start_pruned
        total_latency_pruned += latency_pruned
        total_tokens_generated_pruned += tokens_generated_pruned_sample
        
        pruner_llm._restore_original_layers() # Important to restore for next sample / training

        score_pruned = calc_score(llm_generated_output_pruned, sample['answers'])

        if score_unpruned > score_pruned - 1e-4:
            unpruned_better_count += 1

        if verbose:
            print(f"  Sample {i+1}:")
            print(f"    Actor Input: '{actor_input_text_eval[:200]}...'")
            print(f"    Pruned Indices: {pruned_layer_indices}")
            print(f"    k_val (Num Pruned): {k_val}")
            print(f"    mu_ratio_val (Predicted Ratio): {mu_ratio_val:.4f}")
            print(f"    LLM Output (Pruned): '{llm_generated_output_pruned[:200]}...'")
            print(f"    Tokens Gen (Pruned): {tokens_generated_pruned_sample}, Latency (Pruned): {latency_pruned:.4f}s")
            print(f"    Score (Pruned): {score_pruned:.4f}")
            print(f"    LLM Output (Unpruned): '{llm_generated_output_unpruned[:200]}...'")
            print(f"    Tokens Gen (Unpruned): {tokens_generated_unpruned}, Latency (Unpruned): {latency_unpruned:.4f}s")
            print(f"    Score (Unpruned): {score_unpruned:.4f}")
            print(f"    Sample Answer: '{sample['answers']}'")


        total_eval_score += score_pruned # This is now specifically for the pruned model
        total_k_pruned += k_val
        total_mu_ratio += mu_ratio_val
        num_eval_samples_processed += 1

    avg_eval_score = total_eval_score / num_eval_samples_processed if num_eval_samples_processed > 0 else 0
    avg_k_pruned = total_k_pruned / num_eval_samples_processed if num_eval_samples_processed > 0 else 0
    avg_mu_ratio = total_mu_ratio / num_eval_samples_processed if num_eval_samples_processed > 0 else 0
    avg_unpruned_score = total_unpruned_score / num_eval_samples_processed if num_eval_samples_processed > 0 else 0

    # Calculate average performance metrics
    avg_latency_unpruned = total_latency_unpruned / num_eval_samples_processed if num_eval_samples_processed > 0 else 0
    avg_tokens_unpruned = total_tokens_generated_unpruned / num_eval_samples_processed if num_eval_samples_processed > 0 else 0
    time_per_token_unpruned = total_latency_unpruned / total_tokens_generated_unpruned if total_tokens_generated_unpruned > 0 else 0
    tokens_per_sec_unpruned = total_tokens_generated_unpruned / total_latency_unpruned if total_latency_unpruned > 0 else 0

    avg_latency_pruned = total_latency_pruned / num_eval_samples_processed if num_eval_samples_processed > 0 else 0
    avg_tokens_pruned = total_tokens_generated_pruned / num_eval_samples_processed if num_eval_samples_processed > 0 else 0
    time_per_token_pruned = total_latency_pruned / total_tokens_generated_pruned if total_tokens_generated_pruned > 0 else 0
    tokens_per_sec_pruned = total_tokens_generated_pruned / total_latency_pruned if total_latency_pruned > 0 else 0
    
    speedup_factor = avg_latency_unpruned / avg_latency_pruned if avg_latency_pruned > 0 else float('inf')


    print(f"Avg Pruned Evaluation Score: {avg_eval_score:.4f}")
    print(f"Avg Unpruned Evaluation Score: {avg_unpruned_score:.4f}")
    print(f"Unpruned model was better {unpruned_better_count} times out of {num_eval_samples_processed} samples.")
    print(f"Avg k Pruned: {avg_k_pruned:.2f}, Avg Pred. Mu Ratio: {avg_mu_ratio:.4f}")
    
    print(f"--- Performance Metrics ---")
    print(f"Unpruned Model:")
    print(f"  Avg Latency: {avg_latency_unpruned:.4f}s")
    print(f"  Avg Tokens Generated: {avg_tokens_unpruned:.2f}")
    print(f"  Time per Token: {time_per_token_unpruned:.4f}s/token")
    print(f"  Tokens per Second: {tokens_per_sec_unpruned:.2f} tokens/s")
    print(f"Pruned Model:")
    print(f"  Avg Latency: {avg_latency_pruned:.4f}s")
    print(f"  Avg Tokens Generated: {avg_tokens_pruned:.2f}")
    print(f"  Time per Token: {time_per_token_pruned:.4f}s/token")
    print(f"  Tokens per Second: {tokens_per_sec_pruned:.2f} tokens/s")
    print(f"Speedup (Unpruned Latency / Pruned Latency): {speedup_factor:.2f}x")


    if wandb.run:
        wandb.log({
            "eval/avg_pruned_score": avg_eval_score,
            "eval/avg_unpruned_score": avg_unpruned_score,
            "eval/unpruned_better_ratio": (unpruned_better_count / num_eval_samples_processed) if num_eval_samples_processed > 0 else 0.0,
            "eval/avg_k_pruned": avg_k_pruned,
            "eval/avg_mu_ratio": avg_mu_ratio,
            "eval/latency_unpruned_avg_s": avg_latency_unpruned,
            "eval/tokens_generated_unpruned_avg": avg_tokens_unpruned,
            "eval/time_per_token_unpruned_s": time_per_token_unpruned,
            "eval/tokens_per_sec_unpruned": tokens_per_sec_unpruned,
            "eval/latency_pruned_avg_s": avg_latency_pruned,
            "eval/tokens_generated_pruned_avg": avg_tokens_pruned,
            "eval/time_per_token_pruned_s": time_per_token_pruned,
            "eval/tokens_per_sec_pruned": tokens_per_sec_pruned,
            "eval/speedup_factor": speedup_factor,
            "epoch_eval": epoch, # Distinguish from training epoch
            "batch_eval": batch_num_actor
        })
    
    actor_model_eval.train() # Set actor back to train mode
    return avg_eval_score
