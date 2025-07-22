from typing import List, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm
import numpy as np

from src.llm import generate_llm_output_with_pruning
from src.prune import LLMPruner
from src.router import Router
from src.evaluation import evaluate_model_on_dataset, partial_match_score, ratio_function_calls_score, json_match_score


class DifferentiableTopK(nn.Module):
    """
    Differentiable Top-K operator using the Sinkhorn algorithm.
    This module takes a batch of 1-D scores and returns the log probability
    for each element of being in the top-k.

    Args:
        k (int): The number of top elements to consider.
        epsilon (float): The regularization parameter for the Sinkhorn algorithm.
        n_iters (int): The number of Sinkhorn iterations.
    """
    def __init__(self, k: int, epsilon: float = 1e-3, n_iters: int = 2):
        super().__init__()
        self.k = k
        self.epsilon = epsilon
        self.n_iters = n_iters

    def forward(self, scores: torch.Tensor):
        """
        Forward pass for the differentiable top-k operator.

        Args:
            scores (torch.Tensor): A tensor of scores of shape [batch_size, n].

        Returns:
            torch.Tensor: A tensor of log probabilities of shape [batch_size, n],
                          where each element log_p_i is the log probability of score_i
                          being in the top-k.
        """
        # Ensure scores is 2D for batch processing
        if scores.dim() == 1:
            scores = scores.unsqueeze(0)

        batch_size, n = scores.shape
        if self.k > n:
            raise ValueError(f"k ({self.k}) cannot be larger than n ({n})")

        # --- Sinkhorn sorting to get the soft permutation matrix ---
        scores_sorted, _ = torch.sort(scores, descending=True, dim=-1)
        cost_matrix = (scores.unsqueeze(2) - scores_sorted.unsqueeze(1))**2
        log_P = -cost_matrix / self.epsilon

        for _ in range(self.n_iters):
            log_P = log_P - torch.logsumexp(log_P, dim=-2, keepdim=True)
            log_P = log_P - torch.logsumexp(log_P, dim=-1, keepdim=True)
        
        # --- Sum log probabilities for top-k ranks ---
        # The log probability of element `i` being in the top-k is the log-sum-exp of its
        # log probabilities of being in rank 0, 1, ..., k-1.
        # These correspond to the first k columns of the log permutation matrix.
        top_k_log_probs = torch.logsumexp(log_P[:, :, :self.k], dim=-1)
        return top_k_log_probs


def log1mexp(x: torch.Tensor) -> torch.Tensor:
    """Numerically stable log(1-exp(x)) for x < 0"""
    # For x < -log(2), use log1p(-exp(x))  => numerically stable
    # For x >= -log(2), use log(-expm1(x))  => more accurate
    mask = x < -torch.log(torch.tensor(2.0, device=x.device))
    result = torch.zeros_like(x)
    result[mask] = torch.log1p(-torch.exp(x[mask]))
    result[~mask] = torch.log(-torch.expm1(x[~mask]))
    return result


def calculate_log_pi(
    pruned_layers_batch: List[List[int]],
    layers_scores: torch.Tensor,
    mu_ratio: torch.Tensor,
    log_std_ratio: torch.Tensor,
    num_llm_layers: int,
    verbose: bool = False,
) -> torch.Tensor:
    """
    Calculates the log probability of a batch of pruning actions (log_pi(y|x)).

    The probability of an action `y` (a set of pruned layers) is the product of two factors:
    1. P(k | x): The probability of pruning exactly `k` layers.
    2. P(layers | k, x): The probability of choosing that specific set of layers, given k.

    log_pi(y|x) = log(P(k|x)) + log(P(layers|k,x))

    Args:
        pruned_layers_batch (List[List[int]]): A batch of pruned layer sets.
        layers_scores (torch.Tensor): Scores for each layer from the router.
        mu_ratio (torch.Tensor): Mean of the ratio distribution from the router.
        log_std_ratio (torch.Tensor): Log std dev of the ratio distribution from the router.
        num_llm_layers (int): The total number of prunable layers in the LLM.

    Returns:
        torch.Tensor: A tensor of shape (batch_size, 1) containing the log probability for each action.
    """
    batch_size = layers_scores.shape[0]
    device = layers_scores.device

    # --- Part 1: Calculate log P(k | x) ---
    # `k` is the number of layers pruned for each item in the batch.
    k_batch = torch.tensor([len(p) for p in pruned_layers_batch], device=device, dtype=torch.float32).unsqueeze(1)

    # Create the Normal distribution for the pruning ratio
    std_ratio = torch.exp(log_std_ratio)
    ratio_dist = torch.distributions.Normal(mu_ratio, std_ratio)

    # To get P(k), we find the probability that the continuous ratio `r` falls
    # into the bucket that corresponds to the integer `k`.
    # A ratio `r` results in `k` pruned layers if `k/N <= r < (k+1)/N`.
    # lower_bound = k_batch / num_total_layers
    # upper_bound = (k_batch + 1) / num_total_layers

    # Use the CDF to find the probability of the ratio falling in the range.
    # P(a < X < b) = CDF(b) - CDF(a)
    # prob_k = ratio_dist.cdf(upper_bound) - ratio_dist.cdf(lower_bound)
    # print("prob_k", prob_k)

    # calculate the log probability of the ratio
    log_prob_k = ratio_dist.log_prob(k_batch / num_llm_layers)

    if verbose:
        print("k_batch", k_batch)
        print("mu_ratio", mu_ratio)
        print("std_ratio", std_ratio)
        print("log_prob_k", log_prob_k)

    # --- Part 2: Calculate log P(layers | k, x) using DifferentiableTopK ---
    # We use the DifferentiableTopK to get the probability of each layer being in the top-k
    # and then calculate the log probability of the specific set of layers being selected.
    log_prob_layers_list = []
    for i in range(batch_size):
        k_i = len(pruned_layers_batch[i])
        if k_i == 0:
            # If no layers are pruned, probability is 1 (log probability is 0)
            log_prob_layers_list.append(torch.tensor(0.0, device=device))
            continue
            
        # Create DifferentiableTopK for this specific k
        differentiable_topk = DifferentiableTopK(k=k_i, epsilon=1e-1, n_iters=2)
        
        # Get the scores for this sample and apply DifferentiableTopK
        # layers_scores[i] has shape [num_total_layers]
        sample_scores = layers_scores[i].unsqueeze(0)  # Add batch dimension    
        top_k_log_probs = differentiable_topk(sample_scores)  # Shape: [1, num_total_layers]
    
        # Get the indices of the pruned layers for the current sample
        indices = torch.tensor(pruned_layers_batch[i], device=device, dtype=torch.long)
        
        # Gather the log probabilities for those specific layers
        log_probs_for_pruned_layers = top_k_log_probs[0][indices]  # Shape: [k_i]
        # add not pruned layers to the log probability (log(1-p) = log(1-exp(log_p)))
        # Use numerically stable log1mexp function
        
        # Create a mask for all layers that are NOT pruned
        all_indices = torch.arange(top_k_log_probs.shape[1], device=device)
        not_pruned_mask = ~torch.isin(all_indices, indices)
        log_probs_for_not_pruned_layers = log1mexp(top_k_log_probs[0][not_pruned_mask])
        log_probs_for_layers = torch.cat([log_probs_for_pruned_layers, log_probs_for_not_pruned_layers], dim=0)
        
        # Clamp the log probabilities to ensure stability (no value less than -10)
        log_probs_for_layers = torch.clamp(log_probs_for_layers, min=-10.0)

        # Calculate the log probability of selecting this specific set of layers
        # We use the product of individual probabilities (sum of log probabilities)
        # log_probs_for_layers = torch.log(probs_for_layers + 1e-4)  # Add epsilon for numerical stability
        summed_log_probs = log_probs_for_layers.sum()
        log_prob_layers_list.append(summed_log_probs)

        if verbose:
            print("sample_scores", sample_scores)
            print("top_k_log_probs", top_k_log_probs)
            print("log_probs_for_pruned_layers", log_probs_for_pruned_layers)
            print("log_probs_for_not_pruned_layers", log_probs_for_not_pruned_layers)
            print("log_probs_for_layers", log_probs_for_layers)

    log_prob_layers = torch.stack(log_prob_layers_list).unsqueeze(1)
    
    # --- Part 3: Combine ---
    # Total log probability is the sum of the two components.
    total_log_prob = log_prob_k + log_prob_layers
    if verbose:
        print("log_prob_layers", log_prob_layers)
        print("log_prob_k", log_prob_k)
        print("total_log_prob", total_log_prob)
    
    return total_log_prob


def dpo_loss_function(
    batch: Dict[str, Any],
    router: Router,
    verbose: bool = False,
) -> torch.Tensor:
    """
    Computes the DPO loss for a batch of preference data.
    Assumes the batch is already tokenized and contains tensors.
    """
    layers_scores, mu_ratio, log_std_ratio = router(
        input_ids=batch['input_ids'].to(router.bert.device),
        attention_mask=batch['attention_mask'].to(router.bert.device),
    )
    
    log_pi_winner = calculate_log_pi(
        batch['winner_layers'], layers_scores, mu_ratio, log_std_ratio, router.num_llm_layers, verbose=verbose,
    )
    log_pi_loser = calculate_log_pi(
        batch['loser_layers'], layers_scores, mu_ratio, log_std_ratio, router.num_llm_layers, verbose=verbose,
    )
    
    log_probs_diff = log_pi_winner - log_pi_loser
    loss = -F.logsigmoid(log_probs_diff).mean()
    if verbose:
        print("log_pi_winner", log_pi_winner)
        print("log_pi_loser", log_pi_loser)
        print("log_probs_diff", log_probs_diff)
        print("loss", loss.item())
    return loss

    
def orpo_loss_function(
    batch: Dict[str, Any],
    router: Router,
    *,
    orpo_alpha: float,
    fbc_alpha: float,
    verbose: bool = False,
) -> torch.Tensor:
    """
    Computes the ORPO loss for a batch of preference data.
    ORPO combines a preference loss based on odds ratio with an SFT loss on the winning examples.
    """
    layers_scores, mu_ratio, log_std_ratio = router(
        input_ids=batch['input_ids'].to(router.bert.device),
        attention_mask=batch['attention_mask'].to(router.bert.device),
    )

    # Calculate log probabilities for winner and loser actions
    log_pi_winner = calculate_log_pi(
        batch['winner_layers'], layers_scores, mu_ratio, log_std_ratio, router.num_llm_layers, verbose=verbose,
    )
    log_pi_loser = calculate_log_pi(
        batch['loser_layers'], layers_scores, mu_ratio, log_std_ratio, router.num_llm_layers, verbose=verbose,
    )

    # --- Odds Ratio Preference Loss Part ---
    # log_odds = log(p / (1-p)) = log(p) - log(1-p)
    # Here, p = exp(log_pi), so log(1-p) = log(1-exp(log_pi)) = log1mexp(log_pi)
    log_odds_winner = log_pi_winner - log1mexp(log_pi_winner)
    log_odds_loser = log_pi_loser - log1mexp(log_pi_loser)

    log_odds_ratio = log_odds_winner - log_odds_loser
    preference_loss = -F.logsigmoid(log_odds_ratio).mean()

    # --- SFT Loss Part (on winner data, inspired by fbc_loss_function) ---
    device = layers_scores.device
    num_llm_layers = router.num_llm_layers
    batch_size = layers_scores.shape[0]

    # Construct tensors for winner ratio and layer pruning
    winner_ratios = torch.tensor(
        [len(p) / num_llm_layers for p in batch['winner_layers']], 
        device=device, 
        dtype=mu_ratio.dtype
    ).unsqueeze(1)
    
    winner_is_layer_pruned_list = []
    for i in range(batch_size):
        is_pruned_tensor = torch.zeros(num_llm_layers, device=device, dtype=layers_scores.dtype)
        if batch['winner_layers'][i]:
            pruned_indices = torch.tensor(batch['winner_layers'][i], device=device, dtype=torch.long)
            is_pruned_tensor.scatter_(0, pruned_indices, 1.0)
        winner_is_layer_pruned_list.append(is_pruned_tensor)
    winner_is_layer_pruned = torch.stack(winner_is_layer_pruned_list)

    # Calculate MSE losses for SFT component
    mu_loss = F.mse_loss(mu_ratio, winner_ratios)
    layers_scores_loss = F.mse_loss(layers_scores, winner_is_layer_pruned)
    sft_loss = mu_loss + layers_scores_loss * fbc_alpha

    # --- Combine Preference and SFT losses ---
    loss = preference_loss + orpo_alpha * sft_loss

    if verbose:
        print("preference_loss", preference_loss.item())
        print("sft_loss", sft_loss.item())
        print("total orpo_loss", loss.item())
        
    return loss


def train_router_with_preference_optimization(
    router,
    router_tokenizer,

    # training
    learning_rate: float,
    train_dataloader: DataLoader,
    log_every_n_steps: int,
    loss_fn,
    loss_fn_kwargs: Dict[str, Any],

    # evaluation
    eval_dataset: List[Dict[str, Any]],
    eval_every_n_steps: int,
    llm_model,
    llm_tokenizer,
    adapters,
    score_funcs,
):
    optimizer = AdamW(router.parameters(), lr=learning_rate)

    router.train()
    for step, batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
        optimizer.zero_grad()
        verbose = True if step % log_every_n_steps == log_every_n_steps - 1 else False
        loss = loss_fn(batch, router, verbose=verbose, **loss_fn_kwargs)
        loss.backward()
        optimizer.step()

        if step % eval_every_n_steps == eval_every_n_steps - 1:
            scores = {}
            for sample in eval_dataset:
                output = generate_llm_output_with_pruning(
                    sample=sample,
                    model_llm=llm_model,
                    tokenizer_llm=llm_tokenizer,
                    adapters=adapters,
                    router=router,
                    tokenizer_router=router_tokenizer,
                    verbose=True,
                )
                print(output)
                print(sample['answers'])
                for score_func in score_funcs:
                    if score_func.__name__ not in scores:
                        scores[score_func.__name__] = []
                    score = score_func(output, sample['answers'])
                    scores[score_func.__name__].append(score)
                    print(f"{score_func.__name__}: {score:.4f}")
                
            for score_name, results in scores.items():
                print(f"{score_name}: {np.mean(results):.4f}")
            print("-" * 100)

