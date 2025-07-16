from typing import List, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm

from src.router import Router


class DifferentiableTopK(nn.Module):
    """
    Differentiable Top-K operator using the Sinkhorn algorithm.
    This module takes a batch of 1-D scores and returns the soft probability
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
            torch.Tensor: A tensor of probabilities of shape [batch_size, n],
                          where each element p_i is the probability of score_i
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
        
        soft_permutation = torch.exp(log_P)
        
        # --- Sum probabilities for top-k ranks ---
        # The probability of element `i` being in the top-k is the sum of its
        # probabilities of being in rank 0, 1, ..., k-1.
        # These correspond to the first k columns of the soft permutation matrix.
        top_k_probs = torch.sum(soft_permutation[:, :, :self.k], dim=-1)

        return top_k_probs


def calculate_log_pi(
    pruned_layers_batch: List[List[int]],
    layers_scores: torch.Tensor,
    mu_ratio: torch.Tensor,
    log_std_ratio: torch.Tensor,
    num_llm_layers: int
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
    print("k_batch", k_batch)

    # Create the Normal distribution for the pruning ratio
    std_ratio = torch.exp(log_std_ratio)
    ratio_dist = torch.distributions.Normal(mu_ratio, std_ratio)
    print("mu_ratio", mu_ratio)
    print("std_ratio", std_ratio)

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
        differentiable_topk = DifferentiableTopK(k=k_i, epsilon=1e-2, n_iters=2)
        
        # Get the scores for this sample and apply DifferentiableTopK
        # layers_scores[i] has shape [num_total_layers]
        sample_scores = layers_scores[i].unsqueeze(0)  # Add batch dimension
        top_k_probs = differentiable_topk(sample_scores)  # Shape: [1, num_total_layers]
        
        # Get the indices of the pruned layers for the current sample
        indices = torch.tensor(pruned_layers_batch[i], device=device, dtype=torch.long)
        
        # Gather the probabilities for those specific layers
        probs_for_pruned_layers = top_k_probs[0][indices]  # Shape: [k_i]
        # add not pruned layers to the probability
        probs_for_not_pruned_layers = 1.0 - top_k_probs[0][~indices]
        probs_for_layers = torch.cat([probs_for_pruned_layers, probs_for_not_pruned_layers], dim=0)
        
        print("probs_for_layers", probs_for_layers)

        # Calculate the log probability of selecting this specific set of layers
        # We use the product of individual probabilities (sum of log probabilities)
        log_probs_for_layers = torch.log(probs_for_layers + 1e-4)  # Add epsilon for numerical stability
        summed_log_probs = log_probs_for_layers.sum()
        log_prob_layers_list.append(summed_log_probs)

    log_prob_layers = torch.stack(log_prob_layers_list).unsqueeze(1)
    print("log_prob_layers", log_prob_layers)
    
    # --- Part 3: Combine ---
    # Total log probability is the sum of the two components.
    total_log_prob = log_prob_k + log_prob_layers
    
    return total_log_prob


def dpo_loss_function(
    batch: Dict[str, Any],
    router: Router,
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
        batch['winner_layers'], layers_scores, mu_ratio, log_std_ratio, router.num_llm_layers
    )
    log_pi_loser = calculate_log_pi(
        batch['loser_layers'], layers_scores, mu_ratio, log_std_ratio, router.num_llm_layers
    )

    print("log_pi_winner", log_pi_winner)
    print("log_pi_loser", log_pi_loser)
    
    log_probs_diff = log_pi_winner - log_pi_loser
    loss = -F.logsigmoid(log_probs_diff).mean()
    return loss

    
def train_router_with_dpo(
    router: Router,
    train_dataloader: DataLoader,
    learning_rate: float = 1e-6,
    epochs: int = 3,
):
    optimizer = AdamW(router.parameters(), lr=learning_rate)

    router.train()
    for epoch in tqdm(range(epochs), desc="Epochs"):
        total_loss = 0
        for step, batch in enumerate(train_dataloader):
            optimizer.zero_grad()

            # Compute DPO loss
            loss = dpo_loss_function(batch, router)

            # Backpropagation
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            
            print(f"Step {step + 1}/{len(train_dataloader)}, Loss: {loss.item():.4f}")

            if step == 50:
                exit()

        avg_loss = total_loss / len(train_dataloader)
        print(f"Epoch {epoch + 1} finished. Average Loss: {avg_loss:.4f}")

