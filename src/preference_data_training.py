from typing import List, Dict, Any
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm

from src.router import Router

def calculate_log_pi(
    pruned_layers_batch: List[List[int]],
    layers_log_probs: torch.Tensor,
    mu_ratio: torch.Tensor,
    log_std_ratio: torch.Tensor,
    num_total_layers: int
) -> torch.Tensor:
    """
    Calculates the log probability of a batch of pruning actions (log_pi(y|x)).

    The probability of an action `y` (a set of pruned layers) is the product of two factors:
    1. P(k | x): The probability of pruning exactly `k` layers.
    2. P(layers | k, x): The probability of choosing that specific set of layers, given k.

    log_pi(y|x) = log(P(k|x)) + log(P(layers|k,x))

    Args:
        pruned_layers_batch (List[List[int]]): A batch of pruned layer sets.
        layers_log_probs (torch.Tensor): Log probabilities for each layer from the router.
        mu_ratio (torch.Tensor): Mean of the ratio distribution from the router.
        log_std_ratio (torch.Tensor): Log std dev of the ratio distribution from the router.
        num_total_layers (int): The total number of prunable layers in the LLM.

    Returns:
        torch.Tensor: A tensor of shape (batch_size, 1) containing the log probability for each action.
    """
    batch_size = layers_log_probs.shape[0]
    device = layers_log_probs.device

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
    log_prob_k = ratio_dist.log_prob(k_batch / num_total_layers)
    print("log_prob_k", log_prob_k)
    
    # Add a small epsilon for numerical stability before taking the log.
    # log_prob_k = torch.log(prob_k + 1e-10)

    # --- Part 2: Calculate log P(layers | k, x) ---
    # We approximate the log probability of selecting a specific set of layers
    # as the sum of their individual log probabilities from the router's output.
    log_prob_layers_list = []
    for i in range(batch_size):
        # Get the indices of the pruned layers for the current sample
        indices = torch.tensor(pruned_layers_batch[i], device=device, dtype=torch.long)
        
        # Gather the log probabilities for those specific layers
        # layers_log_probs[i] has shape [num_total_layers]
        # indices has shape [k_i] where k_i is the number of pruned layers for sample i
        log_probs_for_pruned_layers = layers_log_probs[i][indices]
        
        # Sum the log probabilities for the chosen layers
        summed_log_probs = log_probs_for_pruned_layers.sum()
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
    layers_log_probs, mu_ratio, log_std_ratio = router(
        input_ids=batch['input_ids'].to(router.bert.device),
        attention_mask=batch['attention_mask'].to(router.bert.device),
    )
    num_total_layers = router.num_llm_layers_actor
    
    log_pi_winner = calculate_log_pi(
        batch['winner_layers'], layers_log_probs, mu_ratio, log_std_ratio, num_total_layers
    )
    log_pi_loser = calculate_log_pi(
        batch['loser_layers'], layers_log_probs, mu_ratio, log_std_ratio, num_total_layers
    )

    print("log_pi_winner", log_pi_winner)
    print("log_pi_loser", log_pi_loser)
    
    log_probs_diff = log_pi_winner - log_pi_loser
    loss = -F.logsigmoid(log_probs_diff).mean()
    return loss

    
def train_router_with_dpo(
    router: Router,
    train_dataloader: DataLoader,
    learning_rate: float = 1e-3,
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

            if step == 100:
                exit()

        avg_loss = total_loss / len(train_dataloader)
        print(f"Epoch {epoch + 1} finished. Average Loss: {avg_loss:.4f}")

