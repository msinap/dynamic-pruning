import torch
import torch.nn as nn

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
    def __init__(self, k: int, epsilon: float = 1e-3, n_iters: int = 100):
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

if __name__ == '__main__':
    # --- Configuration ---
    k_val = 3
    n_elements = 6
    epsilon_val = 0.01
    n_iterations = 100

    # --- Create the Top-K operator ---
    top_k_operator = DifferentiableTopK(k=k_val, epsilon=epsilon_val, n_iters=n_iterations)

    # --- Create some dummy data that requires gradients ---
    scores = torch.tensor([
        [0.25, 0.27, 0.25, 0.23, 0.21, 0.2] # Top 3 are 1.5, 1.25, 1.0
    ], requires_grad=True)

    print(f"Original Scores:\n{scores.detach().numpy()}")
    print(f"Finding probability of being in Top-{k_val}")
    print("-" * 30)

    # --- Get the probabilities ---
    probabilities = top_k_operator(scores)

    print("Probabilities of each element being in Top-3:")
    print(probabilities.detach().numpy().round(3))
    print("\nNote: The scores 8.0, 5.0, and 3.0 have probabilities close to 1.0.")
    print("-" * 30)

    # --- Verify the Sum-to-K property ---
    prob_sum = torch.sum(probabilities, dim=-1)
    print(f"Sum of probabilities: {prob_sum.item():.4f}")
    print(f"Expected sum: {k_val}")
    if torch.allclose(prob_sum, torch.tensor(float(k_val))):
        print("✅ Sum-to-K property holds.")
    else:
        print("❌ Sum-to-K property does not hold.")
    print("-" * 30)

    # --- Verify differentiability ---
    # Define a dummy loss and backpropagate to check for gradients.
    # e.g., We want to maximize the probability of the first element (score 1.0)
    loss = -probabilities[0, 0] 
    loss.backward()

    print("Verifying gradients...")
    print(f"Gradient w.r.t. original scores:\n{scores.grad.numpy().round(4)}")

    if scores.grad is not None and torch.any(scores.grad != 0):
        print("\n✅ Gradients were computed successfully. The operation is differentiable.")
    else:
        print("\n❌ Gradient computation failed.")
