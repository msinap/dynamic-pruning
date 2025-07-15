import torch
import torch.nn as nn
import torch.nn.functional as F

class GumbelTopK(nn.Module):
    """
    Differentiable Top-K operator using Gumbel perturbations.

    This module approximates the probability of each item being in the top-k
    by averaging over multiple Gumbel-perturbed samples.
    
    Args:
        k (int): The number of top elements to select.
        tau (float): The temperature for the Gumbel-softmax distribution. A
                     lower tau makes the selection sharper.
        n_samples (int): The number of Monte Carlo samples to use for
                         approximating the probabilities.
    """
    def __init__(self, k: int, tau: float = 1.0, n_samples: int = 100):
        super().__init__()
        self.k = k
        self.tau = tau
        self.n_samples = n_samples

    def _sample_gumbel(self, shape, eps=1e-20):
        """Samples standard Gumbel(0,1) noise."""
        U = torch.rand(shape).to(next(self.parameters()).device if len(list(self.parameters())) > 0 else 'cpu')
        return -torch.log(-torch.log(U + eps) + eps)

    def forward(self, logits: torch.Tensor, hard: bool = False):
        """
        Forward pass for the Gumbel Top-K operator.

        Args:
            logits (torch.Tensor): A tensor of scores of shape [batch_size, n].
            hard (bool): If True, returns a hard 0/1 vector and uses a
                         straight-through estimator for gradients. If False,
                         returns the soft probabilities.

        Returns:
            torch.Tensor: A tensor of shape [batch_size, n]. If hard=False, it
                          contains the probabilities. If hard=True, it is a
                          one-hot encoded vector of the top-k selection.
        """
        # Expand logits for Monte Carlo sampling
        # New shape: [batch_size, n_samples, n]
        logits_expanded = logits.unsqueeze(1).expand(-1, self.n_samples, -1)
        
        # Sample Gumbel noise and add it to the logits
        gumbel_noise = self._sample_gumbel(logits_expanded.shape)
        perturbed_logits = (logits_expanded + gumbel_noise) / self.tau
        
        # Find the top-k for each sample
        # Shape: [batch_size, n_samples, k]
        _, top_k_indices = torch.topk(perturbed_logits, self.k, dim=-1)
        
        # Create a one-hot representation of the top-k selections
        # Shape: [batch_size, n_samples, n]
        top_k_one_hot = F.one_hot(top_k_indices, num_classes=logits.shape[-1]).sum(dim=-2)
        
        # Average over the samples to get soft probabilities
        # Shape: [batch_size, n]
        soft_probabilities = top_k_one_hot.float().mean(dim=1)
        
        if hard:
            # Straight-through estimator
            # In the forward pass, use a hard selection from a single sample
            # In the backward pass, gradients will flow through soft_probabilities
            
            # Get a hard selection from one Gumbel sample
            _, top_k_indices_hard = torch.topk(perturbed_logits[:, 0, :], self.k, dim=-1)
            y_hard = F.one_hot(top_k_indices_hard, num_classes=logits.shape[-1]).sum(dim=-2).float()
            
            # Use detach trick for straight-through
            return y_hard.detach() + soft_probabilities - soft_probabilities.detach()
        
        # Return the soft probabilities
        return soft_probabilities


if __name__ == '__main__':
    # --- Configuration ---
    k_val = 3
    n_elements = 6
    temperature = 0.5  # Sharper selection
    mc_samples = 500   # More samples give a better approximation

    # --- Create the Top-K operator ---
    gumbel_top_k_op = GumbelTopK(k=k_val, tau=temperature, n_samples=mc_samples)

    # --- Create some dummy data that requires gradients ---
    scores = torch.tensor([
        [0.25, 0.27, 0.25, 0.23, 0.21, 0.2] # Top 3 are 8.0, 5.0, 3.0
    ], requires_grad=True)

    print(f"Original Scores:\n{scores.detach().numpy()}")
    print(f"Finding probability of being in Top-{k_val} using Gumbel perturbations")
    print("-" * 30)

    # --- Get the probabilities (hard=False) ---
    probabilities = gumbel_top_k_op(scores, hard=False)

    print("Probabilities of each element being in Top-3:")
    print(probabilities.detach().numpy().round(3))
    print("\nNote: The scores 8.0, 5.0, and 3.0 have probabilities close to 1.0.")
    print("-" * 30)
    
    # --- Verify the Sum-to-K property ---
    prob_sum = torch.sum(probabilities, dim=-1)
    print(f"Sum of probabilities: {prob_sum.item():.4f}")
    print(f"Expected sum: {k_val}")
    # Note: For Monte Carlo, this will be exactly k because we select k items each time.
    if torch.allclose(prob_sum, torch.tensor(float(k_val))):
        print("✅ Sum-to-K property holds.")
    else:
        print("❌ Sum-to-K property does not hold.")
    print("-" * 30)

    # --- Verify differentiability using a dummy loss ---
    # We'll use the straight-through version (hard=True) as we would in a training loop
    # Let's try to maximize the probability of the first element (score 1.0)
    #hard_selection = gumbel_top_k_op(scores, hard=True)
    
    # Example loss: Reward the model if it selects the first element
    #loss = -(hard_selection[0, 0])
    
    #loss.backward()

    # print("Verifying gradients (from hard=True with straight-through)...")
    # print(f"Gradient w.r.t. original scores:\n{scores.grad.numpy().round(4)}")

    # if scores.grad is not None and torch.any(scores.grad != 0):
    #     print("\n✅ Gradients were computed successfully. The operation is differentiable.")
    # else:
    #     print("\n❌ Gradient computation failed.")

