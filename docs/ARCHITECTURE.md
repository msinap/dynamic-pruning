# Architecture Documentation

This document provides an in-depth technical explanation of the dynamic layer pruning system.

## Table of Contents

1. [System Overview](#system-overview)
2. [Component Details](#component-details)
3. [Training Pipeline](#training-pipeline)
4. [Inference Pipeline](#inference-pipeline)
5. [Design Decisions](#design-decisions)

## System Overview

The dynamic layer pruning system consists of three main components:

1. **Base LLM**: Pre-trained language model (e.g., xLAM-2-1b)
2. **Adapters**: Lightweight networks that replace pruned layers
3. **Router**: Decision model that selects which layers to prune

### Data Flow

```
Input Query
    ↓
Router Model → [Layer Scores, Pruning Ratio]
    ↓
Layer Selection (Top-K)
    ↓
LLM with Adapted Layers → Output
```

## Component Details

### 1. Base SLM

- **Model**: Salesforce/xLAM-2-1b-fc-r (28 layers)
- **Task**: Tool calling (function calling)
- **Precision**: bfloat16 for efficiency
- **Device**: Auto-distributed across available GPUs
- **Base**: Fine-tuned from Qwen2.5-1B

```python
# Model structure (simplified)
class SLM:
    embedding: Embedding(vocab_size, hidden_dim)
    layers: List[TransformerLayer]  # 28 layers
    lm_head: Linear(hidden_dim, vocab_size)
```

### 2. LoRA Adapter Module

LoRA adapters use a bottleneck architecture to compress layer computation:

```python
class Adapter(nn.Module):
    def __init__(self, io_dim=2048, bottleneck_dim=64):
        self.down = Linear(io_dim, bottleneck_dim)  # Compression
        self.activation = ReLU()
        self.up = Linear(bottleneck_dim, io_dim)    # Expansion
        # Residual connection added in forward()
```

**Key Properties:**
- **Parameter Reduction**: ~96% fewer params than original layer
- **Near-Identity Initialization**: Starts close to identity mapping
- **Residual Connection**: Preserves gradient flow

**Training:**
- Layer-wise distillation from full model
- MSE loss between adapter output and layer output
- Only trains on samples with correct LLM outputs

### 3. Router Model

The router predicts which layers to prune for each input:

```python
class Router(nn.Module):
    def __init__(self, base_model, num_llm_layers):
        self.bert = base_model  # Pre-trained BERT
        
        # Layer prediction head
        self.layer_head = Sequential(
            Linear(bert_dim, 128),
            ReLU(),
            Linear(128, num_llm_layers),
            Sigmoid()  # Output: [0, 1] importance scores
        )
        
        # Ratio prediction head
        self.ratio_head = Sequential(
            Linear(bert_dim, 128),
            ReLU(),
            Linear(128, 2)  # mu and log_std
        )
```

**Outputs:**

1. **Layer Scores**: `[batch, num_layers]` - Prunability score for each layer
   - Higher score = safer to prune (more suitable for replacement with adapter)
   - Sigmoid ensures [0, 1] range

2. **Ratio Distribution Parameters**: `(mu, log_sigma)`
   - `mu`: Mean fraction of layers to prune (after sigmoid, in [0, 1])
   - `log_sigma`: Log std deviation (clipped to [-7, -3] for numerical stability)
   - At inference: sample r ~ Normal(mu, exp(log_sigma))
   - Sampled ratio determines how many layers to prune

**Inference Process:**

```python
# 1. Get predictions from router
scores, mu, log_std = router(input_ids, attention_mask)

# 2. Sample pruning ratio
ratio = Normal(mu, exp(log_std)).sample()
k = int(ratio * num_layers)  # Number of layers to prune

# 3. Select top-k layers by score
pruned_layers = topk(scores, k).indices

# 4. Replace selected layers with adapters
for idx in pruned_layers:
    model.layers[idx] = adapters[idx]
```

## Training Pipeline

### Phase 1: Adapter Training

**Goal**: Train adapters to approximate original layers

```
For each training sample:
    1. Generate output with full LLM (with hidden states)
    2. If output is correct:
        3. For each layer i:
            4. Collect (input, output) pairs across generation steps
            5. Train adapter_i to minimize MSE(adapter_output, layer_output)
```

**Why this works:**
- LoRA adapters see real layer inputs/outputs from full model
- Only learns from good generations (samples with correct function calls)
- Independent per-layer training (fully parallelizable across all 28 adapters)
- White-box distillation provides direct supervision from hidden states

### Phase 2: Offline Data Generation

**Goal**: Explore pruning space to collect training examples

#### DFS Strategy (Recommended)

```
For each input sample:
    scenarios = []
    
    # Start with no pruning
    current_layers = []
    current_score = evaluate(input, pruned_layers=current_layers)
    scenarios.append((current_layers, current_score))
    
    # Iteratively add layers using depth-first search
    while len(current_layers) < max_layers:
        best_layer = None
        best_score = 0
        
        # Try adding each unpruned layer
        for layer in unpruned_layers:
            test_layers = current_layers + [layer]
            score = evaluate(input, pruned_layers=test_layers)
            
            if score > threshold and score > best_score:
                best_layer = layer
                best_score = score
        
        if best_layer is None:
            break  # Can't add more layers without dropping below threshold
        
        current_layers.append(best_layer)
        scenarios.append((current_layers, best_score))
    
    return scenarios
```

This DFS approach produces higher variance data with more aggressive pruning configurations, providing richer training signal for preference learning compared to random exploration.

**Output**: Dataset of (input, pruned_layers, json_match_score) tuples where score represents JSON Match accuracy

### Phase 3: Router Training

Three methods implemented:

#### A. Filtered Behavioral Cloning (FBC)

**Approach**: Supervised learning on successful configurations

```python
# Filter to only high-quality scenarios
good_scenarios = [s for s in scenarios if s.score > 0.99]

# Loss function
def fbc_loss(batch):
    scores, mu, log_std = router(batch.input)
    
    # Predict which layers are pruned
    layer_loss = MSE(scores, batch.is_layer_pruned)
    
    # Predict pruning ratio
    ratio_loss = MSE(mu, batch.ratio)
    
    return ratio_loss + alpha * layer_loss
```

#### B. Direct Preference Optimization (DPO)

**Approach**: Learn from pairwise comparisons

```python
# Generate preference pairs
for input_id in dataset:
    scenarios = get_scenarios(input_id)
    
    for s1, s2 in combinations(scenarios, 2):
        if compare(s1, s2) == 1:  # s1 better than s2
            pairs.append({
                'input': input,
                'winner': s1.pruned_layers,
                'loser': s2.pruned_layers
            })

# Training
def dpo_loss(batch):
    scores, mu, log_std = router(batch.input)
    
    # Calculate log probabilities
    log_pi_winner = log_prob(batch.winner, scores, mu, log_std)
    log_pi_loser = log_prob(batch.loser, scores, mu, log_std)
    
    # DPO objective
    return -log_sigmoid(log_pi_winner - log_pi_loser).mean()
```

**Preference Criterion**:
```python
def compare(s1, s2):
    """
    Compare two pruning scenarios.
    Preference: (1) higher accuracy, OR (2) same accuracy but more layers pruned
    """
    if s1.score > s2.score:
        return 1  # s1 wins (better accuracy)
    elif s1.score < s2.score:
        return -1  # s2 wins (better accuracy)
    elif len(s1.pruned_layers) > len(s2.pruned_layers):
        return 1  # s1 more efficient (same accuracy, more pruning)
    elif len(s1.pruned_layers) < len(s2.pruned_layers):
        return -1  # s2 more efficient
    else:
        return 0  # tie
```

#### C. Odds Ratio Preference Optimization (ORPO)

**Approach**: DPO + supervised learning on winners

```python
def orpo_loss(batch):
    scores, mu, log_std = router(batch.input)
    
    # Preference loss (using odds ratios)
    log_pi_winner = log_prob(batch.winner, scores, mu, log_std)
    log_pi_loser = log_prob(batch.loser, scores, mu, log_std)
    
    log_odds_winner = log_pi_winner - log1mexp(log_pi_winner)
    log_odds_loser = log_pi_loser - log1mexp(log_pi_loser)
    
    preference_loss = -log_sigmoid(log_odds_winner - log_odds_loser).mean()
    
    # SFT loss on winners
    sft_loss = MSE(mu, winner_ratio) + MSE(scores, winner_layers)
    
    return preference_loss + orpo_alpha * sft_loss
```

## Inference Pipeline

1. **Tokenize input** using router tokenizer (BERT)
2. **Router forward pass** → (layer_scores, mu_ratio, log_std_ratio)
3. **Sample pruning ratio**: `r ~ Normal(mu, exp(log_std))`
4. **Calculate k**: `k = int(r * num_layers)`
5. **Select layers**: `pruned = topk(layer_scores, k)`
6. **Swap layers**: Replace `model.layers[i]` with `adapters[i]` for i in pruned
7. **Generate output** with modified model
8. **Restore layers**: Swap back to original layers

**Context Manager Pattern**:
```python
with LLMPruner(model, adapters) as pruner:
    pruner.prune_model([2, 5, 8])
    output = model.generate(...)
# Layers automatically restored here
```

## Design Decisions

### Why LoRA Adapters Instead of Layer Skipping?

**Considered**: Skip layers entirely (identity mapping)
**Chosen**: Replace with LoRA adapters

**Reasons**:
1. Layer skipping can disrupt hidden state distribution
2. LoRA adapters provide learned transformations that approximate original layers
3. Better accuracy preservation (low MSE for middle layers 2-20)
4. Minimal computational overhead (bottleneck dimension: 64, ~1-5% of layer parameters)
5. Pre-trained once, reused for all inputs (no per-query training cost)

### Why BERT for Router?

**Alternatives**: MLP, small GPT, embedding-based

**Chosen**: Pre-trained Sentence-BERT (all-MiniLM-L6-v2)

**Reasons**:
1. Pre-trained on semantic understanding (sentence embeddings)
2. Fast inference (~1-2ms on GPU, equivalent to 1 transformer layer)
3. Good generalization to new queries without fine-tuning base weights
4. Produces contextualized representations via mean pooling
5. Compact size (22M parameters) suitable for resource-constrained deployment

### Why Normal Distribution for Ratio?

**Alternatives**: Fixed ratio, learned discrete distribution

**Chosen**: Normal(μ, σ²) with learned parameters (μ, log σ)

**Reasons**:
1. Exploration during training (via sampling from distribution)
2. Uncertainty quantification (σ represents confidence)
3. Smooth optimization (fully differentiable)
4. Balances determinism and stochasticity
5. Constraining: μ via sigmoid ∈ [0, 1], log σ clipped to [-7, -3] for numerical stability

### Why Differentiable Top-K?

**Problem**: TopK is non-differentiable

**Solution**: Sinkhorn sorting approximation

```python
class DifferentiableTopK(nn.Module):
    def forward(self, scores):
        # Soft permutation matrix via Sinkhorn
        cost = (scores.unsqueeze(2) - scores_sorted.unsqueeze(1)) ** 2
        log_P = -cost / epsilon
        
        # Sinkhorn iterations
        for _ in range(n_iters):
            log_P = log_P - logsumexp(log_P, dim=-2)
            log_P = log_P - logsumexp(log_P, dim=-1)
        
        # Sum probabilities for top-k positions
        return logsumexp(log_P[:, :, :k], dim=-1)
```

**Benefits**:
1. Gradients flow through discrete selection
2. Enables end-to-end training with DPO/ORPO
3. Smooth approximation (controlled by epsilon)

## Performance Characteristics

### Memory Usage

| Component | Parameters | Memory (bfloat16) |
|-----------|-----------|-------------------|
| SLM xLAM-2 1B | 1B | ~2 GB |
| LoRA Adapters (28x) | ~3M | ~6 MB |
| Router (MiniLM) | 22M | ~44 MB |
| **Total** | ~1.025B | ~2.05 GB |

### Latency

- **Router**: ~1-2 ms (equivalent to 1 layer)
- **LoRA Adapter**: ~0.5 ms per layer
- **SLM Layer**: ~5-10 ms per layer
- **Speedup**: ~18% with 5.5 layers pruned on average (ORPO)

### Trade-offs

| Metric | More Pruning | Less Pruning |
|--------|--------------|--------------|
| Speed | ↑ Faster | ↓ Slower |
| Accuracy | ↓ Lower | ↑ Higher |
| Memory | ↓ Less | → Same |
| Quality | ↓ Risk | ↑ Safe |

## Future Improvements

1. **Context-aware adapters**: Train adapters with pruning mask awareness to improve pruned configuration performance
2. **Controllable trade-offs**: Enable users to specify accuracy-efficiency preferences via control parameter
3. **Multi-task generalization**: Extend to multiple tasks by training task-specific adapter sets and task-aware router
4. **Finer granularity**: Explore dynamic pruning at attention head or neuron level for potentially better trade-offs
5. **Router distillation**: Compress BERT router to lightweight model to reduce overhead
6. **Combination with other techniques**: Integrate with quantization or other compression methods for cumulative gains

## References

- Transformers: "Attention Is All You Need" (Vaswani et al., 2017)
- Adapters: "Parameter-Efficient Transfer Learning" (Houlsby et al., 2019)
- DPO: "Direct Preference Optimization" (Rafailov et al., 2023)
- ORPO: "Odds Ratio Preference Optimization" (Hong et al., 2024)
- Differentiable Ranking: "Fast Differentiable Sorting" (Blondel et al., 2020)

