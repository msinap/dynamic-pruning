# Usage Examples

This document provides practical examples for using the dynamic layer pruning system.

## Table of Contents

1. [Basic Inference](#basic-inference)
2. [Training Examples](#training-examples)
3. [Evaluation Examples](#evaluation-examples)
4. [Advanced Usage](#advanced-usage)
5. [Troubleshooting](#troubleshooting)

## Basic Inference

### Example 1: Simple Function Calling

```python
import torch
from src.llm import load_llm, generate_llm_output_with_pruning
from src.adapter import load_adapters
from src.router import get_router_and_tokenizer

# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CONFIG = {
    "llm_model_name": "Salesforce/xLAM-2-1b-fc-r",
    "router_base_model_name": "sentence-transformers/all-MiniLM-L6-v2",
    "adapter_bottleneck_dim": 64,
}

# Load models
print("Loading LLM...")
llm_model, llm_tokenizer = load_llm(CONFIG)

print("Loading router...")
router, router_tokenizer = get_router_and_tokenizer(
    router_base_model_name=CONFIG["router_base_model_name"],
    head_hidden_dim=128,
    num_llm_layers=CONFIG["llm_num_layers"],
    log_std_min=-7,
    log_std_max=-3,
    device=device,
)

print("Loading adapters...")
adapters = load_adapters(
    adapter_path_template="./models/adapter/adapter_{i}.pth",
    adapter_io_dim=CONFIG["llm_hidden_dim"],
    adapter_bottleneck_dim=CONFIG["adapter_bottleneck_dim"],
    num_llm_layers=CONFIG["llm_num_layers"],
    device=device,
)

# Prepare input
sample = {
    "query": "What's the weather like in Tokyo?",
    "tools": '''[
        {
            "name": "get_weather",
            "description": "Get current weather",
            "parameters": {
                "location": {
                    "type": "string",
                    "description": "City name"
                }
            }
        }
    ]'''
}

# Generate with dynamic pruning
output = generate_llm_output_with_pruning(
    sample=sample,
    model_llm=llm_model,
    tokenizer_llm=llm_tokenizer,
    adapters=adapters,
    router=router,
    tokenizer_router=router_tokenizer,
    verbose=True,  # Shows which layers are pruned
)

print(f"Output: {output}")
```

### Example 2: Batch Inference

```python
from datasets import load_dataset

# Load test dataset
dataset = load_dataset("Salesforce/xlam-function-calling-60k")['train']
test_samples = dataset.select(range(100))

# Process batch
results = []
for sample in test_samples:
    output = generate_llm_output_with_pruning(
        sample=sample,
        model_llm=llm_model,
        tokenizer_llm=llm_tokenizer,
        adapters=adapters,
        router=router,
        tokenizer_router=router_tokenizer,
        verbose=False,
    )
    results.append({
        'query': sample['query'],
        'expected': sample['answers'],
        'predicted': output,
    })

# Save results
import json
with open('results.json', 'w') as f:
    json.dump(results, f, indent=2)
```

### Example 3: Compare with Full Model

```python
from src.llm import generate_llm_output
from src.evaluation import json_match_score
import time

sample = {...}  # Your test sample

# Full model
start = time.time()
full_output = generate_llm_output(sample, llm_model, llm_tokenizer)
full_time = time.time() - start

# Pruned model
start = time.time()
pruned_output = generate_llm_output_with_pruning(
    sample, llm_model, llm_tokenizer, adapters, router, router_tokenizer
)
pruned_time = time.time() - start

# Compare
print(f"Full Model:")
print(f"  Time: {full_time:.3f}s")
print(f"  Output: {full_output}")
print(f"  Correct: {json_match_score(full_output, sample['answers'])}")

print(f"\nPruned Model:")
print(f"  Time: {pruned_time:.3f}s")
print(f"  Speedup: {full_time/pruned_time:.2f}x")
print(f"  Output: {pruned_output}")
print(f"  Correct: {json_match_score(pruned_output, sample['answers'])}")
```

## Training Examples

### Example 4: Training Adapters from Scratch

```python
import torch
import wandb
from src.llm import load_llm, load_and_split_dataset
from src.adapter import train_adapters, Adapter, save_adapters

# Configuration
CONFIG = {
    "llm_model_name": "Salesforce/xLAM-2-1b-fc-r",
    "dataset_name": "Salesforce/xlam-function-calling-60k",
    "lr": 1e-4,
    "adapter_bottleneck_dim": 64,
    "num_samples": 5000,
    "eval_every": 500,
}

# Initialize wandb
wandb.init(project="adapter-training", config=CONFIG)

# Load model and data
llm_model, llm_tokenizer = load_llm(CONFIG)
test_data, train_data = load_and_split_dataset(
    CONFIG["dataset_name"], 
    test_size=50
)

# Initialize adapters
adapters = [
    Adapter(
        io_dim=CONFIG["llm_hidden_dim"],
        bottleneck_dim=CONFIG["adapter_bottleneck_dim"]
    ).to("cuda", dtype=torch.bfloat16)
    for _ in range(CONFIG["llm_num_layers"])
]

# Train
train_adapters(
    adapters=adapters,
    train_dataset=train_data.to_list()[:CONFIG["num_samples"]],
    num_layers=CONFIG["llm_num_layers"],
    tokenizer=llm_tokenizer,
    model=llm_model,
    lr=CONFIG["lr"],
    eval_every_n_steps=CONFIG["eval_every"],
    eval_dataset=test_data,
    verbose=True,
)

# Save
save_adapters(
    adapters=adapters,
    run_id=wandb.run.id,
    path_template="./models/adapter/{run_id}",
)

print(f"Adapters saved to ./models/adapter/{wandb.run.id}/")
```

### Example 5: Custom Router Training

```python
from src.router import Router
from src.preference_data_training import train_router_with_preference_optimization, dpo_loss_function
from src.preference_data_generation import PreferenceDataset, get_dataloader

# Initialize router from scratch
router = Router(
    base_bert_model=base_model,
    head_hidden_dim=128,
    num_llm_layers=24,
    log_std_min=-7,
    log_std_max=-3,
).to("cuda", dtype=torch.bfloat16)

# Load preference data
preference_data = PreferenceDataset(
    preference_data_list,
    function_calling_dataset
)
dataloader = get_dataloader(
    preference_data,
    batch_size=128,
    tokenizer=router_tokenizer,
)

# Train with custom hyperparameters
train_router_with_preference_optimization(
    router=router,
    router_tokenizer=router_tokenizer,
    learning_rate=5e-4,  # Custom learning rate
    train_dataloader=dataloader,
    log_every_n_steps=50,
    loss_fn=dpo_loss_function,
    loss_fn_kwargs={},
    eval_dataset=eval_data,
    eval_every_n_steps=200,
    llm_model=llm_model,
    llm_tokenizer=llm_tokenizer,
    adapters=adapters,
    score_funcs=[json_match_score, ratio_function_calls_score],
)

# Save router
torch.save(router.state_dict(), "./models/router/custom_router.pth")
```

## Evaluation Examples

### Example 6: Comprehensive Evaluation

```python
from src.evaluation import (
    evaluate_model_on_dataset,
    exact_match_score,
    json_match_score,
    ratio_function_calls_score,
    partial_match_score
)
from src.prune import LLMPruner

# Define pruning configurations to test
configs = [
    [],  # No pruning
    [5, 10, 15],  # Prune 3 specific layers
    [2, 4, 6, 8, 10, 12],  # Prune 6 layers
    list(range(0, 28, 2)),  # Prune every other layer (14 layers total)
]

# Evaluate each configuration
results = []
for pruned_layers in configs:
    print(f"\nEvaluating with {len(pruned_layers)} pruned layers: {pruned_layers}")
    
    with LLMPruner(llm_model, adapters) as pruner:
        pruner.prune_model(pruned_layers)
        
        scores = evaluate_model_on_dataset(
            model=pruner.llm_model_full,
            tokenizer=llm_tokenizer,
            dataset=test_dataset,
            score_funcs=[
                exact_match_score,
                json_match_score,
                ratio_function_calls_score,
                partial_match_score,
            ],
            verbose=False,
        )
        
        results.append({
            'num_pruned': len(pruned_layers),
            'pruned_layers': pruned_layers,
            'scores': scores,
        })

# Print summary
import pandas as pd
df = pd.DataFrame([
    {
        'Pruned': r['num_pruned'],
        'Exact': r['scores']['exact_match_score'],
        'JSON': r['scores']['json_match_score'],
        'Ratio': r['scores']['ratio_function_calls_score'],
        'Partial': r['scores']['partial_match_score'],
    }
    for r in results
])
print("\n" + df.to_string(index=False))
```

### Example 7: Layer Importance Analysis

```python
from src.adapter import evaluate_adapters_separately
import matplotlib.pyplot as plt

# Evaluate each adapter individually
layer_scores = evaluate_adapters_separately(
    adapters=adapters,
    dataset=test_dataset,
    num_layers=CONFIG["llm_num_layers"],
    tokenizer=llm_tokenizer,
    model=llm_model,
    verbose=True,
)

# Visualize
plt.figure(figsize=(14, 6))
plt.bar(range(len(layer_scores)), layer_scores)
plt.xlabel('Layer Index')
plt.ylabel('Score (when only this layer is pruned)')
plt.title('Individual Layer Pruning Impact - xLAM-2 1B (28 layers)')
plt.axhline(y=0.7, color='r', linestyle='--', label='Acceptable threshold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('layer_importance.png', dpi=300, bbox_inches='tight')
print("Saved visualization to layer_importance.png")

# Identify best/worst layers for pruning
sorted_layers = sorted(enumerate(layer_scores), key=lambda x: x[1], reverse=True)
print("\nEasiest to prune (highest score):")
for idx, score in sorted_layers[:5]:
    print(f"  Layer {idx}: {score:.3f}")

print("\nHardest to prune (lowest score):")
for idx, score in sorted_layers[-5:]:
    print(f"  Layer {idx}: {score:.3f}")
```

## Advanced Usage

### Example 8: Custom Pruning Strategy

```python
class FixedPatternRouter:
    """Always prune the same layers (for testing)."""
    def __init__(self, pruned_layers, num_layers=28):
        self.pruned_layers = pruned_layers
        self.num_layers = num_layers
    
    def __call__(self, *args, **kwargs):
        # Return dummy scores
        scores = torch.zeros(1, self.num_layers)
        scores[0, self.pruned_layers] = 1.0
        mu = torch.tensor([[len(self.pruned_layers) / self.num_layers]])
        log_std = torch.tensor([[-10.0]])  # Very low std = deterministic
        return scores, mu, log_std

# Use fixed pattern (prune layers 5, 10, 15, 20)
fixed_router = FixedPatternRouter([5, 10, 15, 20], num_layers=28)

output = generate_llm_output_with_pruning(
    sample=sample,
    model_llm=llm_model,
    tokenizer_llm=llm_tokenizer,
    adapters=adapters,
    router=fixed_router,
    tokenizer_router=None,  # Not used
    verbose=True,
)
```

### Example 9: Temperature-Controlled Sampling

```python
def generate_with_temperature(sample, temperature=1.0):
    """Control exploration via temperature."""
    inputs = router_tokenizer(
        sample['query'],
        return_tensors="pt",
        padding=True,
        truncation=True,
    ).to(router.bert.device)
    
    with torch.no_grad():
        scores, mu_ratio, log_std_ratio = router(**inputs)
    
    # Adjust std by temperature
    std_ratio = torch.exp(log_std_ratio) * temperature
    ratio_dist = torch.distributions.Normal(mu_ratio, std_ratio)
    ratio = torch.clamp(ratio_dist.sample(), 0, 1).item()
    
    num_pruned = int(ratio * CONFIG["llm_num_layers"])
    pruned_layers = torch.topk(scores, num_pruned).indices.tolist()[0]
    
    with LLMPruner(llm_model, adapters) as pruner:
        pruner.prune_model(pruned_layers)
        return generate_llm_output(sample, llm_model, llm_tokenizer)

# Test different temperatures
for temp in [0.5, 1.0, 2.0]:
    print(f"\nTemperature: {temp}")
    output = generate_with_temperature(sample, temperature=temp)
    print(f"Output: {output}")
```

### Example 10: Ensemble Router

```python
class EnsembleRouter:
    """Combine multiple routers."""
    def __init__(self, routers):
        self.routers = routers
    
    def __call__(self, input_ids, attention_mask):
        all_scores = []
        all_mus = []
        all_log_stds = []
        
        for router in self.routers:
            scores, mu, log_std = router(input_ids, attention_mask)
            all_scores.append(scores)
            all_mus.append(mu)
            all_log_stds.append(log_std)
        
        # Average predictions
        avg_scores = torch.stack(all_scores).mean(dim=0)
        avg_mu = torch.stack(all_mus).mean(dim=0)
        avg_log_std = torch.stack(all_log_stds).mean(dim=0)
        
        return avg_scores, avg_mu, avg_log_std

# Load multiple trained routers
router1 = ...  # Load first router
router2 = ...  # Load second router
ensemble = EnsembleRouter([router1, router2])

# Use ensemble for inference
output = generate_llm_output_with_pruning(
    sample=sample,
    model_llm=llm_model,
    tokenizer_llm=llm_tokenizer,
    adapters=adapters,
    router=ensemble,
    tokenizer_router=router_tokenizer,
)
```

## Troubleshooting

### Issue 1: CUDA Out of Memory

```python
# Solution 1: Reduce batch size
CONFIG["batch_size"] = 32  # Instead of 128

# Solution 2: Use gradient accumulation
optimizer = torch.optim.Adam(router.parameters(), lr=1e-4)
accumulation_steps = 4

for step, batch in enumerate(dataloader):
    loss = compute_loss(batch) / accumulation_steps
    loss.backward()
    
    if (step + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()

# Solution 3: Clear cache periodically
if step % 100 == 0:
    torch.cuda.empty_cache()
```

### Issue 2: Router Predicts Too Much/Little Pruning

```python
# Adjust log_std range
router = get_router_and_tokenizer(
    log_std_min=-5,  # Less conservative (was -7)
    log_std_max=-2,  # More exploration (was -3)
    ...
)

# Or add temperature scaling at inference
ratio = ratio_dist.sample().item()
ratio = 0.5 * ratio + 0.25  # Restrict to [0.25, 0.75] range
```

### Issue 3: Poor Performance on New Domain

```python
# Fine-tune router on new domain data
# 1. Collect small dataset from new domain
new_domain_data = [...]

# 2. Generate offline data
from scripts.generate_offline_rl_data_dfs import generate_offline_data
offline_data = generate_offline_data(
    llm_model, llm_tokenizer, adapters, new_domain_data
)

# 3. Fine-tune router
train_fbc(
    router=router,
    # ... (use smaller lr for fine-tuning)
    learning_rate=1e-5,
)
```

### Issue 4: Evaluation Takes Too Long

```python
# Use smaller test set
test_dataset = test_dataset.select(range(20))  # Instead of 100

# Disable verbose output
verbose = False

# Profile to find bottleneck
import cProfile
import pstats

profiler = cProfile.Profile()
profiler.enable()

# Your evaluation code here
evaluate_model_on_dataset(...)

profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(10)  # Top 10 slowest functions
```

## More Examples

For more examples, check:
- `notebooks/` - Jupyter notebooks with interactive examples
- `tests/` - Unit tests showing component usage
- `scripts/` - Complete training scripts

## Getting Help

If you run into issues not covered here:
1. Check [ARCHITECTURE.md](ARCHITECTURE.md) for technical details
2. Open an issue on GitHub with:
   - Your code
   - Error message
   - Environment info (`python --version`, `torch.__version__`)
3. Ask in discussions for usage questions

