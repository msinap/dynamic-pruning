# Dynamic Layer Pruning for Small Language Models

A novel approach to efficient SLM inference through **dynamic, query-specific layer pruning** using lightweight LoRA adapters and a learned router model. This repository contains the implementation of techniques from a master's thesis exploring preference-based reinforcement learning for adaptive model compression, specifically optimized for tool calling tasks.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 📋 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Training Pipeline](#training-pipeline)
- [Training Methods](#training-methods)
- [Datasets](#datasets)
- [Project Structure](#project-structure)
- [Citation](#citation)

## 🎯 Overview

This project addresses the computational cost of Small Language Model (SLM) inference by dynamically pruning transformer layers based on input complexity. Unlike static pruning approaches, our method:

- **Adapts per query**: Different inputs may skip different layers
- **Maintains accuracy**: Lightweight LoRA adapters replace pruned layers
- **Learns pruning strategy**: Router model trained with offline RL/preference optimization
- **Reduces latency**: Fewer layers = faster inference

### Problem Statement

Running large transformer models is expensive. Not all queries require the full model capacity. Can we learn which layers to skip for each specific input while maintaining output quality?

### Solution

1. **LoRA Adapters**: Train lightweight bottleneck networks via white-box knowledge distillation to approximate each transformer layer
2. **Router**: Train a BERT-based model to predict which layers to prune per query
3. **Offline RL**: Use preference optimization (DPO/ORPO) to learn optimal accuracy-efficiency trade-offs from static datasets

## ✨ Key Features

- **Query-Specific Pruning**: Dynamic layer selection based on input characteristics
- **Lightweight LoRA Adapters**: ~1-5% of original layer parameters (bottleneck dimension: 64)
- **Multiple Training Methods**:
  - Filtered Behavioral Cloning (FBC)
  - Direct Preference Optimization (DPO)
  - Odds Ratio Preference Optimization (ORPO)
- **Differentiable Top-K**: Sinkhorn sorting and Gumbel-Softmax for gradient-based layer selection
- **Function Calling Task**: Evaluated on xLAM-2 1B (28 layers) for tool calling
- **Comprehensive Evaluation**: JSON Match and multiple metrics for function call quality

## 🏗️ Architecture

### System Components

```
┌─────────────────────────────────────────────────────────┐
│                    Input Query                          │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
         ┌───────────────────────┐
         │  Router Model (BERT)  │
         │  - Layer Scores       │
         │  - Pruning Ratio      │
         └───────────┬───────────┘
                     │
                     ▼
         ┌───────────────────────┐
         │  Select Layers to     │
         │  Prune (Top-K)        │
         └───────────┬───────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│               LLM with Dynamic Pruning                  │
│  Layer 0:  [■■■■■] Transformer Layer                   │
│  Layer 1:  [▒▒▒▒▒] Adapter (Pruned)                    │
│  Layer 2:  [■■■■■] Transformer Layer                   │
│  Layer 3:  [▒▒▒▒▒] Adapter (Pruned)                    │
│  ...                                                    │
│  Layer N:  [■■■■■] Transformer Layer                   │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
              Generated Output
```

### Adapter Architecture

```
Input (hidden_dim)
    │
    ├──────────────────┐
    │                  │
    ▼                  │
Linear(down)           │ (Residual)
bottleneck_dim         │
    │                  │
    ▼                  │
  ReLU                 │
    │                  │
    ▼                  │
Linear(up)             │
hidden_dim             │
    │                  │
    ▼                  │
   Add ◄───────────────┘
    │
    ▼
Output (hidden_dim)
```

### Router Model

- **Base**: Pre-trained BERT encoder (sentence-transformers/all-MiniLM-L6-v2)
- **Layer Scoring Head**: Predicts prunability score for each layer (sigmoid output, higher = safer to prune)
- **Pruning Ratio Head**: Predicts parameters (μ, log σ) for Normal distribution over pruning ratio
- **Output**: 
  - Layer scores: Per-layer prunability scores
  - Ratio distribution: N(μ, σ²) for sampling proportion of layers to prune
- **Inference**: Sample ratio r ~ N(μ, σ²), compute k = ⌊r × L⌋, select top-k scored layers

## 📦 Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended)
- 16GB+ GPU memory for 1B parameter models

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/dynamic-pruning.git
cd dynamic-pruning

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install the package in development mode
pip install -e .
```

### Requirements

```
torch>=2.0.0
transformers>=4.30.0
datasets>=2.14.0
accelerate>=0.20.0
wandb>=0.15.0
outlines>=0.0.30
seaborn>=0.12.0
ipykernel>=6.25.0
jupyter>=1.0.0
```

## 🚀 Quick Start

### Download Pre-trained Models & Data

```bash
# Download trained adapters and router models
huggingface-cli download sinap/dynamic-pruning --local-dir ./models/ --repo-type model

# Download offline RL datasets for training
huggingface-cli download sinap/dynamic-pruning-offline-rl-data --local-dir ./datasets/ --repo-type dataset
```

### Basic Inference

```python
import torch
from src.llm import load_llm, generate_llm_output_with_pruning
from src.adapter import load_adapters
from src.router import get_router_and_tokenizer

# Configuration
CONFIG = {
    "llm_model_name": "Salesforce/xLAM-2-1b-fc-r",
    "router_base_model_name": "sentence-transformers/all-MiniLM-L6-v2",
    "adapter_bottleneck_dim": 64,
}

# Load models
device = torch.device("cuda")
llm_model, llm_tokenizer = load_llm(CONFIG)
router, router_tokenizer = get_router_and_tokenizer(
    router_base_model_name=CONFIG["router_base_model_name"],
    head_hidden_dim=128,
    num_llm_layers=CONFIG["llm_num_layers"],
    log_std_min=-7,
    log_std_max=-3,
    device=device,
)
adapters = load_adapters(
    adapter_path_template="./models/adapter/adapter_{i}.pth",
    adapter_io_dim=CONFIG["llm_hidden_dim"],
    adapter_bottleneck_dim=CONFIG["adapter_bottleneck_dim"],
    num_llm_layers=CONFIG["llm_num_layers"],
    device=device,
)

# Generate with dynamic pruning
sample = {
    "query": "Get the current weather in San Francisco",
    "tools": '[{"name": "get_weather", "parameters": {"location": "string"}}]'
}

output = generate_llm_output_with_pruning(
    sample=sample,
    model_llm=llm_model,
    tokenizer_llm=llm_tokenizer,
    adapters=adapters,
    router=router,
    tokenizer_router=router_tokenizer,
    verbose=True,  # Print which layers are pruned
)
print(output)
```

## 🔧 Training Pipeline

### Step 1: Train LoRA Adapters

Train lightweight LoRA adapters to replace each transformer layer via white-box knowledge distillation:

```bash
python scripts/train_adapter.py
```

**What it does:**
- Generates outputs with hidden states from the full SLM
- Trains each LoRA adapter independently to minimize MSE between adapter output and original layer output
- Only uses samples where the SLM produces correct function calls (white-box distillation)
- Enables parallel training across all 28 adapters
- Saves trained adapters to `./models/adapter/`

**Key hyperparameters:**
- `lr`: 1e-4
- `adapter_bottleneck_dim`: 64
- `num_samples_to_train`: 5000

### Step 2: Generate Offline Data

Explore the pruning space to collect training data:

```bash
# Depth-First Search exploration (recommended)
python scripts/generate_offline_rl_data_dfs.py

# Or random exploration
python scripts/generate_offline_rl_data_random.py
```

**What it does:**
- Systematically tries different layer pruning combinations
- Evaluates accuracy for each configuration
- Saves successful configurations to `./datasets/`

**DFS Strategy (Recommended):**
- Depth-first search exploration starting with 0 layers pruned
- Iteratively adds layers that maintain accuracy above threshold
- Backtracks when performance drops
- Produces higher variance data with more aggressive pruning configurations
- Creates clear contrasts between good and bad pruning decisions

### Step 3: Train Router

Choose a training method:

#### Option A: Filtered Behavioral Cloning (Baseline)

```bash
python scripts/train_fbc.py
```

Simple supervised learning from successful configurations.

#### Option B: Direct Preference Optimization (Recommended)

```bash
# First generate preference pairs
python scripts/generate_preference_dataset.py

# Then train with DPO
python scripts/train_dpo.py
```

Learns from pairwise preferences between pruning configurations.

#### Option C: Odds Ratio Preference Optimization

```bash
python scripts/train_orpo.py
```

Combines preference learning with supervised fine-tuning.

## 🎓 Training Methods

### Filtered Behavioral Cloning (FBC)

**Approach**: Supervised learning on successful pruning configurations

**Loss Function**:
```
L_FBC = (μ_pred - μ_target)² + BCE(scores_pred, scores_target)
```
Where BCE is binary cross-entropy loss for layer selection.

**Pros**:
- Simple and stable
- Fast convergence
- Good baseline

**Cons**:
- Doesn't learn from comparisons
- May not generalize to unseen configurations

### Direct Preference Optimization (DPO)

**Approach**: Learn from pairwise preferences between configurations

**Loss Function**:
```
L_DPO = -E[(x,y_w,y_l)][log σ(β log(π_θ(y_w|x)/π_θ(y_l|x)))]
```
where y_w is the preferred (winner) configuration and y_l is the loser, β is a scaling parameter.

**Preference Criterion**:
1. Higher accuracy score, OR
2. Same accuracy but more layers pruned

**Pros**:
- Learns accuracy-efficiency trade-offs
- Doesn't need explicit reward function
- More robust than standard RL

**Cons**:
- Requires preference dataset
- More complex than FBC

### Odds Ratio Preference Optimization (ORPO)

**Approach**: DPO + supervised learning on winning examples

**Loss Function**:
```
L_ORPO = -E[log σ(β log(π_θ(y_w|x)/π_θ(y_l|x))) + log π_θ(y_w|x)]
```
Combines odds-ratio based preference learning with supervised fine-tuning on winning configurations.

**Pros**:
- Combines benefits of DPO and FBC
- Faster convergence than pure DPO
- Better sample efficiency

**Cons**:
- More hyperparameters to tune
- Slightly more complex training loop

## 📊 Datasets

### Function Calling Dataset

- **Source**: [Salesforce/xlam-function-calling-60k](https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k)
- **Task**: Given tools and a query, generate appropriate function calls
- **Format**: JSON function call sequences
- **Evaluation Metrics**:
  - `exact_match_score`: Binary exact match
  - `json_match_score`: Parsed JSON equality
  - `ratio_function_calls_score`: Fraction of correct function calls
  - `partial_match_score`: Partial credit for function names and arguments

### Offline RL Datasets

Generated by systematic exploration (available on HuggingFace):

- **Random Exploration**: 5 random pruning configurations per input
- **DFS Exploration (Recommended)**: Depth-first search with up to 20 scenarios per input
  - Systematically explores high-pruning configurations
  - Backtracks when performance drops below threshold
  - Produces richer training signal for preference learning
- **Format**: Each entry contains:
  - `id`: Sample ID from xlam-function-calling-60k
  - `pruned_layers`: List of layer indices that were replaced with adapters
  - `score`: JSON Match accuracy for this configuration

### Preference Datasets

Generated from offline data (available on HuggingFace):

- **All Pairs**: All pairwise comparisons from DFS data (recommended for training)
- **Random Pairs**: Randomly sampled pairwise comparisons
- **Preference Criterion**:
  - Configuration A preferred over B if: (1) higher accuracy score, OR (2) same accuracy but more layers pruned (more efficient)
- **Format**: Each entry contains:
  - `id`: Sample ID
  - `winner_pruned_layers`: Preferred configuration (better accuracy or more efficient)
  - `loser_pruned_layers`: Less preferred configuration
  - `winner_score`: JSON Match score of winner
  - `loser_score`: JSON Match score of loser

## 📁 Project Structure

```
dynamic-pruning/
├── src/                          # Core library
│   ├── llm.py                   # LLM loading and generation
│   ├── adapter.py               # Adapter architecture and training
│   ├── router.py                # Router model architecture
│   ├── prune.py                 # Layer pruning utilities
│   ├── evaluation.py            # Evaluation metrics
│   ├── fbc.py                   # Filtered Behavioral Cloning
│   ├── preference_data_generation.py    # Preference dataset creation
│   └── preference_data_training.py      # DPO/ORPO training
├── scripts/                      # Training scripts
│   ├── train_adapter.py         # Train adapters
│   ├── train_fbc.py            # Train router with FBC
│   ├── train_dpo.py            # Train router with DPO
│   ├── train_orpo.py           # Train router with ORPO
│   ├── generate_offline_rl_data_dfs.py    # DFS exploration
│   ├── generate_offline_rl_data_random.py # Random exploration
│   └── generate_preference_dataset.py     # Create preference pairs
├── requirements.txt             # Python dependencies
├── LICENSE                      # MIT License
└── README.md                    # This file
```

## 📈 Results

### Performance Metrics

| Method | Accuracy (JSON Match) | Speedup | Avg Layers Pruned |
|--------|----------------------|---------|-------------------|
| Original (No Pruning) | 71% | 0% | 0 |
| Static 2-Layer | 63% | 7.1% | 2 |
| Static 3-Layer | 51% | 10.7% | 3 |
| Static 5-Layer | 38% | 17.6% | 5 |
| Router - FBC | 52% | 16.1% | ~5.5 |
| Router - DPO | 59% | 17.1% | ~5.5 |
| Router - ORPO | **61%** | **17.9%** | ~5.5 |

*Results on xLAM-2 1B (28 layers) evaluated on xlam-function-calling-60k dataset. Speedup includes router overhead (equivalent to 1 layer).*

### Key Findings

1. **Dynamic > Static**: ORPO achieves 61% accuracy with ~18% speedup (equivalent to 5-6 layer pruning), while static 3-layer pruning only achieves 51% accuracy with 10.7% speedup
2. **Preference > Behavioral**: DPO (59%) and ORPO (61%) significantly outperform FBC (52%) by learning from preference comparisons rather than imitation
3. **LoRA Adapters are Effective**: Low MSE reconstruction error for middle layers (2-20), indicating functional redundancy
4. **Layer Importance Varies**: Layer 1 (MSE: 46.75) and final layers (26-27) show high reconstruction error, confirming critical roles in embedding transformation and task-specific output generation

## 🔬 Advanced Usage

### Custom Router Training

```python
from src.router import get_router_and_tokenizer
from src.preference_data_training import train_router_with_preference_optimization, dpo_loss_function

# Initialize router
router, tokenizer = get_router_and_tokenizer(
    router_base_model_name="sentence-transformers/all-MiniLM-L6-v2",
    head_hidden_dim=128,
    num_llm_layers=28,  # xLAM-2 1B has 28 layers
    log_std_min=-7,
    log_std_max=-3,
    device=device,
)

# Train with custom settings
train_router_with_preference_optimization(
    router=router,
    router_tokenizer=tokenizer,
    learning_rate=1e-3,
    train_dataloader=dataloader,
    log_every_n_steps=100,
    loss_fn=dpo_loss_function,
    loss_fn_kwargs={},
    eval_dataset=eval_data,
    eval_every_n_steps=500,
    llm_model=llm,
    llm_tokenizer=llm_tokenizer,
    adapters=adapters,
    score_funcs=[json_match_score],
)
```

### Evaluation Only

```python
from src.evaluation import evaluate_model_on_dataset, ratio_function_calls_score
from src.prune import LLMPruner

# Evaluate with specific layers pruned
with LLMPruner(model, adapters) as pruner:
    pruner.prune_model([2, 5, 8, 11, 14, 17])  # Prune 6 layers
    scores = evaluate_model_on_dataset(
        model=pruner.llm_model_full,
        tokenizer=tokenizer,
        dataset=test_data,
        score_funcs=[ratio_function_calls_score],
        verbose=True,
    )
print(f"Score with pruning: {scores}")
```

## 🤝 Contributing

This is a research project from a master's thesis. Contributions, issues, and feature requests are welcome!

## 📝 Citation

If you use this code in your research, please cite:

```bibtex
@mastersthesis{dynamic-pruning-2024,
  author = {[Your Name]},
  title = {Dynamic Layer Pruning for Large Language Models using Preference-Based Reinforcement Learning},
  school = {[Your University]},
  year = {2024},
  type = {Master's Thesis}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **xLAM-2 1B**: Salesforce for the tool calling SLM and xlam-function-calling-60k dataset
- **LoRA**: Hu et al. for the low-rank adaptation technique
- **Transformers**: HuggingFace for the transformers library
- **DPO**: Rafailov et al. for Direct Preference Optimization
- **ORPO**: Hong et al. for Odds Ratio Preference Optimization
- **Sentence-BERT**: all-MiniLM-L6-v2 model for router base

## 📧 Contact

For questions or feedback:
- Open an issue on GitHub
- Email: [your-email@example.com]

---

**Note**: This project is part of ongoing research. Models and datasets will be made available on HuggingFace after thesis publication.
