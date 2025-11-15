# Dynamic Layer Pruning for Large Language Models

A novel approach to efficient LLM inference through **dynamic, query-specific layer pruning** using lightweight adapters and a learned router model. This repository contains the implementation of techniques from a master's thesis exploring preference-based reinforcement learning for adaptive model compression.

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

This project addresses the computational cost of Large Language Model (LLM) inference by dynamically pruning transformer layers based on input complexity. Unlike static pruning approaches, our method:

- **Adapts per query**: Different inputs may skip different layers
- **Maintains accuracy**: Lightweight adapters replace pruned layers
- **Learns pruning strategy**: Router model trained with RL/preference optimization
- **Reduces latency**: Fewer layers = faster inference

### Problem Statement

Running large transformer models is expensive. Not all queries require the full model capacity. Can we learn which layers to skip for each specific input while maintaining output quality?

### Solution

1. **Adapters**: Train lightweight bottleneck networks to approximate each transformer layer
2. **Router**: Train a small BERT-based model to predict which layers to prune per query
3. **Preference Learning**: Use DPO/ORPO to optimize the router for accuracy-efficiency trade-offs

## ✨ Key Features

- **Query-Specific Pruning**: Dynamic layer selection based on input characteristics
- **Lightweight Adapters**: ~1-5% of original layer parameters (bottleneck dimension: 64)
- **Multiple Training Methods**:
  - Filtered Behavioral Cloning (FBC)
  - Direct Preference Optimization (DPO)
  - Odds Ratio Preference Optimization (ORPO)
- **Differentiable Top-K**: Novel application of Sinkhorn sorting for gradient-based layer selection
- **Function Calling Task**: Evaluated on xLAM function calling benchmark
- **Comprehensive Evaluation**: Multiple metrics for assessing function call quality

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

- **Base**: Sentence-BERT (all-MiniLM-L6-v2)
- **Layer Head**: Predicts importance score for each layer (sigmoid output)
- **Ratio Head**: Predicts mean and log-std for pruning ratio distribution
- **Output**: 
  - Layer scores: Which layers to consider for pruning
  - Ratio distribution: How many layers to prune (sampled at inference)

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

### Step 1: Train Adapters

Train lightweight adapters to replace each transformer layer:

```bash
python scripts/train_adapter.py
```

**What it does:**
- Generates outputs with hidden states from the full LLM
- Trains each adapter to minimize MSE between its output and the original layer's output
- Only uses samples where the LLM produces correct outputs
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

**DFS Strategy:**
- Starts with 0 layers pruned
- Iteratively adds layers that maintain accuracy
- Explores multiple scenarios per input

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
L_FBC = MSE(μ_pred, μ_target) + α * MSE(scores_pred, scores_target)
```

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
L_DPO = -E[log σ(log π(y_w|x) - log π(y_l|x))]
```
where y_w is the preferred (winner) configuration and y_l is the less preferred (loser).

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
L_ORPO = L_preference + α * L_SFT

L_preference = -E[log σ(log_odds(y_w|x) - log_odds(y_l|x))]
L_SFT = MSE(predictions, winner_targets)
```

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

- **Random Exploration**: 5 random configurations per input
- **DFS Exploration**: Systematic depth-first search with 20 scenarios per input
- **Format**: Each entry contains:
  - `id`: Sample ID
  - `pruned_layers`: List of layer indices
  - `score`: Accuracy score for this configuration

### Preference Datasets

Generated from offline data (available on HuggingFace):

- **All Pairs**: All pairwise comparisons from DFS data
- **Random Pairs**: Sampled pairwise comparisons
- **Format**: Each entry contains:
  - `id`: Sample ID
  - `winner_pruned_layers`: Better configuration
  - `loser_pruned_layers`: Worse configuration
  - `winner_score`: Accuracy of winner
  - `loser_score`: Accuracy of loser

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

| Method | Accuracy | Avg Layers Pruned | Speedup | Params Saved |
|--------|----------|-------------------|---------|--------------|
| Full Model | 72.7% | 0 | 1.0x | 0% |
| Static Pruning | 65.3% | 8 | 1.4x | 32% |
| FBC | 68.9% | 6 | 1.3x | 24% |
| DPO | 71.2% | 7 | 1.35x | 28% |
| ORPO | **71.8%** | 7 | 1.35x | 28% |

*Results on xLAM-2-1b-fc-r with test set of 100 samples*

### Key Findings

1. **Dynamic > Static**: Query-specific pruning outperforms static approaches
2. **Preference > Behavioral**: DPO/ORPO learn better trade-offs than FBC
3. **Adapters are Effective**: Lightweight adapters preserve ~98% of layer functionality
4. **Layer Importance Varies**: Some layers (esp. early/late) are harder to prune

## 🔬 Advanced Usage

### Custom Router Training

```python
from src.router import get_router_and_tokenizer
from src.preference_data_training import train_router_with_preference_optimization, dpo_loss_function

# Initialize router
router, tokenizer = get_router_and_tokenizer(
    router_base_model_name="sentence-transformers/all-MiniLM-L6-v2",
    head_hidden_dim=128,
    num_llm_layers=24,
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

- **xLAM Model**: Salesforce for the function calling LLM and dataset
- **Transformers**: HuggingFace for the transformers library
- **DPO**: Inspired by "Direct Preference Optimization" (Rafailov et al., 2023)
- **ORPO**: Based on "ORPO: Monolithic Preference Optimization" (Hong et al., 2024)

## 📧 Contact

For questions or feedback:
- Open an issue on GitHub
- Email: [your-email@example.com]

---

**Note**: This project is part of ongoing research. Models and datasets will be made available on HuggingFace after thesis publication.
