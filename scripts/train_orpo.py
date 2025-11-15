"""
Router Training with Odds Ratio Preference Optimization (ORPO)

This script trains the router model using ORPO, which combines preference learning
with supervised fine-tuning on winning examples. ORPO uses odds ratios instead of
likelihood ratios for more stable training.

Usage:
    python scripts/train_orpo.py

The script will:
    1. Load pre-trained LLM and adapters
    2. Initialize router model
    3. Load preference dataset (pairs of pruning configurations)
    4. Train router using ORPO loss (preference + SFT)
    5. Periodically evaluate on test set

ORPO Training:
    - Combines preference optimization with supervised learning
    - Uses odds ratios for more stable gradients
    - Includes SFT loss on winning configurations
    - Often converges faster than pure DPO

Configuration:
    Edit the CONFIG dictionary to customize:
    - orpo_alpha: Weight for SFT loss component
    - fbc_alpha: Weight for layer scores in SFT loss
"""

import random
import numpy as np
import torch
from datasets import Dataset

from src.llm import load_llm, load_dataset_list, generate_llm_output_with_pruning
from src.evaluation import partial_match_score, ratio_function_calls_score, json_match_score
from src.adapter import load_adapters
from src.preference_data_generation import split_train_test, get_dataloader, PreferenceDataset
from src.preference_data_training import train_router_with_preference_optimization, orpo_loss_function
from src.router import get_router_and_tokenizer

CONFIG = {
    "llm_model_name": "Salesforce/xLAM-2-1b-fc-r",
    "dataset_name": "Salesforce/xlam-function-calling-60k",
    "dataset_eval_size": 20,
    "preference_dataset_path": "/workspace/datasets/preference_dataset_dfs_5000_samples_20_scenarios_all_pairs",

    "router_base_model_name": "sentence-transformers/all-MiniLM-L6-v2",
    "router_head_hidden_dim": 128,
    "router_log_std_min": -7,
    "router_log_std_max": -3,
    
    "adapter_bottleneck_dim": 64,

    "num_samples": 5000,
    "scenarios_per_num_pruned_layers": 5,
    "stop_adding_layers_threshold": 0.1,
    "batch_size": 256,
    "orpo_alpha": 0.1,
    "fbc_alpha": 0.01,
}

device = torch.device("cuda")
seed = 23
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    print("Loading LLM, dataset")
    llm_model, llm_tokenizer = load_llm(CONFIG)
    dataset = load_dataset_list(CONFIG["dataset_name"])
    eval_dataset = dataset[-CONFIG["dataset_eval_size"]:]
    train_dataset = dataset[:-CONFIG["dataset_eval_size"]]

    print("Loading adapters")
    llm_adapters = load_adapters(
        adapter_path_template=f'/workspace/models/adapter/adapter_{{i}}.pth',
        adapter_io_dim=CONFIG["llm_hidden_dim"],
        adapter_bottleneck_dim=CONFIG["adapter_bottleneck_dim"],
        num_llm_layers=CONFIG["llm_num_layers"],
        device=device,
    )

    print("Loading router")
    router, router_tokenizer = get_router_and_tokenizer(
        router_base_model_name=CONFIG["router_base_model_name"],
        head_hidden_dim=CONFIG["router_head_hidden_dim"],
        num_llm_layers=CONFIG["llm_num_layers"],
        log_std_min=CONFIG["router_log_std_min"],
        log_std_max=CONFIG["router_log_std_max"],
        device=device,
    )

    print("Loading preference dataset")
    preference_dataset_list = Dataset.load_from_disk(CONFIG["preference_dataset_path"]).to_list()
    random.shuffle(preference_dataset_list)
    preference_dataset = PreferenceDataset(preference_dataset_list, train_dataset)
    preference_dataloader = get_dataloader(preference_dataset, CONFIG["batch_size"], router_tokenizer)

    print("Training router with ORPO")
    train_router_with_preference_optimization(
        router=router,
        router_tokenizer=router_tokenizer,
        learning_rate=1e-3,
        train_dataloader=preference_dataloader,
        log_every_n_steps=10000,
        loss_fn=orpo_loss_function,
        loss_fn_kwargs={
            "orpo_alpha": CONFIG["orpo_alpha"],
            "fbc_alpha": CONFIG["fbc_alpha"],
        },
        eval_dataset=eval_dataset,
        eval_every_n_steps=100,
        llm_model=llm_model,
        llm_tokenizer=llm_tokenizer,
        adapters=llm_adapters,
        score_funcs=[partial_match_score, ratio_function_calls_score, json_match_score],
    ) 
