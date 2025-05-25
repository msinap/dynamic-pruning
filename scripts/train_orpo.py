from transformers import AutoTokenizer, AutoModelForMaskedLM
import random
import torch
import wandb
import numpy as np
import datasets

from src.llm import *
from src.adapter import *
from src.offline_rl_data import *
from src.preference_data import *
from src.actor import *
from src.orpo import *

CONFIG = {
    "llm_model_name": "Salesforce/xLAM-2-1b-fc-r",
    "actor_base_model_name": "answerdotai/ModernBERT-base",
    "adapter_bottleneck_dim": 512,
    "adapters_path": '/workspace/1_4vw4k59d', # For loading pre-trained adapters
    "offline_rl_dataset_path": "/workspace/offline_20_5000",
    "train_dataset_ratio": 1.0,
    "actor_head_hidden_dim": 256,
    "actor_max_seq_length": 128,
    "orpo_epochs": 5,
    "orpo_batch_size": 64,
    "orpo_lr": 3e-5,
    "orpo_beta": 0.1,
    "num_llm_eval_samples": 50,
    "eval_every_n_batches": 0, # 0 means eval at end of epoch. >0 means eval every N batches.
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "wandb_enabled": True,
    "wandb_project": "orpo_pruning_with_eval",
    "wandb_run_id": f"orpo_eval_run_{random.randint(1000,9999)}",
    "log_std_max": -3,
    "log_std_min": -7,
    "gradient_clip_norm": 1.0,
    "seed": 23,
    "verbose": False,
}

# --- Reproducibility ---
random.seed(CONFIG["seed"])
np.random.seed(CONFIG["seed"])
torch.manual_seed(CONFIG["seed"])
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(CONFIG["seed"])


if __name__ == "__main__":
    llm_model, llm_tokenizer = load_llm(CONFIG)

    actor_tokenizer = AutoTokenizer.from_pretrained(CONFIG["actor_base_model_name"])
    actor_base_model = AutoModelForMaskedLM.from_pretrained(CONFIG["actor_base_model_name"])
    # No need to move actor_base_model to device here, Actor class will handle its copy

    ds = datasets.load_dataset("Salesforce/xlam-function-calling-60k")

    adapter_path_template = f'{CONFIG["adapters_path"]}/adapter_{{i}}.pth'
    llm_adapters = load_adapters(adapter_path_template, CONFIG["adapter_io_dim"], CONFIG["adapter_bottleneck_dim"], CONFIG["num_llm_layers"], CONFIG["device"])

    # Load Offline-RL Dataset
    offline_dataset_hf = HuggingFaceDataset.load_from_disk(CONFIG["offline_rl_dataset_path"])
    samples_scenarios = process_scenarios(offline_dataset_hf.to_list())
    preference_dataset_raw = create_preference_dataset(samples_scenarios, ds)
    preference_dataset_raw = preference_dataset_raw[:int(len(preference_dataset_raw) * CONFIG["train_dataset_ratio"])]

    # Prepare Datasets and Models for Training
    eval_start_index = len(ds['train']) - CONFIG["num_llm_eval_samples"]
    eval_llm_dataset_hf = ds['train'].select(range(eval_start_index, len(ds['train'])))

    actor_model_instance = Actor(
        base_bert_model=actor_base_model.model,
        head_hidden_dim=CONFIG["actor_head_hidden_dim"],
        num_llm_layers_actor=CONFIG["num_llm_layers"], # num_llm_layers from the actual LLM
        log_std_min=CONFIG["log_std_min"],
        log_std_max=CONFIG["log_std_max"],
    )

    preference_dataset = PruningPreferenceDataset(
        preference_dataset_raw,
        actor_tokenizer,
        CONFIG["num_llm_layers"],
        CONFIG["actor_max_seq_length"]
    )

    if CONFIG["wandb_enabled"]:
        wandb.init(project=CONFIG["wandb_project"], id=CONFIG["wandb_run_id"], config=CONFIG)

    # Train actor model with ORPO
    trained_actor = train_orpo_actor(
        actor_model=actor_model_instance,
        train_pref_dataset=preference_dataset,
        num_total_llm_layers_train=CONFIG["num_llm_layers"],
        model_llm_train=llm_model,
        tokenizer_llm_train=llm_tokenizer,
        adapters_llm_train=llm_adapters,
        eval_dataset_llm=eval_llm_dataset_hf,
        tokenizer_actor_train=actor_tokenizer,
        epochs_actor=CONFIG["orpo_epochs"],
        batch_size_actor=CONFIG["orpo_batch_size"],
        lr_actor=CONFIG["orpo_lr"],
        beta_orpo_actor=CONFIG["orpo_beta"],
        current_device=CONFIG["device"],
        max_seq_len_actor_train=CONFIG["actor_max_seq_length"],
        eval_config_batches=CONFIG["eval_every_n_batches"],
        gradient_clip=CONFIG["gradient_clip_norm"],
        checkpoints_path=f"/workspace/checkpoints/{CONFIG['wandb_run_id']}",
        verbose=CONFIG["verbose"],
    )

    # save_actor_to_hf_hub(trained_actor, CONFIG)
