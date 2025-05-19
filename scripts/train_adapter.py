import wandb
import random
from datasets import load_dataset
import numpy as np
import torch

from code.llm import *
from code.adapter import *

CONFIG = {
    "llm_model_name": "Salesforce/xLAM-2-1b-fc-r",
    "lr": 1e-4,
    "bottleneck_dim": 64,
    "base_model": "xlam2-1b",
    "dataset": "Salesforce/xlam-function-calling-60k",
    "wandb_run_name": f"adapter_run_{random.randint(1000,9999)}",
    "seed": 23,
    "num_samples_to_train": 10,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
}

# --- Reproducibility ---
random.seed(CONFIG["seed"])
np.random.seed(CONFIG["seed"])
torch.manual_seed(CONFIG["seed"])
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(CONFIG["seed"])

if __name__ == "__main__":
    llm_model, llm_tokenizer = load_llm(CONFIG)
    ds = load_dataset(CONFIG["dataset"])

    adapters = [Adapter(CONFIG["adapter_io_dim"], CONFIG["bottleneck_dim"]).to(device=CONFIG["device"], dtype=torch.bfloat16) for _ in range(CONFIG["num_llm_layers"])]

    wandb.init(project="layer-pruning-adapter", id=CONFIG["wandb_run_name"], config=CONFIG)
    train_adapters(adapters, ds, CONFIG["num_llm_layers"], llm_tokenizer, llm_model, CONFIG["num_samples_to_train"], CONFIG["device"], CONFIG["lr"])
    wandb.finish()
    save_adapters(adapters, CONFIG["wandb_run_name"])
    
