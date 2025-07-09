import random
from datasets import load_dataset
import numpy as np
import torch
from datasets import Dataset as HuggingFaceDataset
from transformers import AutoTokenizer, AutoModelForMaskedLM

from src.llm import *
from src.adapter import *
from src.offline_rl_data import *
from src.evaluation import *
from src.filtered_behavioral_cloning import *
from src.actor import *

CONFIG = {
    "llm_model_name": "Salesforce/xLAM-2-1b-fc-r",
    "seed": 23,
    "dataset": "Salesforce/xlam-function-calling-60k",
    "adapter_bottleneck_dim": 512,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "offline_rl_path": "/workspace/offline_20_5000",
    "adapters_path": '/workspace/1_4vw4k59d',
    "actor_base_model_name": "answerdotai/ModernBERT-base",
    "actor_head_hidden_dim": 256,
    "actor_max_seq_length": 128,
    "log_std_min": -3,
    "log_std_max": -7,
}

random.seed(CONFIG["seed"])
np.random.seed(CONFIG["seed"])
torch.manual_seed(CONFIG["seed"])
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(CONFIG["seed"])

if __name__ == "__main__":
    llm_model, llm_tokenizer = load_llm(CONFIG)
    ds = load_dataset(CONFIG["dataset"])

    adapter_path_template = f'{CONFIG["adapters_path"]}/adapter_{{i}}.pth'
    llm_adapters = load_adapters(adapter_path_template, CONFIG["adapter_io_dim"], CONFIG["adapter_bottleneck_dim"], CONFIG["num_llm_layers"], CONFIG["device"])

    offline_dataset_hf = HuggingFaceDataset.load_from_disk(CONFIG["offline_rl_path"])
    samples_scenarios = process_scenarios(offline_dataset_hf.to_list())
    fbc_dataset = create_fbc_dataset(samples_scenarios)

    actor_tokenizer = AutoTokenizer.from_pretrained(CONFIG["actor_base_model_name"])
    actor_base_model = AutoModelForMaskedLM.from_pretrained(CONFIG["actor_base_model_name"])

    actor = Actor(
        actor_base_model.model,
        CONFIG["actor_head_hidden_dim"],
        CONFIG["num_llm_layers"],
        CONFIG["log_std_min"],
        CONFIG["log_std_max"],
    ).to(CONFIG["device"])

    train_fbc(
        llm_model, 
        fbc_dataset, 
        actor, 
        actor_tokenizer, 
        is_eval=False, 
        do_calc_score=True, 
        ds=ds, 
        device=CONFIG["device"], 
        adapters=llm_adapters, 
        num_layers=CONFIG["num_llm_layers"], 
        tokenizer=llm_tokenizer
    )
