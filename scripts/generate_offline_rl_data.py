import random
from datasets import load_dataset
import numpy as np
import torch

from src.llm import *
from src.adapter import *
from src.offline_rl_data import *
from src.evaluation import *

CONFIG = {
    "llm_model_name": "Salesforce/xLAM-2-1b-fc-r",
    "seed": 23,
    "dataset": "Salesforce/xlam-function-calling-60k",
    "adapter_bottleneck_dim": 512,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "num_samples": 1,
    "scenarios_per_sample": 10,
    "run_id_adapters": '4vw4k59d',
}


random.seed(CONFIG["seed"])
np.random.seed(CONFIG["seed"])
torch.manual_seed(CONFIG["seed"])
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(CONFIG["seed"])

if __name__ == "__main__":
    llm_model, llm_tokenizer = load_llm(CONFIG)
    ds = load_dataset(CONFIG["dataset"])

    adapter_path_template = f'/workspace/models/1_{CONFIG["run_id_adapters"]}/adapter_{{i}}.pth'
    llm_adapters = load_adapters(adapter_path_template, CONFIG["adapter_io_dim"], CONFIG["adapter_bottleneck_dim"], CONFIG["num_llm_layers"], CONFIG["device"])

    offline_dataset_list = generate_offline_rl_data(ds, llm_model, llm_tokenizer, llm_adapters, CONFIG["num_llm_layers"], CONFIG["num_samples"], CONFIG["scenarios_per_sample"])
    Dataset.from_list(offline_dataset_list).save_to_disk(f"/workspace/offline_rl/{CONFIG['run_id_adapters']}_{CONFIG['num_samples']}_{CONFIG['scenarios_per_sample']}")
    