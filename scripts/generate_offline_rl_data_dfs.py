import random
import numpy as np
import torch
from datasets import Dataset

from src.llm import load_llm, load_and_split_dataset
from src.adapter import load_adapters
from src.offline_rl_data import generate_offline_rl_data_dfs

CONFIG = {
    "llm_model_name": "Salesforce/xLAM-2-1b-fc-r",
    "dataset_name": "Salesforce/xlam-function-calling-60k",
    "dataset_test_size": 100,
    "adapter_bottleneck_dim": 64,
    "num_samples": 5000,
    "scenarios_per_sample": 20,
}

seed = 23
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

if __name__ == "__main__":
    llm_model, llm_tokenizer = load_llm(CONFIG)
    test_dataset, train_dataset = load_and_split_dataset(CONFIG["dataset_name"], test_size=CONFIG["dataset_test_size"])

    adapter_path_template = f'/workspace/models/adapters/adapter_{{i}}.pth'
    llm_adapters = load_adapters(
        adapter_path_template=adapter_path_template,
        adapter_io_dim=CONFIG["llm_hidden_dim"],
        adapter_bottleneck_dim=CONFIG["adapter_bottleneck_dim"],
        num_llm_layers=CONFIG["llm_num_layers"],
        device=torch.device("cuda"),
    )

    offline_dataset_list = generate_offline_rl_data_dfs(train_dataset, llm_model, llm_tokenizer, llm_adapters, CONFIG["llm_num_layers"], CONFIG["num_samples"], CONFIG["scenarios_per_sample"])
    Dataset.from_list(offline_dataset_list).save_to_disk(f"/workspace/datasets/dfs_{CONFIG['num_samples']}_samples_{CONFIG['scenarios_per_sample']}_scenarios")
    