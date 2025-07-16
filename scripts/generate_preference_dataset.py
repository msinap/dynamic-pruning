import random
import numpy as np
import torch
from datasets import Dataset
from src.llm import load_and_split_dataset
from src.preference_data_generation import generate_preference_dataset, generate_random_pairs, generate_all_pairs

CONFIG = {
    "offline_random_dataset_path": "/workspace/datasets/random_5000_samples_5_scenarios",
    "offline_dfs_dataset_path": "/workspace/datasets/dfs_5000_samples_20_scenarios",
}

seed = 23
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

if __name__ == "__main__":
    
    offline_random_dataset = Dataset.load_from_disk(CONFIG["offline_random_dataset_path"]).to_list()
    offline_dfs_dataset = Dataset.load_from_disk(CONFIG["offline_dfs_dataset_path"]).to_list()


    preference_dataset = generate_preference_dataset(
        offline_random_dataset,
        generate_all_pairs,
    )
    Dataset.from_list(preference_dataset).save_to_disk(f"/workspace/datasets/preference_dataset_{CONFIG['offline_random_dataset_path'].split('/')[-1]}_all_pairs")
    
    preference_dataset = generate_preference_dataset(
        offline_dfs_dataset,
        generate_all_pairs,
    )
    Dataset.from_list(preference_dataset).save_to_disk(f"/workspace/datasets/preference_dataset_{CONFIG['offline_dfs_dataset_path'].split('/')[-1]}_all_pairs")

