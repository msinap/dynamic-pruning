import torch
import random
import numpy as np
from datasets import Dataset

from src.llm import load_llm, load_dataset_list
from src.adapter import load_adapters
from src.router import get_router_and_tokenizer
from src.fbc import filter_behavior_dataset, get_dataloader, filter_scenarios_by_number_of_pruned_layers, filter_scenarios_fixed_number, train_fbc
from src.evaluation import partial_match_score, ratio_function_calls_score, json_match_score

CONFIG = {
    "llm_model_name": "Salesforce/xLAM-2-1b-fc-r",
    "dataset_name": "Salesforce/xlam-function-calling-60k",
    "dataset_eval_size": 50,

    "router_base_model_name": "sentence-transformers/all-MiniLM-L6-v2",
    "router_head_hidden_dim": 128,
    "router_log_std_min": -7,
    "router_log_std_max": -3,
    
    "adapter_bottleneck_dim": 64,

    "behavior_dataset_path": "/workspace/datasets/random_5000_samples_5_scenarios",

    "num_samples": 5000,
    "scenarios_per_num_pruned_layers": 5,
    "stop_adding_layers_threshold": 0.1,
    "batch_size": 128,
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

    print("Load & Filter Behavior Dataset")
    behavior_dataset = Dataset.load_from_disk(CONFIG["behavior_dataset_path"]).to_list()
    filtered_behavior_dataset_list = filter_behavior_dataset(behavior_dataset, filter_scenarios_fixed_number)

    print("Prepare Dataset, Dataloader")
    train_dataloader = get_dataloader(
        function_calling_dataset=dataset,
        filtered_behavior_dataset_list=filtered_behavior_dataset_list,
        router_tokenizer=router_tokenizer,
        batch_size=CONFIG["batch_size"],
        num_llm_layers=CONFIG["llm_num_layers"],
    )

    print("Train Filtered Behavioral Cloning")
    train_fbc(
        router=router,
        router_tokenizer=router_tokenizer,

        # training        
        learning_rate=1e-4,
        train_dataloader=train_dataloader,
        log_every_n_steps=100,

        # evaluatio n
        eval_dataset=eval_dataset,
        eval_every_n_steps=100,
        llm_model=llm_model,
        llm_tokenizer=llm_tokenizer,
        adapters=llm_adapters,
        score_funcs=[partial_match_score, ratio_function_calls_score, json_match_score],
    )
