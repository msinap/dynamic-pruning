"""
Filtered Behavioral Cloning (FBC) for Router Training

This module implements Filtered Behavioral Cloning, a supervised learning approach
for training the router model. FBC trains on successful pruning configurations
observed from offline exploration, filtering out poor-performing scenarios.

Key Components:
    - BehaviorDataset: PyTorch dataset for behavioral cloning data
    - fbc_loss_function: MSE-based loss for supervised router training
    - train_fbc: Training loop with evaluation
    - filter_behavior_dataset: Various filtering strategies for selecting good behaviors

Training Approach:
    The router learns to directly predict successful pruning configurations
    (which layers to prune and how many) from offline data collected by
    systematically exploring the pruning space.
"""

from typing import List, Dict, Any
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset
import torch

from src.llm import generate_llm_output_with_pruning
from src.preference_data_generation import split_by_input

def filter_behavior_dataset(behavior_dataset, filter_scenarios_function):
    filtered_behavior_dataset = []
    scenarios_by_input = split_by_input(behavior_dataset)
    for sample_id, scenarios in scenarios_by_input.items():
        filtered_behavior_dataset.extend(filter_scenarios_function(scenarios))
    return filtered_behavior_dataset

def filter_scenarios_by_number_of_pruned_layers(scenarios, threshold_with_max_num_pruned_layers=0.6):
    max_num_pruned_layers = max([len(scenario['pruned_layers']) for scenario in scenarios if scenario['score'] > 0.99])
    return [scenario for scenario in scenarios if scenario['score'] > 0.99 and len(scenario['pruned_layers']) >= max_num_pruned_layers * threshold_with_max_num_pruned_layers]

def filter_scenarios_fixed_number(scenarios, number_of_scenarios=5):
    sorted_scenarios = sorted(scenarios, key=lambda x: (x['score'], x['pruned_layers']), reverse=True)
    top_scenarios = sorted_scenarios[:number_of_scenarios]
    return [scenario for scenario in top_scenarios if scenario['score'] > 0.99]


class BehaviorDataset(Dataset):
    """PyTorch Dataset for behavior data."""
    def __init__(
        self,
        behavior_data: List[Dict[str, Any]],
        function_calling_dataset: List[Dict[str, Any]],
        num_llm_layers: int,
    ):
        self.data = behavior_data
        self.function_calling_dataset = function_calling_dataset
        self.num_llm_layers = num_llm_layers

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.data[idx]
        query = self.function_calling_dataset[sample["id"]]["query"]
        tools = self.function_calling_dataset[sample["id"]]["tools"]
        input = f"tools: {tools}\nquery: {query}"
        return {
            "id": sample["id"],
            "input": input,
            "ratio": len(sample["pruned_layers"]) / self.num_llm_layers,
            "is_layer_pruned": [1 if layer in sample["pruned_layers"] else 0 for layer in range(self.num_llm_layers)],
        }

class DataCollatorWithPadding:
    def __init__(self, tokenizer: Any):
        self.tokenizer = tokenizer

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Tokenizes and batches the data."""
        input = [item["input"] for item in batch]

        tokenized_inputs = self.tokenizer(
            input,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids": tokenized_inputs["input_ids"],
            "attention_mask": tokenized_inputs["attention_mask"],
            "ratio": torch.tensor([item["ratio"] for item in batch]),
            "is_layer_pruned": torch.tensor([item["is_layer_pruned"] for item in batch]),
        }

def get_dataloader(
    function_calling_dataset,
    filtered_behavior_dataset_list,
    router_tokenizer,
    batch_size,
    num_llm_layers,
):
    filtered_behavior_dataset = BehaviorDataset(
        behavior_data=filtered_behavior_dataset_list,
        function_calling_dataset=function_calling_dataset,
        num_llm_layers=num_llm_layers,
    )
    return DataLoader(
        filtered_behavior_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=DataCollatorWithPadding(router_tokenizer),
        num_workers=8,
    )



def fbc_loss_function(
    batch,
    router,
    alpha: float = 0.01,
    verbose: bool = False,
):
    layers_scores, mu_ratio, log_std_ratio = router(
        input_ids=batch['input_ids'],
        attention_mask=batch['attention_mask'],
    )
    mu_loss = F.mse_loss(mu_ratio, batch['ratio'].to(dtype=torch.bfloat16))
    layers_scores_loss = F.mse_loss(layers_scores, batch['is_layer_pruned'].to(dtype=torch.bfloat16))
    return mu_loss + layers_scores_loss * alpha
    

def train_fbc(
    router,
    router_tokenizer,

    # training
    learning_rate: float,
    train_dataloader: DataLoader,
    log_every_n_steps: int,

    # evaluation
    eval_dataset: List[Dict[str, Any]],
    eval_every_n_steps: int,
    llm_model,
    llm_tokenizer,
    adapters,
    score_funcs,
):
    optimizer = AdamW(router.parameters(), lr=learning_rate)

    router.train()
    for step, batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
        batch = {k: v.to(router.bert.device) for k, v in batch.items()}

        optimizer.zero_grad()
        verbose = True if step % log_every_n_steps == log_every_n_steps - 1 else False
        loss = fbc_loss_function(
            batch=batch,
            router=router,
            verbose=verbose,
        )
        loss.backward()
        optimizer.step()

        if step % eval_every_n_steps == eval_every_n_steps - 1:
            scores = {}
            for sample in eval_dataset:
                output = generate_llm_output_with_pruning(
                    sample=sample,
                    model_llm=llm_model,
                    tokenizer_llm=llm_tokenizer,
                    adapters=adapters,
                    router=router,
                    tokenizer_router=router_tokenizer,
                    verbose=True,
                )
                print(output)
                print(sample['answers'])
                for score_func in score_funcs:
                    if score_func.__name__ not in scores:
                        scores[score_func.__name__] = []
                    score = score_func(output, sample['answers'])
                    scores[score_func.__name__].append(score)
                    print(f"{score_func.__name__}: {score:.4f}")
                
            for score_name, results in scores.items():
                print(f"{score_name}: {np.mean(results):.4f}")
            print("-" * 100)
