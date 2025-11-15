"""
Preference Data Generation for Router Training

This module generates preference pairs from offline exploration data for training
the router with preference optimization methods (DPO, ORPO). It converts raw
pruning scenarios into pairwise comparisons.

Key Functions:
    - generate_preference_dataset: Create preference pairs from offline data
    - compare_scenarios: Compare two pruning scenarios to determine preference
    - generate_all_pairs: Generate all pairwise comparisons
    - generate_random_pairs: Sample random pairwise comparisons

Preference Criteria:
    Scenario A is preferred over B if:
    1. A has higher accuracy score, OR
    2. Same accuracy but A prunes more layers (more efficient)
"""

import random
from typing import List, Dict, Any
from torch.utils.data import Dataset, DataLoader

def generate_preference_dataset(offline_dataset, preference_generator_for_sample_id):
    preference_dataset = []
    scenarios_by_input = split_by_input(offline_dataset)
    for sample_id, scenarios in scenarios_by_input.items():
        preference_dataset.extend(preference_generator_for_sample_id(scenarios))
    return preference_dataset

def generate_all_pairs(scenarios):
    pairs = []
    for i in range(len(scenarios)):
        for j in range(i+1, len(scenarios)):
            comparison_result = compare_scenarios(scenarios[i], scenarios[j])
            if comparison_result == 1:
                pairs.append(generate_preference_sample(scenarios[i], scenarios[j]))
            elif comparison_result == -1:
                pairs.append(generate_preference_sample(scenarios[j], scenarios[i]))
    return pairs

def generate_random_pairs(scenarios, num_pairs):
    pairs = []
    while len(pairs) < num_pairs:
        scenario_1, scenario_2 = random.sample(scenarios, 2)
        comparison_result = compare_scenarios(scenario_1, scenario_2)
        if comparison_result == 1:
            pairs.append(generate_preference_sample(scenario_1, scenario_2))
        elif comparison_result == -1:
            pairs.append(generate_preference_sample(scenario_2, scenario_1))
    return pairs

def generate_preference_sample(winner_scenario, loser_scenario):
    return {
        "id": winner_scenario['id'],
        "winner_pruned_layers": winner_scenario['pruned_layers'],
        "loser_pruned_layers": loser_scenario['pruned_layers'],
        "winner_score": winner_scenario['score'],
        "loser_score": loser_scenario['score'],
    }

def compare_scenarios(scenario_1, scenario_2):
    """
    Returns:
        1 if scenario_1 is preferred over scenario_2, 
        -1 if scenario_2 is preferred over scenario_1, 
        0 if they are equally preferred.
    """
    if scenario_1['score'] == scenario_2['score']:
        if len(scenario_1['pruned_layers']) > len(scenario_2['pruned_layers']):
            return 1
        elif len(scenario_1['pruned_layers']) < len(scenario_2['pruned_layers']):
            return -1
        else:
            return 0
    elif scenario_1['score'] > scenario_2['score']:
        return 1
    else:
        return -1


def split_by_input(offline_dataset):
    scenarios_by_input = {}
    for scenario in offline_dataset:
        if scenario['id'] not in scenarios_by_input:
            scenarios_by_input[scenario['id']] = []
        scenarios_by_input[scenario['id']].append({
            "id": scenario['id'],
            "pruned_layers": scenario['pruned_layers'],
            "score": scenario['score'],
        })
    return scenarios_by_input


def split_train_test(preference_dataset, test_ratio):
    return preference_dataset[:int(len(preference_dataset) * (1 - test_ratio))], preference_dataset[int(len(preference_dataset) * (1 - test_ratio)):]

class PreferenceDataset(Dataset):
    """PyTorch Dataset for preference data."""
    def __init__(self, preference_data: List[Dict[str, Any]], function_calling_dataset: List[Dict[str, Any]]):
        self.data = preference_data
        self.function_calling_dataset = function_calling_dataset

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
            "winner_layers": sample["winner_pruned_layers"],
            "loser_layers": sample["loser_pruned_layers"]
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
            return_tensors="pt"
        )

        return {
            "input_ids": tokenized_inputs["input_ids"],
            "attention_mask": tokenized_inputs["attention_mask"],
            "winner_layers": [item["winner_layers"] for item in batch],
            "loser_layers": [item["loser_layers"] for item in batch]
        }

def get_dataloader(preference_dataset, batch_size, tokenizer):
    return DataLoader(
        preference_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=DataCollatorWithPadding(tokenizer),
        num_workers=8,
    )
