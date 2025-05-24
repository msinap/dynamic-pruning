import random
from tqdm import tqdm
from datasets import Dataset

from code.llm import *
from code.adapter import *
from code.evaluation import *


def process_scenarios(offline_dataset_list):
    samples_scenarios = {}
    for scenario in offline_dataset_list:
        sample_id = scenario['id']
        if sample_id not in samples_scenarios:
            samples_scenarios[sample_id] = {
                "scenarios": [], "max_score": -float('inf'), "max_pruned_layers": -1,
            }
        samples_scenarios[sample_id]["scenarios"].append(scenario)
        if scenario['score'] > samples_scenarios[sample_id]['max_score']: # Use > for score
            samples_scenarios[sample_id]['max_score'] = scenario['score']
            # Tie-breaking: if scores are equal, prefer fewer pruned layers
            if len(scenario['pruned_layers']) > samples_scenarios[sample_id]['max_pruned_layers']:
                 samples_scenarios[sample_id]['max_pruned_layers'] = len(scenario['pruned_layers'])
        elif scenario['score'] == samples_scenarios[sample_id]['max_score']:
            if len(scenario['pruned_layers']) < samples_scenarios[sample_id]['max_pruned_layers']: # Prefer fewer layers for same score
                 samples_scenarios[sample_id]['max_pruned_layers'] = len(scenario['pruned_layers'])
    return samples_scenarios


sample_id_threshold = 0

def search_in_pruned_layers_space(prev_pruned_layers, prev_score, branch_factor, sample, adapters, teacher_model, num_layers, tokenizer, offline_dataset_list):
    global sample_id_threshold

    for _ in range(branch_factor):
        new_layers_to_prune = random.sample([i for i in range(num_layers) if i not in prev_pruned_layers], 1)
        pruned_layers = prev_pruned_layers + new_layers_to_prune
        
        output = generate_llm_output(sample, prune_layers(teacher_model, pruned_layers, adapters), tokenizer)
        score = calc_score(output, sample['answers'])

        print(f"Sample {sample['id']}, pruned_layers: {pruned_layers}, score: {score}")
        offline_dataset_list.append({
            "id": sample['id'],
            "pruned_layers": pruned_layers,
            "output": output,
            "score": score,
        })
        sample_id_threshold -= 1
        if sample_id_threshold <= 0:
            return

        if score >= prev_score - 1e-6:
            search_in_pruned_layers_space(pruned_layers, score, max(2, branch_factor//2), sample, adapters, teacher_model, num_layers, tokenizer, offline_dataset_list)



def generate_offline_rl_data(ds, model, tokenizer, adapters, num_layers, num_samples, scenrios_per_sample):
    global sample_id_threshold

    offline_dataset_list = []
    for sample_id in tqdm(range(num_samples)):
        if sample_id % 100 == 99:
            Dataset.from_list(offline_dataset_list).save_to_disk(f"offline_20/{sample_id//100}")
        offline_dataset_list = []

        sample = ds['train'][sample_id]
        sample_id_threshold = scenrios_per_sample

        output = generate_llm_output(sample, model, tokenizer)
        score = calc_score(output, sample['answers'])
        if score <= 0:
            continue
        offline_dataset_list.append({
            "id": sample['id'],
            "pruned_layers": [],
            "output": output,
            "score": score,
        })

        search_in_pruned_layers_space([], score, 8, sample, adapters, model, num_layers, tokenizer, offline_dataset_list)

    return offline_dataset_list
