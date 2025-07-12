import random
from tqdm import tqdm
from datasets import Dataset

from src.llm import generate_llm_output
from src.prune import prune_layers, LLMPruner
from src.evaluation import partial_match_score, json_match_score


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
        score = partial_match_score(output, sample['answers'])

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



def generate_offline_rl_data_dfs(ds, model, tokenizer, adapters, num_layers, num_samples, scenrios_per_sample):
    global sample_id_threshold

    offline_dataset_list = []
    for sample_id in tqdm(range(num_samples)):
        if sample_id % 100 == 99:
            Dataset.from_list(offline_dataset_list).save_to_disk(f"offline_20/{sample_id//100}")
        offline_dataset_list = []

        sample = ds[sample_id]
        sample_id_threshold = scenrios_per_sample

        output = generate_llm_output(sample, model, tokenizer)
        score = partial_match_score(output, sample['answers'])
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


def generate_offline_rl_data_random(ds, model, tokenizer, adapters, num_layers, num_samples, scenarios_per_num_pruned_layers, stop_adding_layers_threshold, verbose=False):
    offline_dataset_list = []
    for sample_id in tqdm(range(num_samples+1), desc="Generating offline RL data with random scenarios"):
        sample = ds[sample_id]
        output = generate_llm_output(sample, model, tokenizer)
        if json_match_score(output, sample['answers']) < 1:
            continue
        
        offline_dataset_list.append({
            "id": sample['id'],
            "pruned_layers": [],
            "output": output,
            "score": 1.0,
        })

        for num_pruned_layers in range(1, num_layers):
            scores_sum = 0
            for _ in range(scenarios_per_num_pruned_layers):
                pruned_layers = random.sample(range(2, 27), num_pruned_layers)  # because pruning 0, 1, 27 layers degrades too much
                with LLMPruner(model, adapters) as pruner:
                    pruner.prune_model(pruned_layers)
                    output = generate_llm_output(sample, model, tokenizer)
                score = json_match_score(output, sample['answers'])
                scores_sum += score
                offline_dataset_list.append({
                    "id": sample['id'],
                    "pruned_layers": pruned_layers,
                    "output": output,
                    "score": score,
                })
            average_score = scores_sum / scenarios_per_num_pruned_layers
            if verbose:
                print(f"Sample {sample['id']}, num_pruned_layers: {num_pruned_layers}, average_score: {average_score}")
            if average_score < stop_adding_layers_threshold:  # Stop if the average score is less than the threshold
                break
    
    return offline_dataset_list
