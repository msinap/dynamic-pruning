import random
import torch
import math
import torch.nn.functional as F

from code.llm import *
from code.prune import *
from code.adapter import *
from code.evaluation import *


def create_fbc_dataset(samples_scenarios):
    sft_dataset = []

    for _, sample_scenarios in samples_scenarios.items():
        for scenario in sample_scenarios['scenarios']:
            if abs(scenario['score'] - sample_scenarios['max_score']) < 1e-6 and len(scenario['pruned_layers']) >= sample_scenarios['max_pruned_layers'] * 0.8:
                sft_dataset.append({
                    "id": scenario['id'],
                    "pruned_layers": scenario['pruned_layers'],
                    "output": scenario['output'],
                    "score": scenario['score'],
                })

    random.shuffle(sft_dataset)
    return sft_dataset


def calc_sft_loss(pruned_layers, logits, num_layers, device):
    answer = torch.tensor([1.0 if layer_id in pruned_layers else 0.0 for layer_id in range(num_layers)], device=device)
    loss = F.binary_cross_entropy(logits, answer)
    return loss

def calc_kl_loss(pruned_layers, scores_log, num_layers, device):
    answer = torch.tensor([1.0 / len(pruned_layers) if layer_id in pruned_layers else 0.0 for layer_id in range(num_layers)], device=device)
    loss = F.kl_div(scores_log, answer, reduction='batchmean')
    return loss

def calc_mse_loss(ratio, pruned_layers, num_layers, device):
    answer = torch.tensor(len(pruned_layers) / num_layers, device=device)
    loss = F.mse_loss(ratio, answer)
    if ratio > answer:
        loss *= 10.0
    return loss

def sort_indices_high_to_low(arr):
    indexed_arr = list(enumerate(arr))
    sorted_arr = sorted(indexed_arr, key=lambda x: x[1], reverse=True)
    return [index for index, _ in sorted_arr], [round(math.exp(value), 2) for _, value in sorted_arr]


def prune_by_probability_ratio(scores, ratio, model, adapters):
    sorted_layer_idxs, _ = sort_indices_high_to_low(scores)
    num_pruned_layers = int(len(sorted_layer_idxs) * ratio.item())
    layers_to_prune = sorted_layer_idxs[:num_pruned_layers]
    return prune_layers(model, layers_to_prune, adapters), layers_to_prune

def train_fbc(
        model, 
        dataset, 
        actor, 
        actor_tokenizer, 
        is_eval, 
        do_calc_score, 
        ds, 
        device, 
        adapters, 
        num_layers, 
        tokenizer
    ):
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=1e-5)

    for scenario in tqdm(dataset):
        sample = ds['train'][scenario['id']]
        input = sample['tools'] + '\n' + sample['query']
        inputs = actor_tokenizer(input, return_tensors="pt").to(device=device)
        print(inputs)
        if is_eval:
            with torch.no_grad():
                scores_log, ratio = actor(**inputs)
        else:
            scores_log, ratio = actor(**inputs)
        scores_log = scores_log.squeeze(0)
        ratio = ratio.squeeze(0)

        if len(scenario['pruned_layers']) > 0:
            kl_loss = calc_kl_loss(scenario['pruned_layers'], scores_log)
        else:
            kl_loss = torch.tensor(0.0, device=device)
        ratio_loss = calc_mse_loss(ratio, scenario['pruned_layers'])
        loss = kl_loss + ratio_loss * 10.0

        if do_calc_score:
            pruned_model, layer_idxs = prune_by_probability_ratio(scores_log.tolist(), ratio, model, adapters)
            output = generate_llm_output(sample, pruned_model, tokenizer)
            score = calc_score(output, sample['answers'])
            print(f"score: {score:.2f} vs {scenario['score']:.2f}, loss: {loss.item():.2f}, kl_loss: {kl_loss.item():.2f}, ratio_loss: {10.0 * ratio_loss.item():.2f}, ratio: {ratio.item():.2f} vs {len(scenario['pruned_layers']) / num_layers:.2f}, pruned_layers: {layer_idxs} vs {scenario['pruned_layers']}")
            print(sort_indices_high_to_low(scores_log.tolist()))
        
        if not is_eval:
            actor_optimizer.zero_grad()
            loss.backward()
            actor_optimizer.step()

