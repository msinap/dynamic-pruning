import random
import torch
import math
import torch.nn.functional as F
from torch.distributions import Normal
from code.llm import *
from code.prune import *
from code.adapter import *
from code.evaluation import *

# TODO: wandb, eval in training, batch, dataloader, 

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
    loss = F.kl_div(scores_log, answer, reduction='sum')
    return loss

def calc_mse_loss(ratio, pruned_layers, num_layers, device):
    answer = torch.tensor(len(pruned_layers) / num_layers, device=device)
    loss = F.mse_loss(ratio, answer)
    if ratio > answer:
        loss *= 10.0
    return loss

def calc_gaussian_nll_loss_cdf(ratio, ratio_dist, target_num_pruned, num_layers):
    y_low = target_num_pruned / num_layers
    y_high = (target_num_pruned + 1) / num_layers
    cdf_y_high = ratio_dist.cdf(y_high)
    cdf_y_low = ratio_dist.cdf(y_low)
    print(f"cdf_y_high: {cdf_y_high.item():.4f}, cdf_y_low: {cdf_y_low.item():.4f}")
    # Add small epsilon to prevent log(0) and clip the loss
    eps = 1e-6
    prob = torch.clamp(cdf_y_high - cdf_y_low, min=eps)
    loss = -torch.log(prob)
    # Clip the final loss value
    max_loss = 10.0  # You can adjust this value
    return torch.clamp(loss, max=max_loss)

def calc_gaussian_nll_loss_pdf(mu_ratio, log_std_ratio, target_ratio):
    # Use a larger epsilon for numerical stability
    var_ratio = torch.exp(2 * log_std_ratio) + 1e-4
    # Add gradient clipping to prevent exploding gradients
    var_ratio = torch.clamp(var_ratio, min=1e-4, max=1.0)
    return F.gaussian_nll_loss(mu_ratio, target_ratio, var_ratio)


def sort_indices_high_to_low(arr):
    indexed_arr = list(enumerate(arr))
    sorted_arr = sorted(indexed_arr, key=lambda x: x[1], reverse=True)
    return [index for index, _ in sorted_arr], [round(math.exp(value), 2) for _, value in sorted_arr]


def prune_by_probability_ratio(scores, ratio, model, adapters):
    sorted_layer_idxs, _ = sort_indices_high_to_low(scores)
    num_pruned_layers = round(len(sorted_layer_idxs) * ratio.item())
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
        if is_eval:
            with torch.no_grad():
                layers_log_probs, mu_ratio, log_std_ratio = actor(**inputs)
        else:
            layers_log_probs, mu_ratio, log_std_ratio = actor(**inputs)
        scores_log = layers_log_probs.squeeze(0)
        mu_ratio = mu_ratio.squeeze(0)
        log_std_ratio = log_std_ratio.squeeze(0)
        ratio_dist = Normal(mu_ratio, log_std_ratio.exp())
        ratio = ratio_dist.sample()

        if len(scenario['pruned_layers']) > 0:
            kl_loss = calc_kl_loss(scenario['pruned_layers'], scores_log, num_layers, device)
        else:
            kl_loss = torch.tensor(0.0, device=device)
        ratio_loss = calc_gaussian_nll_loss_pdf(mu_ratio, log_std_ratio, len(scenario['pruned_layers']) / num_layers)
        loss = kl_loss + ratio_loss * 1.0 

        if do_calc_score:
            pruned_model, layer_idxs = prune_by_probability_ratio(scores_log.tolist(), ratio, model, adapters)
            output = generate_llm_output(sample, pruned_model, tokenizer)
            score = calc_score(output, sample['answers'])
            print(f"score: {score:.2f} vs {scenario['score']:.2f}, loss: {loss.item():.2f}, kl_loss: {kl_loss.item():.2f}, ratio_loss: {ratio_loss.item():.2f}, mu: {mu_ratio.item():.2f}, std: {log_std_ratio.exp().item():.4f}, ratio: {ratio.item():.2f} vs {len(scenario['pruned_layers']) / num_layers:.2f}, pruned_layers: {layer_idxs} vs {scenario['pruned_layers']}")
            print(sort_indices_high_to_low(scores_log.tolist()))
        
        if not is_eval:
            actor_optimizer.zero_grad()
            loss.backward()
            actor_optimizer.step()

