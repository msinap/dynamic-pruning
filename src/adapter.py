import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import wandb
from tqdm import tqdm

from src.llm import generate_llm_output
from src.evaluation import json_match_score, evaluate_model_on_dataset, ratio_function_calls_score
from src.prune import LLMPruner

class Adapter(nn.Module):
    def __init__(self, io_dim, bottleneck_dim):
        super(Adapter, self).__init__()
        self.layer1 = nn.Linear(io_dim, bottleneck_dim)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(bottleneck_dim, io_dim)
        self._initialize_weights(io_dim)
        self.attention_type = "full_attention"  # necessary attribute in Qwen2 model

    def _initialize_weights(self, io_dim):
        std = 1.0 / math.sqrt(io_dim)
        nn.init.normal_(self.layer1.weight, mean=0.0, std=std)
        nn.init.zeros_(self.layer1.bias)
        nn.init.zeros_(self.layer2.weight)
        nn.init.zeros_(self.layer2.bias)

    def forward(self, x, **_): # Match signature of LLM layer forward
        h = self.layer1(x[0] if isinstance(x, tuple) else x) # Handle tuple input from LLM layers
        h = self.relu(h)
        y = self.layer2(h)
        output = (x[0] if isinstance(x, tuple) else x) + y
        # Return in the same format as the original layer (often a tuple)
        if isinstance(x, tuple):
            return (output,) + x[1:]
        return (output,)

def load_adapters(
    adapter_path_template,
    adapter_io_dim,
    adapter_bottleneck_dim,
    num_llm_layers,
    device,
):
    llm_adapters = [
        Adapter(adapter_io_dim, adapter_bottleneck_dim)
        for _ in range(num_llm_layers)
    ]
    for i, adapter_module in enumerate(llm_adapters):
        adapter_module.load_state_dict(torch.load(adapter_path_template.format(i=i), map_location=device))
        adapter_module.to(device, dtype=torch.bfloat16)
    return llm_adapters

def train_adapters(
    adapters,
    train_dataset,
    num_layers,
    tokenizer,
    model,
    lr,
    eval_every_n_steps,
    eval_dataset,
    verbose=False,
):
    optimizers = [torch.optim.Adam(adapters[i].parameters(), lr=lr) for i in range(num_layers)]
    for step, sample in enumerate(tqdm(train_dataset, desc="Training adapters")):
        layers_input_output = [[] for _ in range(num_layers)]

        # generate data
        output_text, hidden_states = generate_llm_output(sample, model, tokenizer, output_hidden_states=True)
        if json_match_score(output_text, sample['answers']) == 0:
            continue

        for step_hidden_states in hidden_states:
            for layer_idx in range(num_layers):
                layers_input_output[layer_idx].append((
                    step_hidden_states[layer_idx].squeeze(0),  # intput
                    step_hidden_states[layer_idx+1].squeeze(0),  # output
                ))

        # train
        layer_losses = {}
        for i in range(num_layers):
            input_batch = torch.cat([input_tensor for input_tensor, _ in layers_input_output[i]], dim=0)
            output_batch = torch.cat([output_tensor for _, output_tensor in layers_input_output[i]], dim=0)
            adapter_output = adapters[i](input_batch)[0]
            loss = F.mse_loss(adapter_output, output_batch)
            layer_losses[f"loss_layer_{i}"] = loss.item()
            optimizers[i].zero_grad()
            loss.backward()
            optimizers[i].step()
        wandb.log(layer_losses)
        if verbose:
            print(f"Layer {i} loss: {loss.item()}")

        # eval
        if step % eval_every_n_steps == 0:
            scores = evaluate_adapters_separately(
                adapters, eval_dataset, num_layers, tokenizer, model, verbose=verbose)
            wandb.log({f"layer_score_{layer_idx}": score for layer_idx, score in enumerate(scores)})
            if verbose:
                print(f"Step {step} scores: {scores}")



def save_adapters(adapters, run_id, path_template):
    os.makedirs(f'{path_template.format(run_id=run_id)}', exist_ok=True)
    for i, adapter in enumerate(adapters):
        torch.save(adapter.state_dict(), f'{path_template.format(run_id=run_id)}/adapter_{i}.pth')

def evaluate_adapters_separately(adapters, dataset, num_layers, tokenizer, model, verbose=False):
    scores = []
    for layer_idx in tqdm(range(num_layers), desc="Evaluating adapters"):
        if verbose:
            print(f"Evaluating layer {layer_idx}")
        with LLMPruner(model, adapters) as pruner:
            pruner.prune_model([layer_idx])
            score = evaluate_model_on_dataset(
                model=pruner.llm_model_full,
                tokenizer=tokenizer,
                dataset=dataset,
                score_funcs=[ratio_function_calls_score],
                verbose=verbose,
            )[ratio_function_calls_score.__name__]
            scores.append(score)
            if verbose:
                print(f"Layer {layer_idx} ratio_function_calls_score: {score}")
    return scores



