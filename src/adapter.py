"""
Adapter Module for Dynamic LLM Pruning

This module provides lightweight adapter networks that can replace full transformer layers
during inference. Adapters learn to mimic the behavior of their corresponding layers,
enabling efficient layer pruning with minimal accuracy loss.

Key Components:
    - Adapter: Bottleneck adapter architecture with residual connections
    - train_adapters: Training procedure using layer-wise distillation
    - evaluate_adapters_separately: Evaluate each adapter's quality independently
"""

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
    """
    Lightweight bottleneck adapter for replacing transformer layers.
    
    Architecture: Input -> Linear(down) -> ReLU -> Linear(up) -> Residual Add -> Output
    
    The adapter uses a bottleneck architecture to reduce parameters while maintaining
    expressiveness. Weights are initialized to produce near-identity transformations
    at the start of training.
    
    Args:
        io_dim (int): Input and output dimension (should match layer hidden size)
        bottleneck_dim (int): Dimension of the bottleneck layer (e.g., 64)
        
    Attributes:
        attention_type (str): Required attribute for compatibility with Qwen2 models
    """
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
        """
        Forward pass with residual connection.
        
        Args:
            x: Input tensor or tuple of tensors (matches LLM layer output format)
            **_: Additional keyword arguments (ignored, for compatibility)
            
        Returns:
            tuple: Output tensor(s) in same format as input
        """
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
    """
    Load pre-trained adapter modules from disk.
    
    Args:
        adapter_path_template (str): Path template with {i} placeholder for layer index
            Example: '/models/adapter/adapter_{i}.pth'
        adapter_io_dim (int): Input/output dimension of adapters
        adapter_bottleneck_dim (int): Bottleneck dimension of adapters
        num_llm_layers (int): Number of layers in the LLM
        device: Target device for adapter modules
        
    Returns:
        list: List of loaded Adapter modules, one per layer
    """
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
    """
    Train adapter modules using layer-wise knowledge distillation.
    
    This function trains adapters to mimic the input-output behavior of their
    corresponding transformer layers. It only uses samples where the LLM produces
    correct outputs, ensuring adapters learn good layer approximations.
    
    Training Strategy:
        1. Generate LLM output with hidden states for a sample
        2. Skip sample if LLM output is incorrect
        3. For each layer, collect (input, output) pairs from all generation steps
        4. Train adapter to minimize MSE between adapter output and layer output
        5. Evaluate adapters periodically by pruning individual layers
    
    Args:
        adapters (list): List of Adapter modules to train
        train_dataset (list): Training samples with 'query', 'tools', 'answers'
        num_layers (int): Number of layers in the LLM
        tokenizer: LLM tokenizer
        model: LLM model
        lr (float): Learning rate for Adam optimizer
        eval_every_n_steps (int): Evaluation frequency
        eval_dataset: Dataset for evaluation
        verbose (bool): If True, print detailed training information
    """
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
    """
    Save trained adapter modules to disk.
    
    Args:
        adapters (list): List of trained Adapter modules
        run_id (str): Unique identifier for this training run
        path_template (str): Directory path template with {run_id} placeholder
            Example: '/models/adapter/{run_id}'
    """
    os.makedirs(f'{path_template.format(run_id=run_id)}', exist_ok=True)
    for i, adapter in enumerate(adapters):
        torch.save(adapter.state_dict(), f'{path_template.format(run_id=run_id)}/adapter_{i}.pth')

def evaluate_adapters_separately(adapters, dataset, num_layers, tokenizer, model, verbose=False):
    """
    Evaluate the quality of each adapter by testing single-layer pruning.
    
    This function tests each adapter individually by replacing only that layer
    and measuring the model's performance on a dataset. This helps identify
    which layers can be safely pruned and which are more critical.
    
    Args:
        adapters (list): List of trained Adapter modules
        dataset: Evaluation dataset
        num_layers (int): Number of layers in the LLM
        tokenizer: LLM tokenizer
        model: LLM model
        verbose (bool): If True, print detailed evaluation information
        
    Returns:
        list: Scores for each layer (higher = better adapter quality)
    """
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



