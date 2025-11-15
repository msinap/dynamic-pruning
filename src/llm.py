"""
LLM Utilities Module

This module provides utility functions for loading, tokenizing, and generating outputs
from Large Language Models (LLMs) for function calling tasks. It includes support for
dynamic layer pruning during inference.

Key Functions:
    - load_llm: Load a pre-trained LLM and tokenizer
    - generate_llm_output: Generate text from the LLM
    - generate_llm_output_with_pruning: Generate text with dynamic layer pruning
    - tokenize_for_llm: Prepare inputs for the LLM with function calling tools
"""

from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import torch
from datasets import load_dataset

from src.prune import LLMPruner

# --- LLM Utilities ---
def load_llm(CONFIG):
    """
    Load a pre-trained Large Language Model and its tokenizer.
    
    This function loads a causal language model with bfloat16 precision and
    automatically distributes it across available devices using accelerate.
    It also updates the CONFIG dictionary with model-specific parameters.
    
    Args:
        CONFIG (dict): Configuration dictionary containing:
            - llm_model_name (str): HuggingFace model identifier
            
    Returns:
        tuple: (llm_model, llm_tokenizer)
            - llm_model: AutoModelForCausalLM instance in eval mode
            - llm_tokenizer: AutoTokenizer instance
            
    Side Effects:
        Updates CONFIG with:
            - llm_num_layers (int): Number of transformer layers in the model
            - llm_hidden_dim (int): Hidden dimension size of the model
    """
    # --- Model and Tokenizer Loading ---
    llm_tokenizer = AutoTokenizer.from_pretrained(CONFIG["llm_model_name"])
    llm_model = AutoModelForCausalLM.from_pretrained(
        CONFIG["llm_model_name"],
        torch_dtype=torch.bfloat16,
        device_map="auto" # Uses accelerate for device mapping
    )
    llm_model.eval() # Set LLM to eval mode by default

    CONFIG["llm_num_layers"] = len(llm_model.model.layers)
    CONFIG["llm_hidden_dim"] = llm_model.model.embed_tokens.weight.shape[-1]
    
    return llm_model, llm_tokenizer

def generate_prompt_str(sample, tokenizer_llm):
    """
    Generate a formatted prompt string for function calling.
    
    Args:
        sample (dict): Dataset sample containing 'query' and 'tools' fields
        tokenizer_llm: Tokenizer with chat template support
        
    Returns:
        str: Formatted prompt string with chat template and tools
    """
    messages = [{"role": "user", "content": sample['query']}]
    tools = json.loads(sample['tools'])
    return tokenizer_llm.apply_chat_template(messages, tools=tools, add_generation_prompt=True, tokenize=False)

def tokenize_for_llm(sample, tokenizer_llm, device):
    """
    Tokenize a function calling sample for LLM input.
    
    Args:
        sample (dict): Dataset sample containing 'query' and 'tools' fields
        tokenizer_llm: Tokenizer with chat template support
        device: Target device for tensors (e.g., 'cuda', 'cpu')
        
    Returns:
        dict: Tokenized inputs ready for model forward pass
    """
    messages = [{"role": "user", "content": sample['query']}]
    tools = json.loads(sample['tools'])
    return tokenizer_llm.apply_chat_template(
        messages, tools=tools, add_generation_prompt=True, return_dict=True, return_tensors="pt"
    ).to(device)

def generate_llm_output(sample, model_llm, tokenizer_llm, output_hidden_states=False):
    """
    Generate text output from the LLM for a given sample.
    
    Args:
        sample (dict): Dataset sample containing 'query' and 'tools' fields
        model_llm: Loaded LLM model
        tokenizer_llm: Tokenizer for the LLM
        output_hidden_states (bool): If True, return hidden states for all layers
        
    Returns:
        str or tuple: 
            - If output_hidden_states=False: Generated text
            - If output_hidden_states=True: (generated_text, hidden_states)
    """
    inputs = tokenize_for_llm(sample, tokenizer_llm, model_llm.device)
    input_len = inputs["input_ids"].shape[-1]
    with torch.no_grad():
        outputs = model_llm.generate(
            **inputs,
            max_new_tokens=150,
            num_return_sequences=1,
            output_hidden_states=output_hidden_states,
            return_dict_in_generate=True,
        )
    output_ids = outputs.sequences[0][input_len:]
    output_text = tokenizer_llm.decode(output_ids, skip_special_tokens=True)
    if output_hidden_states:
        return output_text, outputs.hidden_states
    else:
        return output_text

def generate_llm_output_with_pruning(sample, model_llm, tokenizer_llm, adapters, router, tokenizer_router, verbose=False):
    """
    Generate text output from the LLM with dynamic layer pruning.
    
    This function uses a router model to determine which layers to prune dynamically
    based on the input query. Pruned layers are replaced with lightweight adapters.
    
    Args:
        sample (dict): Dataset sample containing 'query', 'tools', and 'answers' fields
        model_llm: Loaded LLM model
        tokenizer_llm: Tokenizer for the LLM
        adapters (list): List of adapter modules, one per layer
        router: Router model that decides which layers to prune
        tokenizer_router: Tokenizer for the router model
        verbose (bool): If True, print pruning information
        
    Returns:
        str: Generated text with pruned model
        
    Notes:
        The router outputs:
        1. Layer scores: Importance scores for each layer
        2. Pruning ratio distribution: Normal(mu, sigma) for the fraction of layers to prune
        
        Higher scoring layers are more likely to be pruned.
    """
    inputs = tokenizer_router(sample['query'], return_tensors="pt", padding=True, truncation=True, return_token_type_ids=False).to(router.bert.device)
    with torch.no_grad():
        layers_scores, mu_ratio, log_std_ratio = router(**inputs)
    
    std_ratio = torch.exp(log_std_ratio)
    ratio_dist = torch.distributions.Normal(mu_ratio, std_ratio)
    ratio = ratio_dist.sample().item()
    num_pruned_layers = int(ratio * len(model_llm.model.layers))
    pruned_layers = torch.topk(layers_scores, num_pruned_layers).indices.tolist()[0]

    if verbose:
        # print(f"Pruning {num_pruned_layers} layers")
        print(f"Pruned layers: {pruned_layers}")
        # print(f"Ratio: {ratio}")
        # print(f"Mu ratio: {mu_ratio}")
        # print(f"Std ratio: {std_ratio}")
        # print(f"Layers scores: {layers_scores}")

    with LLMPruner(model_llm, adapters) as pruner:
        pruner.prune_model(pruned_layers)
        return generate_llm_output(sample, model_llm, tokenizer_llm)
    
def load_and_split_dataset(dataset_name, test_size):
    """
    Load a dataset and split it into test and train sets.
    
    Args:
        dataset_name (str): HuggingFace dataset identifier
        test_size (int): Number of samples to use for testing
        
    Returns:
        tuple: (test_dataset, train_dataset)
            - test_dataset: Last test_size samples
            - train_dataset: All samples except the last test_size
    """
    dataset = load_dataset(dataset_name)['train']
    return dataset.select(range(len(dataset) - test_size, len(dataset))), dataset.select(range(len(dataset) - test_size))

def load_dataset_list(dataset_name):
    """
    Load a dataset and convert it to a list of samples.
    
    Args:
        dataset_name (str): HuggingFace dataset identifier
        
    Returns:
        list: Dataset samples as a list of dictionaries
    """
    return load_dataset(dataset_name)['train'].to_list()
