from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import torch
from datasets import load_dataset

from src.prune import LLMPruner

# --- LLM Utilities ---
def load_llm(CONFIG):
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
    messages = [{"role": "user", "content": sample['query']}]
    tools = json.loads(sample['tools'])
    return tokenizer_llm.apply_chat_template(messages, tools=tools, add_generation_prompt=True, tokenize=False)

def tokenize_for_llm(sample, tokenizer_llm, device):
    messages = [{"role": "user", "content": sample['query']}]
    tools = json.loads(sample['tools'])
    return tokenizer_llm.apply_chat_template(
        messages, tools=tools, add_generation_prompt=True, return_dict=True, return_tensors="pt"
    ).to(device)

def generate_llm_output(sample, model_llm, tokenizer_llm, output_hidden_states=False):
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
    dataset = load_dataset(dataset_name)['train']
    return dataset.select(range(len(dataset) - test_size, len(dataset))), dataset.select(range(len(dataset) - test_size))

def load_dataset_list(dataset_name):
    return load_dataset(dataset_name)['train'].to_list()
