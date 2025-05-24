from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import torch
from copy import deepcopy

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

    # --- Update Config with Model Dimensions ---
    CONFIG["num_llm_layers"] = len(llm_model.model.layers)
    CONFIG["adapter_io_dim"] = llm_model.model.embed_tokens.weight.shape[-1]
    
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

def generate_llm_output(sample, model_llm, tokenizer_llm, return_token_count=False, **_):
    inputs = tokenize_for_llm(sample, tokenizer_llm, model_llm.device)
    input_len = inputs["input_ids"].shape[-1]
    with torch.no_grad():
        outputs = model_llm.generate(
            **inputs,
            max_new_tokens=150,
            num_return_sequences=1,
            do_sample=False,
        )
    output_ids = outputs[0][input_len:]
    output_text = tokenizer_llm.decode(output_ids, skip_special_tokens=True)
    if return_token_count:
        return output_text, len(output_ids)
    return output_text


class LLMPruner:
    def __init__(self, model_llm, model_adapters):
        self.llm_model_full = model_llm
        self.adapters_list = model_adapters
        self.original_layers_backup = {}

    def prune_model(self, layer_indices_to_replace_with_adapters):
        self._restore_original_layers() # Restore any previous state
        self.original_layers_backup.clear()

        for layer_idx in layer_indices_to_replace_with_adapters:
            if 0 <= layer_idx < len(self.llm_model_full.model.layers):
                self.original_layers_backup[layer_idx] = self.llm_model_full.model.layers[layer_idx]
                self.llm_model_full.model.layers[layer_idx] = self.adapters_list[layer_idx]
            else:
                print(f"Warning: Invalid layer index {layer_idx} for pruning.")

    def _restore_original_layers(self):
        for layer_idx, original_layer in self.original_layers_backup.items():
            if 0 <= layer_idx < len(self.llm_model_full.model.layers):
                self.llm_model_full.model.layers[layer_idx] = original_layer
        self.original_layers_backup.clear()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._restore_original_layers() # Ensure restoration on exit


def prune_layers(model, layer_idxs, adapters):
    pruned_model = deepcopy(model)    
    for layer_idx in layer_idxs:
        #pruned_model.model.layers[layer_idx] = Identity()
        pruned_model.model.layers[layer_idx] = deepcopy(adapters[layer_idx])
    return pruned_model

