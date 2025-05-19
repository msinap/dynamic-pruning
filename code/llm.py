from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import torch

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
    CONFIG["num_llm_layers_explicit"] = len(llm_model.model.layers)
    CONFIG["adapter_io_dim_explicit"] = llm_model.model.embed_tokens.weight.shape[-1]
    
    return llm_model, llm_tokenizer

def generate_prompt_str(sample, tokenizer_llm): # Renamed to avoid conflict, returns string
    messages = [{"role": "user", "content": sample['query']}]
    tools = json.loads(sample['tools'])
    return tokenizer_llm.apply_chat_template(messages, tools=tools, add_generation_prompt=True, tokenize=False)

def tokenize_for_llm(sample, tokenizer_llm, device): # Renamed for clarity
    messages = [{"role": "user", "content": sample['query']}]
    tools = json.loads(sample['tools'])
    return tokenizer_llm.apply_chat_template(
        messages, tools=tools, add_generation_prompt=True, return_dict=True, return_tensors="pt"
    ).to(device)

def generate_llm_output(sample, model_llm, tokenizer_llm, device): # Renamed
    # Assuming model_llm is already on the correct device(s) via device_map
    inputs = tokenize_for_llm(sample, tokenizer_llm, model_llm.device) # Use model's device for inputs
    input_len = inputs["input_ids"].shape[-1]
    with torch.no_grad():
        outputs = model_llm.generate(
            **inputs,
            max_new_tokens=150,
            num_return_sequences=1,
            do_sample=False,
        )
    output_text = tokenizer_llm.decode(outputs[0][input_len:], skip_special_tokens=True)
    return output_text

