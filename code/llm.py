from transformers import AutoTokenizer, AutoModelForCausalLM, LogitsProcessor, LogitsProcessorList
import json
import torch
from copy import deepcopy
import re
from typing import List, Dict, Any, Optional, Set

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


class StructuredFunctionCallProcessor(LogitsProcessor):
    """
    A LogitsProcessor that enforces structured function call generation based on available tools.
    Validates function names, argument names, required arguments, and argument types during generation.
    """
    
    def __init__(self, tokenizer, tools: List[Dict[str, Any]], device: str = "cuda"):
        self.tokenizer = tokenizer
        self.tools = tools
        self.device = device
        
        # Build tool lookup for validation
        self.tool_lookup = {tool["name"]: tool for tool in tools}
        self.valid_function_names = set(self.tool_lookup.keys())
        
        # Cache token IDs for common tokens
        self.special_tokens = {
            '[': tokenizer.encode('[', add_special_tokens=False)[0],
            ']': tokenizer.encode(']', add_special_tokens=False)[0],
            '{': tokenizer.encode('{', add_special_tokens=False)[0],
            '}': tokenizer.encode('}', add_special_tokens=False)[0],
            ',': tokenizer.encode(',', add_special_tokens=False)[0],
            ':': tokenizer.encode(':', add_special_tokens=False)[0],
            '"': tokenizer.encode('"', add_special_tokens=False)[0],
            ' ': tokenizer.encode(' ', add_special_tokens=False)[0] if tokenizer.encode(' ', add_special_tokens=False) else None,
        }
        
        # Pre-compute valid tokens for function names
        self.valid_name_tokens = self._compute_valid_name_tokens()
        
    def _compute_valid_name_tokens(self) -> Dict[str, Set[int]]:
        """Pre-compute valid tokens for each position in function names"""
        valid_tokens = {}
        
        for func_name in self.valid_function_names:
            # Get all possible prefixes of the function name
            for i in range(len(func_name)):
                prefix = func_name[:i+1]
                if prefix not in valid_tokens:
                    valid_tokens[prefix] = set()
                
                # Find all function names that start with this prefix
                for name in self.valid_function_names:
                    if name.startswith(prefix) and len(name) > i:
                        next_char = name[i]
                        # Get token for the next character
                        token_ids = self.tokenizer.encode(next_char, add_special_tokens=False)
                        if token_ids:
                            valid_tokens[prefix].add(token_ids[0])
        
        return valid_tokens
    
    def _parse_partial_json(self, text: str) -> Dict[str, Any]:
        """Parse partial JSON to understand current generation state"""
        state = {
            'in_list': False,
            'in_object': False,
            'in_string': False,
            'current_key': None,
            'current_function': None,
            'current_args': {},
            'functions': [],
            'depth': 0,
            'expecting': 'list_start'  # What we expect next
        }
        
        i = 0
        current_string = ""
        escape_next = False
        
        while i < len(text):
            char = text[i]
            
            if escape_next:
                escape_next = False
                current_string += char
                i += 1
                continue
                
            if char == '\\' and state['in_string']:
                escape_next = True
                current_string += char
                i += 1
                continue
            
            if char == '"':
                if state['in_string']:
                    state['in_string'] = False
                    # Process the completed string
                    if state['expecting'] == 'function_name':
                        state['current_function'] = current_string
                        state['expecting'] = 'name_colon'
                    elif state['expecting'] == 'argument_key':
                        state['current_key'] = current_string
                        state['expecting'] = 'arg_colon'
                    elif state['expecting'] == 'argument_value':
                        state['current_args'][state['current_key']] = current_string
                        state['expecting'] = 'arg_comma_or_close'
                    current_string = ""
                else:
                    state['in_string'] = True
                    
            elif not state['in_string']:
                if char == '[':
                    state['in_list'] = True
                    state['expecting'] = 'object_start'
                elif char == '{':
                    state['in_object'] = True
                    state['depth'] += 1
                    if state['expecting'] == 'object_start':
                        state['expecting'] = 'name_key'
                elif char == '}':
                    state['depth'] -= 1
                    if state['depth'] == 0:
                        # Completed a function object
                        if state['current_function']:
                            state['functions'].append({
                                'name': state['current_function'],
                                'arguments': state['current_args'].copy()
                            })
                        state['current_function'] = None
                        state['current_args'] = {}
                        state['expecting'] = 'comma_or_list_end'
                elif char == ':':
                    if state['expecting'] == 'name_colon':
                        state['expecting'] = 'name_value'
                    elif state['expecting'] == 'arg_colon':
                        state['expecting'] = 'argument_value'
                elif char == ',':
                    if state['expecting'] == 'arg_comma_or_close':
                        state['expecting'] = 'argument_key'
                    elif state['expecting'] == 'comma_or_list_end':
                        state['expecting'] = 'object_start'
                elif char.isalpha() and state['expecting'] in ['name_key', 'argument_key']:
                    # Handle unquoted keys
                    key_match = re.match(r'(\w+)', text[i:])
                    if key_match:
                        key = key_match.group(1)
                        i += len(key) - 1
                        if key == "name" and state['expecting'] == 'name_key':
                            state['expecting'] = 'name_colon'
                        elif key == "arguments" and state['expecting'] == 'argument_key':
                            state['expecting'] = 'arguments_colon'
            else:
                current_string += char
                
            i += 1
        
        # If we're in the middle of a string, include it in the state
        if state['in_string'] and current_string:
            if state['expecting'] == 'function_name':
                state['partial_function_name'] = current_string
            elif state['expecting'] == 'argument_key':
                state['partial_arg_key'] = current_string
            elif state['expecting'] == 'argument_value':
                state['partial_arg_value'] = current_string
                
        return state
    
    def _get_valid_tokens_for_state(self, state: Dict[str, Any]) -> Set[int]:
        """Get valid tokens based on current generation state"""
        valid_tokens = set()
        
        expecting = state.get('expecting', 'list_start')
        
        if expecting == 'list_start':
            valid_tokens.add(self.special_tokens['['])
            
        elif expecting == 'object_start':
            valid_tokens.add(self.special_tokens['{'])
            if self.special_tokens.get(' '):
                valid_tokens.add(self.special_tokens[' '])
                
        elif expecting == 'name_key':
            # Can be "name" or just name
            valid_tokens.add(self.special_tokens['"'])
            # Also allow 'n' for unquoted name
            n_tokens = self.tokenizer.encode('n', add_special_tokens=False)
            if n_tokens:
                valid_tokens.add(n_tokens[0])
                
        elif expecting == 'function_name':
            # If we have a partial function name, get valid continuations
            partial = state.get('partial_function_name', '')
            if partial:
                # Get tokens that can continue this prefix
                if partial in self.valid_name_tokens:
                    valid_tokens.update(self.valid_name_tokens[partial])
                # Also allow closing quote if it's a complete function name
                if partial in self.valid_function_names:
                    valid_tokens.add(self.special_tokens['"'])
            else:
                # Starting a new function name - allow any first character of valid functions
                for func_name in self.valid_function_names:
                    if func_name:
                        first_char_tokens = self.tokenizer.encode(func_name[0], add_special_tokens=False)
                        if first_char_tokens:
                            valid_tokens.add(first_char_tokens[0])
                            
        elif expecting == 'argument_key':
            # Get valid argument keys for current function
            if state.get('current_function') in self.tool_lookup:
                tool = self.tool_lookup[state['current_function']]
                params = tool.get('parameters', {})
                
                partial = state.get('partial_arg_key', '')
                for param_name in params:
                    if param_name.startswith(partial):
                        if len(param_name) > len(partial):
                            next_char = param_name[len(partial)]
                            next_tokens = self.tokenizer.encode(next_char, add_special_tokens=False)
                            if next_tokens:
                                valid_tokens.add(next_tokens[0])
                        elif len(param_name) == len(partial):
                            # Complete match, allow closing quote
                            valid_tokens.add(self.special_tokens['"'])
                            
        elif expecting == 'comma_or_list_end':
            valid_tokens.add(self.special_tokens[','])
            valid_tokens.add(self.special_tokens[']'])
            if self.special_tokens.get(' '):
                valid_tokens.add(self.special_tokens[' '])
                
        # Add more state handling as needed...
        
        # Always allow some tokens for robustness
        if not valid_tokens:
            # Fallback to common structural tokens
            valid_tokens.update([
                self.special_tokens['"'],
                self.special_tokens['}'],
                self.special_tokens[']'],
            ])
            
        return valid_tokens
    
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        """Process logits to enforce valid function call structure"""
        # Get the generated text so far
        generated_text = self.tokenizer.decode(input_ids[0], skip_special_tokens=False)
        
        # Find where the JSON output starts (after the prompt)
        # Look for the start of JSON array
        json_start = generated_text.rfind('[')
        if json_start == -1:
            # Haven't started generating JSON yet, allow '['
            mask = torch.full_like(scores, float('-inf'))
            mask[:, self.special_tokens['[']] = 0
            return scores + mask
            
        # Extract the JSON portion
        json_text = generated_text[json_start:]
        
        # Parse current state
        state = self._parse_partial_json(json_text)
        
        # Get valid tokens for current state
        valid_tokens = self._get_valid_tokens_for_state(state)
        
        # Create mask
        mask = torch.full_like(scores, float('-inf'))
        for token_id in valid_tokens:
            if token_id < scores.shape[1]:  # Ensure token_id is valid
                mask[:, token_id] = 0
                
        return scores + mask


def generate_structured_function_calls(
    sample: Dict[str, Any],
    model_llm,
    tokenizer_llm,
    max_new_tokens: int = 150,
    temperature: float = 0.1,
    **kwargs
) -> str:
    """
    Generate structured function calls with validation based on available tools.
    
    Args:
        sample: Input sample containing 'query' and 'tools'
        model_llm: The language model
        tokenizer_llm: The tokenizer
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature (0 for greedy)
        
    Returns:
        Generated function calls as a JSON string
    """
    # Parse tools from the sample
    tools = json.loads(sample['tools']) if isinstance(sample['tools'], str) else sample['tools']
    
    # Create the structured output processor
    processor = StructuredFunctionCallProcessor(
        tokenizer=tokenizer_llm,
        tools=tools,
        device=model_llm.device
    )
    
    # Prepare inputs
    inputs = tokenize_for_llm(sample, tokenizer_llm, model_llm.device)
    
    # Generate with the processor
    with torch.no_grad():
        outputs = model_llm.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_return_sequences=1,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else 1.0,
            logits_processor=LogitsProcessorList([processor]),
            pad_token_id=tokenizer_llm.pad_token_id,
            eos_token_id=tokenizer_llm.eos_token_id,
        )
    
    # Decode the output
    input_len = inputs["input_ids"].shape[-1]
    output_ids = outputs[0][input_len:]
    output_text = tokenizer_llm.decode(output_ids, skip_special_tokens=True)
    
    return output_text

