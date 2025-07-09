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


# class StructuredFunctionCallProcessor(LogitsProcessor):
#     """
#     A LogitsProcessor that enforces structured function call generation based on available tools.
#     Validates function names, argument names, required arguments, and argument types during generation.
#     """
    
#     def __init__(self, tokenizer, tools: List[Dict[str, Any]], device: str = "cuda"):
#         self.tokenizer = tokenizer
#         self.tools = tools
#         self.device = device
        
#         # Build tool lookup for validation
#         self.tool_lookup = {tool["name"]: tool for tool in tools}
#         self.valid_function_names = set(self.tool_lookup.keys())
        
#         # Cache token IDs for common tokens
#         self.special_tokens = {
#             '[': tokenizer.encode('[', add_special_tokens=False)[0],
#             ']': tokenizer.encode(']', add_special_tokens=False)[0],
#             '{': tokenizer.encode('{', add_special_tokens=False)[0],
#             '}': tokenizer.encode('}', add_special_tokens=False)[0],
#             ',': tokenizer.encode(',', add_special_tokens=False)[0],
#             ':': tokenizer.encode(':', add_special_tokens=False)[0],
#             '"': tokenizer.encode('"', add_special_tokens=False)[0],
#             ' ': tokenizer.encode(' ', add_special_tokens=False)[0] if tokenizer.encode(' ', add_special_tokens=False) else None,
#         }
        
#         # Pre-compute valid tokens for function names
#         self.valid_name_tokens = self._compute_valid_name_tokens()
        
#     def _compute_valid_name_tokens(self) -> Dict[str, Set[int]]:
#         """Pre-compute valid tokens for each position in function names"""
#         valid_tokens = {}
        
#         for func_name in self.valid_function_names:
#             # Get all possible prefixes of the function name
#             for i in range(len(func_name)):
#                 prefix = func_name[:i+1]
#                 if prefix not in valid_tokens:
#                     valid_tokens[prefix] = set()
                
#                 # Find all function names that start with this prefix
#                 for name in self.valid_function_names:
#                     if name.startswith(prefix) and len(name) > i:
#                         next_char = name[i]
#                         # Get token for the next character
#                         token_ids = self.tokenizer.encode(next_char, add_special_tokens=False)
#                         if token_ids:
#                             valid_tokens[prefix].add(token_ids[0])
        
#         return valid_tokens
    
#     def _parse_partial_json(self, text: str) -> Dict[str, Any]:
#         """Parse partial JSON to understand current generation state"""
#         state = {
#             'in_list': False,
#             'in_object': False,
#             'in_string': False,
#             'current_key': None,
#             'current_function': None,
#             'current_args': {},
#             'functions': [],
#             'depth': 0,
#             'expecting': 'list_start'  # What we expect next
#         }
        
#         i = 0
#         current_string = ""
#         escape_next = False
        
#         while i < len(text):
#             char = text[i]
            
#             if escape_next:
#                 escape_next = False
#                 current_string += char
#                 i += 1
#                 continue
                
#             if char == '\\' and state['in_string']:
#                 escape_next = True
#                 current_string += char
#                 i += 1
#                 continue
            
#             if char == '"':
#                 if state['in_string']:
#                     state['in_string'] = False
#                     # Process the completed string
#                     if state['expecting'] == 'function_name':
#                         state['current_function'] = current_string
#                         state['expecting'] = 'name_colon'
#                     elif state['expecting'] == 'argument_key':
#                         state['current_key'] = current_string
#                         state['expecting'] = 'arg_colon'
#                     elif state['expecting'] == 'argument_value':
#                         state['current_args'][state['current_key']] = current_string
#                         state['expecting'] = 'arg_comma_or_close'
#                     current_string = ""
#                 else:
#                     state['in_string'] = True
                    
#             elif not state['in_string']:
#                 if char == '[':
#                     if state['context'] == 'start':
#                         state['stack'].append('array')
#                         state['context'] = 'array_start'
#                     elif state['context'] in ['expecting_args_value', 'expecting_arg_value']:
#                         # Array as argument value
#                         state['stack'].append('array')
#                         state['context'] = 'in_arg_array'
#                         if state['current_key']:
#                             state['current_args'][state['current_key']] = []  # Initialize as array
#                     elif state['context'] == 'in_arg_array':
#                         # Nested array inside argument array
#                         state['stack'].append('array')
#                         # Stay in in_arg_array context
#                     else:
#                         # Nested array in any other context is invalid for function calls
#                         state['is_valid'] = False
                    
#                 elif char == '{':
#                     state['in_object'] = True
#                     state['depth'] += 1
#                     if state['expecting'] == 'object_start':
#                         state['expecting'] = 'name_key'
#                 elif char == '}':
#                     state['depth'] -= 1
#                     if state['depth'] == 0:
#                         # Completed a function object
#                         if state['current_function']:
#                             state['functions'].append({
#                                 'name': state['current_function'],
#                                 'arguments': state['current_args'].copy()
#                             })
#                         state['current_function'] = None
#                         state['current_args'] = {}
#                         state['expecting'] = 'comma_or_list_end'
#                 elif char == ':':
#                     if state['expecting'] == 'name_colon':
#                         state['expecting'] = 'name_value'
#                     elif state['expecting'] == 'arg_colon':
#                         state['expecting'] = 'argument_value'
#                 elif char == ',':
#                     if state['expecting'] == 'arg_comma_or_close':
#                         state['expecting'] = 'argument_key'
#                     elif state['expecting'] == 'comma_or_list_end':
#                         state['expecting'] = 'object_start'
#                 elif char in ' \t\n\r':
#                     # Whitespace is generally allowed
#                     pass
                    
#                 else:
#                     current_string += char
                
#             else:
#                 current_string += char
                
#             i += 1
        
#         # If we're in the middle of a string, include it in the state
#         if state['in_string'] and current_string:
#             if state['expecting'] == 'function_name':
#                 state['partial_function_name'] = current_string
#             elif state['expecting'] == 'argument_key':
#                 state['partial_arg_key'] = current_string
#             elif state['expecting'] == 'argument_value':
#                 state['partial_arg_value'] = current_string
                
#         return state
    
#     def _get_valid_tokens_for_state(self, state: Dict[str, Any]) -> Set[int]:
#         """Get valid tokens based on current generation state"""
#         valid_tokens = set()
        
#         expecting = state.get('expecting', 'list_start')
        
#         if expecting == 'list_start':
#             valid_tokens.add(self.special_tokens['['])
            
#         elif expecting == 'object_start':
#             valid_tokens.add(self.special_tokens['{'])
#             if self.special_tokens.get(' '):
#                 valid_tokens.add(self.special_tokens[' '])
                
#         elif expecting == 'name_key':
#             # Can be "name" or just name
#             valid_tokens.add(self.special_tokens['"'])
#             # Also allow 'n' for unquoted name
#             n_tokens = self.tokenizer.encode('n', add_special_tokens=False)
#             if n_tokens:
#                 valid_tokens.add(n_tokens[0])
                
#         elif expecting == 'function_name':
#             # If we have a partial function name, get valid continuations
#             partial = state.get('partial_function_name', '')
#             if partial:
#                 # Get tokens that can continue this prefix
#                 if partial in self.valid_name_tokens:
#                     valid_tokens.update(self.valid_name_tokens[partial])
#                 # Also allow closing quote if it's a complete function name
#                 if partial in self.valid_function_names:
#                     valid_tokens.add(self.special_tokens['"'])
#             else:
#                 # Starting a new function name - allow any first character of valid functions
#                 for func_name in self.valid_function_names:
#                     if func_name:
#                         first_char_tokens = self.tokenizer.encode(func_name[0], add_special_tokens=False)
#                         if first_char_tokens:
#                             valid_tokens.add(first_char_tokens[0])
                            
#         elif expecting == 'argument_key':
#             # Get valid argument keys for current function
#             if state.get('current_function') in self.tool_lookup:
#                 tool = self.tool_lookup[state['current_function']]
#                 params = tool.get('parameters', {})
                
#                 partial = state.get('partial_arg_key', '')
#                 for param_name in params:
#                     if param_name.startswith(partial):
#                         if len(param_name) > len(partial):
#                             next_char = param_name[len(partial)]
#                             next_tokens = self.tokenizer.encode(next_char, add_special_tokens=False)
#                             if next_tokens:
#                                 valid_tokens.add(next_tokens[0])
#                         elif len(param_name) == len(partial):
#                             # Complete match, allow closing quote
#                             valid_tokens.add(self.special_tokens['"'])
                            
#         elif expecting == 'comma_or_list_end':
#             valid_tokens.add(self.special_tokens[','])
#             valid_tokens.add(self.special_tokens[']'])
#             if self.special_tokens.get(' '):
#                 valid_tokens.add(self.special_tokens[' '])
                
#         # Add more state handling as needed...
        
#         # Always allow some tokens for robustness
#         if not valid_tokens:
#             # Fallback to common structural tokens
#             valid_tokens.update([
#                 self.special_tokens['"'],
#                 self.special_tokens['}'],
#                 self.special_tokens[']'],
#             ])
            
#         return valid_tokens
    
#     def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
#         """Process logits to enforce valid function call structure"""
#         # Get the generated text so far
#         generated_text = self.tokenizer.decode(input_ids[0], skip_special_tokens=False)
        
#         # Find where the JSON output starts (after the prompt)
#         # Look for the start of JSON array
#         json_start = generated_text.rfind('[')
#         if json_start == -1:
#             # Haven't started generating JSON yet, allow '['
#             mask = torch.full_like(scores, float('-inf'))
#             mask[:, self.special_tokens['[']] = 0
#             return scores + mask
            
#         # Extract the JSON portion
#         json_text = generated_text[json_start:]
        
#         # Parse current state
#         state = self._parse_partial_json(json_text)
        
#         # Get valid tokens for current state
#         valid_tokens = self._get_valid_tokens_for_state(state)
        
#         # Create mask
#         mask = torch.full_like(scores, float('-inf'))
#         for token_id in valid_tokens:
#             if token_id < scores.shape[1]:  # Ensure token_id is valid
#                 mask[:, token_id] = 0
                
#         return scores + mask


# def generate_structured_function_calls(
#     sample: Dict[str, Any],
#     model_llm,
#     tokenizer_llm,
#     max_new_tokens: int = 150,
#     temperature: float = 0.1,
#     **kwargs
# ) -> str:
#     """
#     Generate structured function calls with validation based on available tools.
    
#     Args:
#         sample: Input sample containing 'query' and 'tools'
#         model_llm: The language model
#         tokenizer_llm: The tokenizer
#         max_new_tokens: Maximum tokens to generate
#         temperature: Sampling temperature (0 for greedy)
        
#     Returns:
#         Generated function calls as a JSON string
#     """
#     # Parse tools from the sample
#     tools = json.loads(sample['tools']) if isinstance(sample['tools'], str) else sample['tools']
    
#     # Create the structured output processor
#     processor = StructuredFunctionCallProcessor(
#         tokenizer=tokenizer_llm,
#         tools=tools,
#         device=model_llm.device
#     )
    
#     # Prepare inputs
#     inputs = tokenize_for_llm(sample, tokenizer_llm, model_llm.device)
    
#     # Generate with the processor
#     with torch.no_grad():
#         outputs = model_llm.generate(
#             **inputs,
#             max_new_tokens=max_new_tokens,
#             num_return_sequences=1,
#             do_sample=temperature > 0,
#             temperature=temperature if temperature > 0 else 1.0,
#             logits_processor=LogitsProcessorList([processor]),
#             pad_token_id=tokenizer_llm.pad_token_id,
#             eos_token_id=tokenizer_llm.eos_token_id,
#         )
    
#     # Decode the output
#     input_len = inputs["input_ids"].shape[-1]
#     output_ids = outputs[0][input_len:]
#     output_text = tokenizer_llm.decode(output_ids, skip_special_tokens=True)
    
#     return output_text


# def is_valid_output_prefix(tools: List[Dict[str, Any]], output_prefix: str) -> bool:
#     """
#     Check if output_prefix is a valid prefix of a function call output.
    
#     Args:
#         tools: List of available tools with their specifications
#         output_prefix: The partial output to validate
        
#     Returns:
#         bool: True if output_prefix could lead to a valid output, False otherwise
#     """
#     # Build tool lookup
#     tool_lookup = {tool["name"]: tool for tool in tools}
#     valid_function_names = set(tool_lookup.keys())
    
#     # Remove any leading/trailing whitespace
#     prefix = output_prefix.strip()
    
#     # Empty prefix is valid (can lead to valid output)
#     if not prefix:
#         return True
    
#     # Must start with '[' or be building towards it
#     if not prefix.startswith('['):
#         return prefix in ['', '[']
    
#     try:
#         # Try to parse what we have so far
#         state = _parse_prefix_state(prefix, tool_lookup, valid_function_names)
#         return state['is_valid']
#     except:
#         # If parsing fails, check if it could be a valid partial JSON
#         return _is_valid_partial_json(prefix)


# def generate_with_prefix_validation(
#     model,
#     tokenizer,
#     tools: List[Dict[str, Any]],
#     input_ids: torch.Tensor,
#     max_new_tokens: int = 150,
#     temperature: float = 0.1,
#     top_k: int = 50,
#     top_p: float = 0.95,
#     **kwargs
# ) -> str:
#     """
#     Generate LLM output by only adding tokens that maintain valid prefixes.
    
#     Args:
#         model: The language model
#         tokenizer: The tokenizer
#         tools: List of available tools with their specifications
#         input_ids: Input token IDs
#         max_new_tokens: Maximum number of tokens to generate
#         temperature: Sampling temperature (0 for greedy)
#         top_k: Top-k sampling parameter
#         top_p: Top-p (nucleus) sampling parameter
#         **kwargs: Additional generation parameters
        
#     Returns:
#         str: Generated output that maintains valid prefixes at every step
#     """
#     # Get the initial prompt length
#     input_length = input_ids.shape[-1]
    
#     # Start with the input
#     generated_ids = input_ids.clone()
    
#     # Track the generated text
#     generated_text = ""
    
#     # Generate token by token
#     for _ in range(max_new_tokens):
#         # Get model outputs
#         with torch.no_grad():
#             outputs = model(generated_ids)
#             logits = outputs.logits[:, -1, :]
            
#         # Apply temperature
#         if temperature > 0:
#             logits = logits / temperature
            
#         # Get probabilities
#         probs = torch.softmax(logits, dim=-1)
        
#         # Apply top-k filtering
#         if top_k > 0:
#             top_k_probs, top_k_indices = torch.topk(probs, min(top_k, probs.shape[-1]))
#             probs = torch.zeros_like(probs).scatter_(1, top_k_indices, top_k_probs)
            
#         # Apply top-p (nucleus) filtering
#         if top_p < 1.0:
#             sorted_probs, sorted_indices = torch.sort(probs, descending=True)
#             cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            
#             # Remove tokens with cumulative probability above the threshold
#             sorted_indices_to_remove = cumulative_probs > top_p
#             # Shift the indices to the right to keep also the first token above the threshold
#             sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
#             sorted_indices_to_remove[..., 0] = 0
            
#             # Set probabilities to 0
#             indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
#             probs[indices_to_remove] = 0
            
#         # Normalize probabilities
#         probs = probs / probs.sum(dim=-1, keepdim=True)
        
#         # Get candidate tokens sorted by probability
#         sorted_probs, sorted_indices = torch.sort(probs[0], descending=True)
        
#         # Find the first token that maintains a valid prefix
#         valid_token_found = False
#         for idx in range(len(sorted_indices)):
#             if sorted_probs[idx].item() == 0:
#                 continue
                
#             candidate_token = sorted_indices[idx].unsqueeze(0).unsqueeze(0)
            
#             # Try adding this token
#             candidate_ids = torch.cat([generated_ids, candidate_token], dim=-1)
            
#             # Decode only the generated part
#             generated_part = tokenizer.decode(
#                 candidate_ids[0][input_length:], 
#                 skip_special_tokens=True
#             )
            
#             # Check if this creates a valid prefix
#             if is_valid_output_prefix(tools, generated_part):
#                 # This token maintains a valid prefix
#                 generated_ids = candidate_ids
#                 generated_text = generated_part
#                 valid_token_found = True
#                 break
                
#         if not valid_token_found:
#             # No valid token found, stop generation
#             break
            
#         # Check for EOS token
#         if candidate_token.item() == tokenizer.eos_token_id:
#             break
            
#         # Check if we have a complete valid output (ends with ']')
#         if generated_text.strip().endswith(']'):
#             # Verify it's actually complete and valid
#             try:
#                 json.loads(generated_text.strip())
#                 break  # Valid complete JSON, stop generation
#             except:
#                 pass  # Not complete yet, continue
                
#     return generated_text


# def generate_with_prefix_validation_batch(
#     model,
#     tokenizer,
#     tools: List[Dict[str, Any]],
#     input_ids: torch.Tensor,
#     max_new_tokens: int = 150,
#     num_return_sequences: int = 1,
#     temperature: float = 0.1,
#     **kwargs
# ) -> List[str]:
#     """
#     Generate multiple sequences with prefix validation.
    
#     Args:
#         model: The language model
#         tokenizer: The tokenizer
#         tools: List of available tools
#         input_ids: Input token IDs
#         max_new_tokens: Maximum number of tokens to generate
#         num_return_sequences: Number of sequences to generate
#         temperature: Sampling temperature
#         **kwargs: Additional generation parameters
        
#     Returns:
#         List[str]: List of generated outputs
#     """
#     results = []
    
#     for _ in range(num_return_sequences):
#         output = generate_with_prefix_validation(
#             model=model,
#             tokenizer=tokenizer,
#             tools=tools,
#             input_ids=input_ids,
#             max_new_tokens=max_new_tokens,
#             temperature=temperature,
#             **kwargs
#         )
#         results.append(output)
        
#     return results


# def _parse_prefix_state(prefix: str, tool_lookup: Dict[str, Dict], valid_function_names: Set[str]) -> Dict[str, Any]:
#     """
#     Parse the prefix to understand its current state and validity.
#     """
#     state = {
#         'is_valid': True,
#         'in_string': False,
#         'escape_next': False,
#         'current_string': '',
#         'stack': [],  # Track JSON structure
#         'context': 'start',  # Current parsing context
#         'function_calls': [],
#         'current_function': None,
#         'current_args': {},
#         'current_key': None,
#         'completed_functions': []  # Track completed function calls
#     }
    
#     i = 0
#     while i < len(prefix) and state['is_valid']:
#         char = prefix[i]
        
#         if state['escape_next']:
#             state['escape_next'] = False
#             state['current_string'] += char
#             i += 1
#             continue
            
#         if char == '\\' and state['in_string']:
#             state['escape_next'] = True
#             state['current_string'] += char
#             i += 1
#             continue
        
#         if char == '"':
#             if state['in_string']:
#                 state['in_string'] = False
#                 # Process completed string based on context
#                 if not _process_completed_string(state, tool_lookup, valid_function_names):
#                     state['is_valid'] = False
#                 state['current_string'] = ''
#             else:
#                 state['in_string'] = True
        
#         elif state['in_string']:
#             state['current_string'] += char
            
#         else:  # Not in string
#             if char == '[':
#                 if state['context'] == 'start':
#                     state['stack'].append('array')
#                     state['context'] = 'array_start'
#                 elif state['context'] in ['expecting_args_value', 'expecting_arg_value']:
#                     # Array as argument value
#                     state['stack'].append('array')
#                     state['context'] = 'in_arg_array'
#                     if state['current_key']:
#                         state['current_args'][state['current_key']] = []  # Initialize as array
#                 elif state['context'] == 'in_arg_array':
#                     # Nested array inside argument array
#                     state['stack'].append('array')
#                     # Stay in in_arg_array context
#                 else:
#                     # Nested array in any other context is invalid for function calls
#                     state['is_valid'] = False
                    
#             elif char == ']':
#                 if not state['stack'] or state['stack'][-1] != 'array':
#                     state['is_valid'] = False
#                 else:
#                     state['stack'].pop()
#                     if len(state['stack']) == 0:
#                         state['context'] = 'complete'
#                         # If we're completing the main array, validate all completed functions
#                         for func_call in state['completed_functions']:
#                             if not _validate_completed_function_call(func_call, tool_lookup):
#                                 state['is_valid'] = False
#                                 break
#                     elif state['context'] == 'in_arg_array':
#                         # Completed an array argument value
#                         state['context'] = 'after_arg_value'
                        
#             elif char == '{':
#                 state['stack'].append('object')
#                 if state['context'] in ['array_start', 'after_comma']:
#                     state['context'] = 'object_start'
#                     state['current_function'] = None
#                     state['current_args'] = {}
#                 elif state['context'] == 'expecting_args_value':
#                     state['context'] = 'args_object_start'
#                 elif state['context'] == 'expecting_arg_value':
#                     # Object as argument value
#                     state['context'] = 'in_arg_object'
#                     if state['current_key']:
#                         state['current_args'][state['current_key']] = {}  # Initialize as object
                    
#             elif char == '}':
#                 if not state['stack'] or state['stack'][-1] != 'object':
#                     state['is_valid'] = False
#                 else:
#                     state['stack'].pop()
#                     if state['context'] in ['after_function_name', 'after_args', 'args_object_start', 'after_arg_value']:
#                         # Completing a function call object
#                         if state['current_function']:
#                             state['completed_functions'].append({
#                                 'name': state['current_function'],
#                                 'arguments': state['current_args'].copy()
#                             })
#                         state['context'] = 'after_object'
#                     elif state['context'] == 'in_arg_object':
#                         # Completed an object argument value
#                         state['context'] = 'after_arg_value'
                        
#             elif char == ':':
#                 if state['context'] == 'after_name_key':
#                     state['context'] = 'expecting_function_name'
#                 elif state['context'] == 'after_args_key':
#                     state['context'] = 'expecting_args_value'
#                 elif state['context'] == 'after_arg_key':
#                     state['context'] = 'expecting_arg_value'
#                 else:
#                     # Colon in unexpected place
#                     state['is_valid'] = False
                    
#             elif char == ',':
#                 if state['context'] == 'after_object':
#                     state['context'] = 'after_comma'
#                 elif state['context'] == 'after_arg_value':
#                     state['context'] = 'args_object_start'
#                 elif state['context'] == 'after_function_name':
#                     state['context'] = 'after_name_comma'
#                 elif state['context'] == 'in_arg_array':
#                     # Comma inside array - valid
#                     pass
                    
#             elif char in ' \t\n\r':
#                 # Whitespace is generally allowed
#                 pass
                
#             elif char == '.':
#                 # Decimal point in number
#                 if state['context'] in ['expecting_arg_value', 'in_arg_array'] and i > 0 and prefix[i-1].isdigit():
#                     # Part of a decimal number
#                     pass
#                 else:
#                     state['is_valid'] = False
                    
#             elif char == '-':
#                 # Negative number
#                 if state['context'] in ['expecting_arg_value', 'in_arg_array']:
#                     # Could be start of negative number
#                     pass
#                 else:
#                     state['is_valid'] = False
                
#             else:
#                 # Unexpected character outside string
#                 if char.isalpha() and state['context'] in ['object_start', 'after_name_comma', 'args_object_start']:
#                     # Might be an unquoted key - check if it's valid
#                     key_match = re.match(r'([a-zA-Z_]\w*)', prefix[i:])
#                     if key_match:
#                         key = key_match.group(1)
#                         i += len(key) - 1
#                         if not _process_unquoted_key(state, key):
#                             state['is_valid'] = False
#                 elif char.isdigit() and state['context'] in ['expecting_arg_value', 'in_arg_array']:
#                     # Could be a number value
#                     num_match = re.match(r'(-?\d+\.?\d*)', prefix[i:])
#                     if num_match:
#                         num_str = num_match.group(1)
#                         i += len(num_str) - 1
#                         if state['context'] == 'expecting_arg_value' and state['current_key']:
#                             try:
#                                 state['current_args'][state['current_key']] = float(num_str) if '.' in num_str else int(num_str)
#                             except:
#                                 pass
#                             state['context'] = 'after_arg_value'
#                 elif char in 'tf' and state['context'] in ['expecting_arg_value', 'in_arg_array']:
#                     # Could be boolean true/false
#                     if prefix[i:].startswith('true'):
#                         if state['context'] == 'expecting_arg_value' and state['current_key']:
#                             state['current_args'][state['current_key']] = True
#                             state['context'] = 'after_arg_value'
#                         i += 3  # Skip 'rue'
#                     elif prefix[i:].startswith('false'):
#                         if state['context'] == 'expecting_arg_value' and state['current_key']:
#                             state['current_args'][state['current_key']] = False
#                             state['context'] = 'after_arg_value'
#                         i += 4  # Skip 'alse'
#                     else:
#                         state['is_valid'] = False
#                 elif char == 'n' and state['context'] in ['expecting_arg_value', 'in_arg_array']:
#                     # Could be null
#                     if prefix[i:].startswith('null'):
#                         if state['context'] == 'expecting_arg_value' and state['current_key']:
#                             state['current_args'][state['current_key']] = None
#                             state['context'] = 'after_arg_value'
#                         i += 3  # Skip 'ull'
#                     else:
#                         state['is_valid'] = False
#                 else:
#                     state['is_valid'] = False
                
#         i += 1
    
#     # Check if we're in a valid partial state
#     if state['is_valid'] and state['in_string']:
#         # We're in the middle of a string - check if it could be valid
#         state['is_valid'] = _is_valid_partial_string(state, tool_lookup, valid_function_names)
        
#     return state


# def _process_completed_string(state: Dict, tool_lookup: Dict, valid_function_names: Set[str]) -> bool:
#     """Process a completed string based on the current context."""
#     string_value = state['current_string']
    
#     if state['context'] == 'expecting_function_name':
#         if string_value not in valid_function_names:
#             return False
#         state['current_function'] = string_value
#         state['context'] = 'after_function_name'
        
#     elif state['context'] == 'object_start':
#         if string_value == 'name':
#             state['context'] = 'after_name_key'
#         elif string_value == 'arguments':
#             state['context'] = 'after_args_key'
#         else:
#             return False
            
#     elif state['context'] == 'after_name_comma':
#         if string_value == 'arguments':
#             state['context'] = 'after_args_key'
#         else:
#             return False
            
#     elif state['context'] == 'args_object_start':
#         # This should be an argument key
#         if state['current_function'] and state['current_function'] in tool_lookup:
#             tool = tool_lookup[state['current_function']]
#             params = tool.get('parameters', {})
#             if string_value not in params:
#                 return False
#             state['current_key'] = string_value
#             state['context'] = 'after_arg_key'
#         else:
#             return False
            
#     elif state['context'] == 'expecting_arg_value':
#         # Store the argument value
#         if state['current_key']:
#             state['current_args'][state['current_key']] = string_value
#             state['context'] = 'after_arg_value'
            
#     return True


# def _process_unquoted_key(state: Dict, key: str) -> bool:
#     """Process an unquoted key."""
#     if state['context'] == 'object_start':
#         if key == 'name':
#             state['context'] = 'after_name_key'
#             return True
#         elif key == 'arguments':
#             state['context'] = 'after_args_key'
#             return True
            
#     elif state['context'] == 'after_name_comma':
#         if key == 'arguments':
#             state['context'] = 'after_args_key'
#             return True
            
#     return False


# def _is_valid_partial_string(state: Dict, tool_lookup: Dict, valid_function_names: Set[str]) -> bool:
#     """Check if a partial string could lead to a valid completion."""
#     partial = state['current_string']
    
#     if state['context'] == 'expecting_function_name':
#         # Check if any function name starts with this prefix
#         return any(name.startswith(partial) for name in valid_function_names)
        
#     elif state['context'] == 'object_start':
#         # Check if 'name' or 'arguments' starts with this
#         return 'name'.startswith(partial) or 'arguments'.startswith(partial)
        
#     elif state['context'] == 'after_name_comma':
#         return 'arguments'.startswith(partial)
        
#     elif state['context'] == 'args_object_start':
#         # Check if any parameter name starts with this prefix
#         if state['current_function'] in tool_lookup:
#             tool = tool_lookup[state['current_function']]
#             params = tool.get('parameters', {})
#             return any(param.startswith(partial) for param in params)
            
#     # For other contexts, assume it could be valid
#     return True


# def _validate_completed_function_call(func_call: Dict, tool_lookup: Dict) -> bool:
#     """Validate a fully completed function call including required parameters."""
#     func_name = func_call.get('name')
#     if not func_name or func_name not in tool_lookup:
#         return False
        
#     tool = tool_lookup[func_name]
#     params = tool.get('parameters', {})
#     provided_args = func_call.get('arguments', {})
    
#     # Check if all required parameters are present
#     for param_name, param_spec in params.items():
#         if param_spec.get('required', False) and param_name not in provided_args:
#             return False
            
#     # Check if all provided arguments are valid
#     for arg_name, arg_value in provided_args.items():
#         if arg_name not in params:
#             return False
            
#         # Basic type checking
#         param_spec = params[arg_name]
#         expected_type = param_spec.get('type', 'str')
        
#         if expected_type == 'str' and not isinstance(arg_value, str):
#             return False
#         elif expected_type == 'int' and not isinstance(arg_value, (int, float)):
#             return False
#         elif expected_type == 'bool' and not isinstance(arg_value, bool):
#             return False
            
#     return True


# def _validate_function_call(state: Dict, tool_lookup: Dict) -> bool:
#     """Validate a function call that may still be under construction."""
#     if not state['current_function']:
#         return False
        
#     if state['current_function'] not in tool_lookup:
#         return False
        
#     tool = tool_lookup[state['current_function']]
#     params = tool.get('parameters', {})
    
#     # For partial validation, we only check if provided arguments are valid
#     # We don't check for required arguments unless the function call is complete
#     for arg_name in state['current_args']:
#         if arg_name not in params:
#             return False
            
#     return True


# def _is_valid_partial_json(prefix: str) -> bool:
#     """
#     Fallback check for valid partial JSON structure.
#     This is a more lenient check for cases where detailed parsing fails.
#     """
#     # Count brackets and quotes
#     open_brackets = prefix.count('[') - prefix.count(']')
#     open_braces = prefix.count('{') - prefix.count('}')
    
#     # Both should be non-negative
#     if open_brackets < 0 or open_braces < 0:
#         return False
        
#     # Count quotes (excluding escaped ones)
#     quote_count = 0
#     i = 0
#     while i < len(prefix):
#         if prefix[i] == '"' and (i == 0 or prefix[i-1] != '\\'):
#             quote_count += 1
#         i += 1
        
#     # Quotes should be balanced (even number)
#     if quote_count % 2 != 0:
#         # Odd number of quotes is okay - we might be in the middle of a string
#         return True
        
#     return True

