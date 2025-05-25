class JsonSchemaLogitsProcessor(LogitsProcessor):
    # --- State Definitions ---
    STATE_EXPECTING_LIST_START = "EXPECTING_LIST_START" # Expecting '['
    STATE_IN_LIST_EXPECTING_OBJECT_START_OR_LIST_END = "IN_LIST_EXPECTING_OBJECT_START_OR_LIST_END" # Expecting '{' or ']' or ','

    STATE_IN_OBJECT_EXPECTING_KEY_QUOTED = "IN_OBJECT_EXPECTING_KEY_QUOTED" # Expecting "key_name"
    STATE_EXPECTING_OBJECT_KEY_STRING_START = "EXPECTING_OBJECT_KEY_STRING_START"
    STATE_EXPECTING_OBJECT_KEY_STRING_VALUE = "EXPECTING_OBJECT_KEY_STRING_VALUE" # e.g. "name" or "arguments"
    STATE_EXPECTING_OBJECT_KEY_STRING_END = "EXPECTING_OBJECT_KEY_STRING_END"
    
    STATE_IN_OBJECT_AFTER_KEY_EXPECTING_COLON = "IN_OBJECT_AFTER_KEY_EXPECTING_COLON"
    
    STATE_EXPECTING_VALUE = "EXPECTING_VALUE" # General state before knowing type
    STATE_EXPECTING_TOOL_NAME_STRING_START = "EXPECTING_TOOL_NAME_STRING_START" # Expecting " for tool name
    STATE_EXPECTING_TOOL_NAME_STRING_VALUE = "EXPECTING_TOOL_NAME_STRING_VALUE" # Expecting tool name characters
    STATE_EXPECTING_TOOL_NAME_STRING_END = "EXPECTING_TOOL_NAME_STRING_END"   # Expecting " after tool name

    STATE_EXPECTING_ARGUMENTS_OBJECT_START = "EXPECTING_ARGUMENTS_OBJECT_START" # Expecting { for arguments
    
    STATE_IN_ARGUMENTS_OBJECT_EXPECTING_KEY_OR_END = "IN_ARGUMENTS_OBJECT_EXPECTING_KEY_OR_END" # Expecting "arg_key" or }
    STATE_EXPECTING_ARGUMENT_KEY_STRING_START = "EXPECTING_ARGUMENT_KEY_STRING_START"
    STATE_EXPECTING_ARGUMENT_KEY_STRING_VALUE = "EXPECTING_ARGUMENT_KEY_STRING_VALUE"
    STATE_EXPECTING_ARGUMENT_KEY_STRING_END = "EXPECTING_ARGUMENT_KEY_STRING_END"

    STATE_AFTER_ARGUMENT_KEY_EXPECTING_COLON = "AFTER_ARGUMENT_KEY_EXPECTING_COLON"

    STATE_EXPECTING_ARGUMENT_VALUE = "EXPECTING_ARGUMENT_VALUE" # General state, branches by type
    STATE_EXPECTING_STRING_ARGUMENT_START = "EXPECTING_STRING_ARGUMENT_START"
    STATE_EXPECTING_STRING_ARGUMENT_VALUE = "EXPECTING_STRING_ARGUMENT_VALUE"
    STATE_EXPECTING_STRING_ARGUMENT_END = "EXPECTING_STRING_ARGUMENT_END"
    # Add states for other types like NUMBER, BOOLEAN as needed (e.g. STATE_EXPECTING_NUMBER_VALUE)

    STATE_IN_OBJECT_AFTER_VALUE_EXPECTING_COMMA_OR_END = "IN_OBJECT_AFTER_VALUE_EXPECTING_COMMA_OR_END" # Expecting ',' or '}'
    STATE_IN_ARGUMENTS_OBJECT_AFTER_VALUE_EXPECTING_COMMA_OR_END = "IN_ARGUMENTS_OBJECT_AFTER_VALUE_EXPECTING_COMMA_OR_END"

    STATE_COMPLETE = "STATE_COMPLETE" # Generation is considered complete by schema

    def __init__(self, tools_json_string, tokenizer, llm_device):
        self.tokenizer = tokenizer
        self.llm_device = llm_device
        self.schema = self._parse_tools(tools_json_string)
        self.token_id_cache = {} 
        self.current_tool_definition = None 
        self.current_arg_schema = None

        self._pre_tokenize_schema_elements()

        # State stack: each element is a dict {"type": STATE_STRING, "context": {...}, "processed_input_len": N}
        # "processed_input_len" tracks how much of the input_ids has been consumed by this state level
        self.state_stack = [{"type": self.STATE_EXPECTING_LIST_START, "processed_input_len": 0, "context": {}}]


    def _parse_tools(self, tools_json_string):
        tools_list = json.loads(tools_json_string)
        for tool in tools_list:
            if "parameters" not in tool or tool["parameters"] is None:
                tool["parameters"] = {}
            elif not isinstance(tool["parameters"], dict):
                tool["parameters"] = {}
        parsed_schema = {
            "tool_names": [tool["name"] for tool in tools_list],
            "tool_definitions": {tool["name"]: tool for tool in tools_list}
        }
        return parsed_schema

    def _get_token_ids(self, text_string, force_single_token=False):
        if not text_string: return []
        cache_key = text_string
        if cache_key not in self.token_id_cache:
            token_ids = self.tokenizer.encode(text_string, add_special_tokens=False)
            self.token_id_cache[cache_key] = token_ids
        
        token_ids = self.token_id_cache[cache_key]
        if force_single_token and len(token_ids) != 1:
            # This can be a warning or error if a critical punctuation/keyword isn't a single token
            pass 
        return token_ids

    def _pre_tokenize_schema_elements(self):
        common_tokens = ['{', '}', '[', ']', ',', ':', '"', 'name', 'arguments', 'true', 'false', 'null']
        for token_str in common_tokens:
            self._get_token_ids(token_str) 

        for name in self.schema["tool_names"]:
            self._get_token_ids(name) 

        for tool_def in self.schema["tool_definitions"].values():
            if tool_def["parameters"]:
                for param_name in tool_def["parameters"].keys():
                    self._get_token_ids(param_name)

    def _get_allowed_punctuation_tokens(self, punctuations):
        allowed_ids = set()
        for punc in punctuations:
            ids = self._get_token_ids(punc, force_single_token=True) 
            if ids: 
                allowed_ids.add(ids[0]) 
        return list(allowed_ids)

    def _get_allowed_keyword_tokens(self, keywords, current_input_ids, state_context):
        # This now handles multi-token keywords by checking partial matches
        allowed_next_token_ids = set()
        start_idx = state_context.get("start_of_keyword_token_idx", -1)
        current_partial_keyword_tokens = []
        if start_idx != -1 and start_idx < len(current_input_ids):
            current_partial_keyword_tokens = current_input_ids[start_idx:].tolist()

        for keyword in keywords:
            keyword_token_ids = self._get_token_ids(keyword)
            if not keyword_token_ids: continue

            is_prefix = True
            if len(current_partial_keyword_tokens) > len(keyword_token_ids):
                is_prefix = False
            else:
                for i in range(len(current_partial_keyword_tokens)):
                    if current_partial_keyword_tokens[i] != keyword_token_ids[i]:
                        is_prefix = False
                        break
            
            if is_prefix:
                if len(current_partial_keyword_tokens) < len(keyword_token_ids):
                    allowed_next_token_ids.add(keyword_token_ids[len(current_partial_keyword_tokens)])
                # If fully matched, transition is handled by _synchronize_state_with_input
        return list(allowed_next_token_ids)


    def _get_allowed_specific_string_tokens(self, strings_list, current_input_ids, state_context):
        allowed_next_token_ids = set()
        start_idx = state_context.get("start_of_value_token_idx", -1)
        current_partial_string_tokens = []

        if start_idx != -1 and start_idx < len(current_input_ids): # Ensure start_idx is valid
            current_partial_string_tokens = current_input_ids[start_idx:].tolist()
        
        for s in strings_list:
            s_token_ids = self._get_token_ids(s)
            if not s_token_ids: continue

            is_prefix = True
            if len(current_partial_string_tokens) > len(s_token_ids):
                is_prefix = False
            else:
                for i in range(len(current_partial_string_tokens)):
                    if current_partial_string_tokens[i] != s_token_ids[i]:
                        is_prefix = False
                        break
            
            if is_prefix:
                if len(current_partial_string_tokens) < len(s_token_ids):
                    allowed_next_token_ids.add(s_token_ids[len(current_partial_string_tokens)])
        return list(allowed_next_token_ids)

    def _synchronize_state_with_input(self, input_ids: torch.LongTensor):
        """
        Parses input_ids from the last processed point and updates the state_stack.
        This is the core FSM logic.
        """
        current_input_idx = self.state_stack[-1]["processed_input_len"]
        
        while current_input_idx < len(input_ids):
            if not self.state_stack: # Should not happen if initialized
                # print("Error: State stack empty during synchronization.")
                return

            state_info = self.state_stack[-1]
            state_type = state_info["type"]
            token_id = input_ids[current_input_idx].item()
            
            next_state_pushed = False

            # print(f"Sync: Idx: {current_input_idx}, Token: {self.tokenizer.decode([token_id])}, State: {state_type}")

            if state_type == self.STATE_EXPECTING_LIST_START:
                if token_id == self._get_token_ids('[', True)[0]:
                    self.state_stack.pop()
                    self.state_stack.append({"type": self.STATE_IN_LIST_EXPECTING_OBJECT_START_OR_LIST_END, "processed_input_len": current_input_idx + 1, "context": {}})
                    next_state_pushed = True
                # else: Error or unexpected token. For now, we assume LLM follows logits.
            
            elif state_type == self.STATE_IN_LIST_EXPECTING_OBJECT_START_OR_LIST_END:
                if token_id == self._get_token_ids('{', True)[0]:
                    self.state_stack.pop()
                    self.state_stack.append({"type": self.STATE_IN_OBJECT_EXPECTING_KEY_QUOTED, "processed_input_len": current_input_idx + 1, "context": {"expected_keys": ["name", "arguments"]}})
                    next_state_pushed = True
                elif token_id == self._get_token_ids(']', True)[0]:
                    self.state_stack.pop()
                    self.state_stack.append({"type": self.STATE_COMPLETE, "processed_input_len": current_input_idx + 1, "context": {}})
                    next_state_pushed = True
                elif token_id == self._get_token_ids(',', True)[0]: # After a completed object
                    state_info["processed_input_len"] = current_input_idx + 1 # Consume comma, stay in state for next object
                # else: Error or unexpected.

            elif state_type == self.STATE_IN_OBJECT_EXPECTING_KEY_QUOTED: # Expects "name" or "arguments"
                if token_id == self._get_token_ids('"', True)[0]:
                    self.state_stack.pop()
                    self.state_stack.append({"type": self.STATE_EXPECTING_OBJECT_KEY_STRING_VALUE, 
                                             "processed_input_len": current_input_idx + 1, 
                                             "context": {"start_of_value_token_idx": current_input_idx + 1, 
                                                         "valid_keys": state_info["context"]["expected_keys"]}})
                    next_state_pushed = True

            elif state_type == self.STATE_EXPECTING_OBJECT_KEY_STRING_VALUE:
                # Check if current input forms a valid key
                start_idx = state_info["context"]["start_of_value_token_idx"]
                # The actual key string tokens are from start_idx up to current_input_idx (inclusive of current token)
                current_key_tokens = input_ids[start_idx : current_input_idx + 1].tolist()
                current_key_text = self.tokenizer.decode(current_key_tokens)

                matched_key = None
                for k_text in state_info["context"]["valid_keys"]:
                    k_tokens = self._get_token_ids(k_text)
                    if current_key_tokens == k_tokens:
                        matched_key = k_text
                        break
                
                if matched_key:
                    self.state_stack.pop()
                    self.state_stack.append({"type": self.STATE_EXPECTING_OBJECT_KEY_STRING_END, 
                                             "processed_input_len": current_input_idx + 1, 
                                             "context": {"current_key": matched_key}})
                    next_state_pushed = True

            elif state_type == self.STATE_EXPECTING_OBJECT_KEY_STRING_END:
                if token_id == self._get_token_ids('"', True)[0]:
                    self.state_stack.pop()
                    self.state_stack.append({"type": self.STATE_IN_OBJECT_AFTER_KEY_EXPECTING_COLON, 
                                             "processed_input_len": current_input_idx + 1, 
                                             "context": state_info["context"]}) # Pass current_key
                    next_state_pushed = True

            elif state_type == self.STATE_IN_OBJECT_AFTER_KEY_EXPECTING_COLON:
                if token_id == self._get_token_ids(':', True)[0]:
                    self.state_stack.pop()
                    key = state_info["context"]["current_key"]
                    next_s_type = self.STATE_EXPECTING_VALUE
                    if key == "name":
                        next_s_type = self.STATE_EXPECTING_TOOL_NAME_STRING_START
                    elif key == "arguments":
                        next_s_type = self.STATE_EXPECTING_ARGUMENTS_OBJECT_START
                    
                    self.state_stack.append({"type": next_s_type, 
                                             "processed_input_len": current_input_idx + 1, 
                                             "context": state_info["context"]})
                    next_state_pushed = True

            elif state_type == self.STATE_EXPECTING_TOOL_NAME_STRING_START:
                if token_id == self._get_token_ids('"', True)[0]:
                    self.state_stack.pop()
                    self.state_stack.append({"type": self.STATE_EXPECTING_TOOL_NAME_STRING_VALUE, 
                                             "processed_input_len": current_input_idx + 1, 
                                             "context": {"start_of_value_token_idx": current_input_idx + 1}})
                    next_state_pushed = True

            elif state_type == self.STATE_EXPECTING_TOOL_NAME_STRING_VALUE:
                start_idx = state_info["context"]["start_of_value_token_idx"]
                current_value_tokens = input_ids[start_idx : current_input_idx + 1].tolist()
                current_value_text = self.tokenizer.decode(current_value_tokens)

                matched_tool_name = None
                for t_name in self.schema["tool_names"]:
                    if current_value_text == t_name: # Exact match
                        matched_tool_name = t_name
                        break
                
                if matched_tool_name:
                    self.current_tool_definition = self.schema["tool_definitions"][matched_tool_name]
                    self.state_stack.pop()
                    self.state_stack.append({"type": self.STATE_EXPECTING_TOOL_NAME_STRING_END, 
                                             "processed_input_len": current_input_idx + 1, 
                                             "context": {"tool_name": matched_tool_name}})
                    next_state_pushed = True
                else: # Partial match, stay
                    state_info["processed_input_len"] = current_input_idx + 1
            
            elif state_type == self.STATE_EXPECTING_TOOL_NAME_STRING_END:
                 if token_id == self._get_token_ids('"', True)[0]:
                    self.state_stack.pop()
                    # After "name": "value", expect comma or end of object
                    # The "object" here is the tool call { "name": ..., "arguments": ...}
                    # Update context of IN_OBJECT_EXPECTING_KEY_QUOTED
                    # Find parent IN_OBJECT_EXPECTING_KEY_QUOTED and update its context
                    for s in reversed(self.state_stack):
                        if s["type"] == self.STATE_IN_OBJECT_EXPECTING_KEY_QUOTED:
                            s["context"]["expected_keys"] = ["arguments"] # Next key must be arguments
                            s["processed_input_len"] = current_input_idx + 1
                            break 
                    # This pop and then modification of parent state is a bit indirect.
                    # Alternative: a dedicated STATE_IN_OBJECT_AFTER_NAME_VALUE
                    self.state_stack.append({"type": self.STATE_IN_OBJECT_AFTER_VALUE_EXPECTING_COMMA_OR_END, 
                                             "processed_input_len": current_input_idx + 1, 
                                             "context": {"after_key": "name", "current_tool_name": state_info["context"].get("tool_name")}})
                    next_state_pushed = True

            elif state_type == self.STATE_EXPECTING_ARGUMENTS_OBJECT_START:
                if token_id == self._get_token_ids('{', True)[0]:
                    self.state_stack.pop()
                    # Determine available argument keys for the current tool
                    arg_keys = []
                    if self.current_tool_definition and self.current_tool_definition["parameters"]:
                        arg_keys = list(self.current_tool_definition["parameters"].keys())

                    self.state_stack.append({"type": self.STATE_IN_ARGUMENTS_OBJECT_EXPECTING_KEY_OR_END, 
                                             "processed_input_len": current_input_idx + 1, 
                                             "context": {"available_arg_keys": arg_keys, "defined_args": set(), "tool_name": self.current_tool_definition["name"] if self.current_tool_definition else None}})
                    next_state_pushed = True
            
            elif state_type == self.STATE_IN_ARGUMENTS_OBJECT_EXPECTING_KEY_OR_END:
                if token_id == self._get_token_ids('"', True)[0]: # Start of an argument key
                    self.state_stack.pop()
                    self.state_stack.append({"type": self.STATE_EXPECTING_ARGUMENT_KEY_STRING_VALUE, 
                                             "processed_input_len": current_input_idx + 1, 
                                             "context": {"start_of_value_token_idx": current_input_idx + 1, 
                                                         "parent_context": state_info["context"]}}) # Pass available_arg_keys, defined_args
                    next_state_pushed = True
                elif token_id == self._get_token_ids('}', True)[0]: # End of arguments object
                    self.state_stack.pop()
                     # After "arguments": {...}, expect comma or end of main tool object
                    self.state_stack.append({"type": self.STATE_IN_OBJECT_AFTER_VALUE_EXPECTING_COMMA_OR_END, 
                                             "processed_input_len": current_input_idx + 1, 
                                             "context": {"after_key": "arguments", "current_tool_name": state_info["context"].get("tool_name")}})
                    next_state_pushed = True

            elif state_type == self.STATE_EXPECTING_ARGUMENT_KEY_STRING_VALUE:
                start_idx = state_info["context"]["start_of_value_token_idx"]
                current_key_tokens = input_ids[start_idx : current_input_idx + 1].tolist()
                current_key_text = self.tokenizer.decode(current_key_tokens)
                
                parent_ctx = state_info["context"]["parent_context"]
                available_keys = parent_ctx["available_arg_keys"]
                defined_args = parent_ctx["defined_args"]
                
                matched_arg_key = None
                for key_text in available_keys:
                    if key_text not in defined_args and current_key_text == key_text:
                        matched_arg_key = key_text
                        break
                
                if matched_arg_key:
                    self.current_arg_schema = self.current_tool_definition["parameters"][matched_arg_key]
                    self.state_stack.pop()
                    self.state_stack.append({"type": self.STATE_EXPECTING_ARGUMENT_KEY_STRING_END, 
                                             "processed_input_len": current_input_idx + 1, 
                                             "context": {"current_arg_key": matched_arg_key, 
                                                         "parent_context": parent_ctx}})
                    next_state_pushed = True
                else: # Partial match
                    state_info["processed_input_len"] = current_input_idx + 1
            
            elif state_type == self.STATE_EXPECTING_ARGUMENT_KEY_STRING_END:
                if token_id == self._get_token_ids('"', True)[0]:
                    self.state_stack.pop()
                    self.state_stack.append({"type": self.STATE_AFTER_ARGUMENT_KEY_EXPECTING_COLON, 
                                             "processed_input_len": current_input_idx + 1, 
                                             "context": state_info["context"]}) # Pass current_arg_key, parent_context
                    next_state_pushed = True

            elif state_type == self.STATE_AFTER_ARGUMENT_KEY_EXPECTING_COLON:
                if token_id == self._get_token_ids(':', True)[0]:
                    self.state_stack.pop()
                    # Determine value type based on self.current_arg_schema
                    # For now, assume string
                    self.state_stack.append({"type": self.STATE_EXPECTING_STRING_ARGUMENT_START, 
                                             "processed_input_len": current_input_idx + 1, 
                                             "context": state_info["context"]}) 
                    next_state_pushed = True
            
            elif state_type == self.STATE_EXPECTING_STRING_ARGUMENT_START:
                if token_id == self._get_token_ids('"', True)[0]:
                    self.state_stack.pop()
                    self.state_stack.append({"type": self.STATE_EXPECTING_STRING_ARGUMENT_VALUE, 
                                             "processed_input_len": current_input_idx + 1, 
                                             "context": {"start_of_value_token_idx": current_input_idx + 1,
                                                          "current_arg_key": state_info["context"]["current_arg_key"],
                                                          "parent_context": state_info["context"]["parent_context"]
                                                          }})
                    next_state_pushed = True
            
            elif state_type == self.STATE_EXPECTING_STRING_ARGUMENT_VALUE:
                # For strings, we allow most tokens until a closing quote.
                # The actual check for "end of string" is if the *next* token is a quote.
                # So, this state consumes any token and relies on __call__ to offer the closing quote.
                # If the current token IS a quote, it means the string is ending.
                if token_id == self._get_token_ids('"', True)[0]: # String is ending
                    self.state_stack.pop()
                    # Add current_arg_key to defined_args in parent context
                    state_info["context"]["parent_context"]["defined_args"].add(state_info["context"]["current_arg_key"])
                    self.state_stack.append({"type": self.STATE_IN_ARGUMENTS_OBJECT_AFTER_VALUE_EXPECTING_COMMA_OR_END, 
                                             "processed_input_len": current_input_idx + 1, 
                                             "context": {"parent_context": state_info["context"]["parent_context"]}})
                    next_state_pushed = True
                else: # Continue string
                    state_info["processed_input_len"] = current_input_idx + 1
            
            elif state_type == self.STATE_IN_ARGUMENTS_OBJECT_AFTER_VALUE_EXPECTING_COMMA_OR_END:
                parent_context = state_info["context"]["parent_context"]
                if token_id == self._get_token_ids(',', True)[0]:
                    self.state_stack.pop() # Pop this state
                    # Go back to expecting another argument key
                    self.state_stack.append({"type": self.STATE_IN_ARGUMENTS_OBJECT_EXPECTING_KEY_OR_END, 
                                             "processed_input_len": current_input_idx + 1, 
                                             "context": parent_context }) # Pass updated defined_args
                    next_state_pushed = True
                elif token_id == self._get_token_ids('}', True)[0]: # End of arguments
                    self.state_stack.pop() # Pop this state
                    # Pop the STATE_IN_ARGUMENTS_OBJECT_EXPECTING_KEY_OR_END state as well (or its equivalent if factoring changes)
                    # This logic needs to ensure correct stack depth after arguments object.
                    # The parent should be STATE_IN_OBJECT_AFTER_VALUE_EXPECTING_COMMA_OR_END for the main tool object
                    self.state_stack.append({"type": self.STATE_IN_OBJECT_AFTER_VALUE_EXPECTING_COMMA_OR_END, 
                                             "processed_input_len": current_input_idx + 1, 
                                             "context": {"after_key": "arguments", "current_tool_name": parent_context.get("tool_name")}}) # After "arguments": {...}
                    next_state_pushed = True

            elif state_type == self.STATE_IN_OBJECT_AFTER_VALUE_EXPECTING_COMMA_OR_END:
                # This is after "name":"value" or "arguments":{...} for a tool call object
                after_key = state_info["context"].get("after_key")
                if after_key == "name": # Must be followed by a comma, then "arguments"
                    allowed_token_ids = set()
                    allowed_token_ids.update(self._get_allowed_punctuation_tokens([',']))
                elif after_key == "arguments": # Tool call object is complete
                    allowed_token_ids = set()
                    allowed_token_ids.update(self._get_allowed_punctuation_tokens([',', '}'])) # Comma for next tool, or } to close this tool object


            elif state_type == self.STATE_COMPLETE:
                # Should not consume more tokens if complete.
                # print("Warning: Consuming token in STATE_COMPLETE")
                break


            if not next_state_pushed and self.state_stack and self.state_stack[-1]["type"] == state_type : # If no state transition occurred for this token but it was consumed by current state
                self.state_stack[-1]["processed_input_len"] = current_input_idx + 1
            
            current_input_idx += 1
            if self.state_stack and self.state_stack[-1]["type"] == self.STATE_COMPLETE: # Stop if outer list is closed
                break
        
        # Clean up stack if processed_input_len matches current_input_idx for multiple top states
        # This can happen if states are just passing through context without consuming tokens themselves.
        # More robust way: ensure each state *consumes* a token or makes a definitive transition.


    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        current_input_ids_tensor = input_ids[0] # Operate on the first batch element

        # Synchronize state_stack with the latest input_ids
        self._synchronize_state_with_input(current_input_ids_tensor)

        if not self.state_stack:
            # print("Error: State stack empty in __call__")
            # Fallback: allow EOS
            final_mask = torch.zeros(scores.shape[-1], dtype=torch.bool, device=self.llm_device)
            if self.tokenizer.eos_token_id is not None:
                final_mask[self.tokenizer.eos_token_id] = True
            scores[0][~final_mask] = -float("inf")
            return scores

        current_state_info = self.state_stack[-1]
        current_state_type = current_state_info["type"]
        # print(f"Call: Input len: {len(current_input_ids_tensor)}, State: {current_state_type}, Processed: {current_state_info['processed_input_len']}")
        
        allowed_token_ids = set()

        if current_state_type == self.STATE_EXPECTING_LIST_START:
            allowed_token_ids.update(self._get_allowed_punctuation_tokens(['[']))
        
        elif current_state_type == self.STATE_IN_LIST_EXPECTING_OBJECT_START_OR_LIST_END:
            allowed_token_ids.update(self._get_allowed_punctuation_tokens(['{'])) # Start new object
             # Can we close the list? If input ends with '{' or '[', probably not yet.
            if len(current_input_ids_tensor) > current_state_info["processed_input_len"] or \
               (len(current_input_ids_tensor) > 0 and current_input_ids_tensor[-1] not in self._get_token_ids('[',True) + self._get_token_ids('{',True) ): # check if not just opened list/obj
                allowed_token_ids.update(self._get_allowed_punctuation_tokens([']'])) # End list
            
            # Allow comma if an object was just completed (i.e., input ends with '}')
            if len(current_input_ids_tensor) > 0 and current_input_ids_tensor[-1] == self._get_token_ids('}', True)[0]:
                 allowed_token_ids.update(self._get_allowed_punctuation_tokens([',']))


        elif current_state_type == self.STATE_IN_OBJECT_EXPECTING_KEY_QUOTED:
            allowed_token_ids.update(self._get_allowed_punctuation_tokens(['"'])) # Start of key quote

        elif current_state_type == self.STATE_EXPECTING_OBJECT_KEY_STRING_VALUE:
            # Allow tokens that form valid keys ("name", "arguments")
            valid_keys = current_state_info["context"]["valid_keys"]
            allowed_token_ids.update(
                self._get_allowed_specific_string_tokens(valid_keys, current_input_ids_tensor, current_state_info["context"])
            )
        
        elif current_state_type == self.STATE_EXPECTING_OBJECT_KEY_STRING_END:
            allowed_token_ids.update(self._get_allowed_punctuation_tokens(['"']))

        elif current_state_type == self.STATE_IN_OBJECT_AFTER_KEY_EXPECTING_COLON:
            allowed_token_ids.update(self._get_allowed_punctuation_tokens([':']))

        elif current_state_type == self.STATE_EXPECTING_TOOL_NAME_STRING_START:
            allowed_token_ids.update(self._get_allowed_punctuation_tokens(['"']))
        
        elif current_state_type == self.STATE_EXPECTING_TOOL_NAME_STRING_VALUE:
            tool_names = self.schema["tool_names"]
            allowed_token_ids.update(
                self._get_allowed_specific_string_tokens(tool_names, current_input_ids_tensor, current_state_info["context"])
            )
            # If a full tool name is typed, next should be a quote
            current_partial_text = self.tokenizer.decode(current_input_ids_tensor[current_state_info["context"].get("start_of_value_token_idx", len(current_input_ids_tensor)):])
            if current_partial_text in tool_names:
                 allowed_token_ids.clear() # Force only quote if full name typed
                 allowed_token_ids.update(self._get_allowed_punctuation_tokens(['"']))


        elif current_state_type == self.STATE_EXPECTING_TOOL_NAME_STRING_END:
            allowed_token_ids.update(self._get_allowed_punctuation_tokens(['"']))

        elif current_state_type == self.STATE_EXPECTING_ARGUMENTS_OBJECT_START:
            allowed_token_ids.update(self._get_allowed_punctuation_tokens(['{']))

        elif current_state_type == self.STATE_IN_ARGUMENTS_OBJECT_EXPECTING_KEY_OR_END:
            # Allow start of new arg key
            allowed_token_ids.update(self._get_allowed_punctuation_tokens(['"'])) 
            # Allow end of arguments object '}'
            # Only allow '}' if all required args are present or no args are defined for the tool.
            # This check needs enhancement based on schema's required fields.
            # For now, allow if no specific available_arg_keys left or if tool has no params.
            can_close = True
            if self.current_tool_definition and self.current_tool_definition["parameters"]:
                available_arg_keys = current_state_info["context"]["available_arg_keys"]
                defined_args = current_state_info["context"]["defined_args"]
                # A simple check: if there are still available keys that are not yet defined.
                # More complex: check for *required* keys.
                if any(key not in defined_args for key in available_arg_keys):
                     # Simplified: if there are any available keys not yet defined, don't strongly push '}'
                     # unless it's the only option left by schema.
                     pass # Don't restrict '}' if some optional keys are still possible
            
            if can_close : # or if no params defined current_tool_definition and not self.current_tool_definition.get("parameters")
                allowed_token_ids.update(self._get_allowed_punctuation_tokens(['}']))


        elif current_state_type == self.STATE_EXPECTING_ARGUMENT_KEY_STRING_VALUE:
            if self.current_tool_definition and self.current_tool_definition["parameters"]:
                parent_context = current_state_info["context"]["parent_context"]
                arg_names = [k for k in parent_context["available_arg_keys"] if k not in parent_context["defined_args"]]
                allowed_token_ids.update(
                    self._get_allowed_specific_string_tokens(arg_names, current_input_ids_tensor, current_state_info["context"])
                )
                # If a full arg name is typed, next should be a quote
                current_partial_text = self.tokenizer.decode(current_input_ids_tensor[current_state_info["context"].get("start_of_value_token_idx", len(current_input_ids_tensor)):])
                if current_partial_text in arg_names: # Check against remaining available arg_names
                    allowed_token_ids.clear()
                    allowed_token_ids.update(self._get_allowed_punctuation_tokens(['"']))


        elif current_state_type == self.STATE_EXPECTING_ARGUMENT_KEY_STRING_END:
            allowed_token_ids.update(self._get_allowed_punctuation_tokens(['"']))
        
        elif current_state_type == self.STATE_AFTER_ARGUMENT_KEY_EXPECTING_COLON:
            allowed_token_ids.update(self._get_allowed_punctuation_tokens([':']))

        elif current_state_type == self.STATE_EXPECTING_STRING_ARGUMENT_START:
             allowed_token_ids.update(self._get_allowed_punctuation_tokens(['"']))

        elif current_state_type == self.STATE_EXPECTING_STRING_ARGUMENT_VALUE:
            # Allow any token *except* an unescaped quote.
            # The main constraint is that the string must eventually end with a quote.
            all_vocab_ids = set(range(self.tokenizer.vocab_size))
            # For simplicity, we allow the quote here, and _synchronize_state_with_input will transition if it's picked.
            # A more restrictive approach would disallow quote here and force transition based on it.
            # forbidden_ids = set(self._get_allowed_punctuation_tokens(['"'])) 
            # allowed_token_ids.update(all_vocab_ids - forbidden_ids)
            allowed_token_ids.update(all_vocab_ids) # Allow any token, including quote to end the string.
                                                    # The _synchronize method will handle the transition.

        # ELIF for STATE_EXPECTING_STRING_ARGUMENT_END is removed as sync logic handles quote.

        elif current_state_type == self.STATE_IN_ARGUMENTS_OBJECT_AFTER_VALUE_EXPECTING_COMMA_OR_END:
            allowed_token_ids.update(self._get_allowed_punctuation_tokens([',', '}']))

        elif current_state_type == self.STATE_IN_OBJECT_AFTER_VALUE_EXPECTING_COMMA_OR_END:
            # After "name": "value" (key "name"), expect comma for "arguments"
            # After "arguments": {...} (key "arguments"), expect comma for next tool obj in list, or } to close tool obj
            after_key = current_state_info["context"].get("after_key")
            if after_key == "name": # Must be followed by a comma, then "arguments"
                allowed_token_ids.update(self._get_allowed_punctuation_tokens([',']))
            elif after_key == "arguments": # Tool call object is complete
                allowed_token_ids.update(self._get_allowed_punctuation_tokens([',', '}'])) # Comma for next tool, or } to close this tool object

        elif current_state_type == self.STATE_COMPLETE:
            if self.tokenizer.eos_token_id is not None:
                allowed_token_ids.add(self.tokenizer.eos_token_id)

        if not allowed_token_ids and current_state_type != self.STATE_COMPLETE:
            # Fallback if logic error leads to no allowed tokens
            # print(f"Warning: No tokens allowed by __call__ for state {current_state_type}. Allowing EOS.")
            if self.tokenizer.eos_token_id is not None:
                 allowed_token_ids.add(self.tokenizer.eos_token_id)

        final_mask = torch.zeros(scores.shape[-1], dtype=torch.bool, device=self.llm_device)
        if allowed_token_ids: # Ensure not empty before trying to index
            final_mask[list(allowed_token_ids)] = True
        else: # If truly no tokens allowed (e.g. only EOS was option but no EOS id)
             pass # scores will remain -inf for all, or model handles it.

        scores[0][~final_mask] = -float("inf")
        return scores

    # def _MANAGE_STATE_TRANSITIONS_CONCEPT(self, chosen_token_id): # Renamed to avoid execution
    #     # This function is conceptual and demonstrates how state transitions would be managed
    #     # based on the chosen token. The actual logic is now in _synchronize_state_with_input.
    #     current_state_info = self.state_stack[-1]
    #     current_state_type = current_state_info["type"]

    #     # Example transitions (conceptual):
    #     if current_state_type == self.STATE_EXPECTING_LIST_START and chosen_token_id == self._get_token_ids('[', True)[0]:
    #         # self.state_stack.append({"type": self.STATE_IN_LIST_EXPECTING_OBJECT_START_OR_LIST_END, "context_data": "example"})
    #         pass
        
    #     elif current_state_type == self.STATE_EXPECTING_TOOL_NAME_STRING_VALUE:
    #         # current_text = self.tokenizer.decode(current_input_ids + [chosen_token_id]) # pseudo code
    #         # partial_name = current_text[current_state_info["context"].get("start_of_value_token_idx", 0):]
    #         # if partial_name in self.schema["tool_names"]:
    #         #    self.state_stack.pop() 
    #         #    self.state_stack.append({"type": self.STATE_EXPECTING_TOOL_NAME_STRING_END, "context_data": "example"})
    #         # elif any(tn.startswith(partial_name) for tn in self.schema["tool_names"]):
    #         #    pass # Stay in this state
    #         # else: 
    #         #    pass # Error or unexpected token
    #         pass


def generate_structured_llm_output(sample, model_llm, tokenizer_llm, **_):
    """
    Generates structured output from the LLM based on the input query and tools,
    using a LogitsProcessor for schema adherence.

    Args:
        sample (dict): A dictionary containing 'query' and 'tools'.
                       'tools' is a JSON string describing the available tools.
        model_llm: The language model.
        tokenizer_llm: The tokenizer for the language model.

    Returns:
        list: A list of dictionaries representing the function calls, or an empty list if parsing fails or generation is constrained.
    """
    
    tools_list_for_prompt = json.loads(sample['tools']) # For the prompt
    
    system_prompt = """You are a helpful assistant that can use tools.
Based on the user's query, you should decide which tools to use and what parameters to pass to them.
Respond with a JSON list of function calls. Each function call should be an object with "name" and "arguments" keys.
If no tool is appropriate, or if you cannot determine the arguments, respond with an empty list []."""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Query: {sample['query']}"}
    ]

    # The apply_chat_template might handle some tool formatting for certain models.
    # The LogitsProcessor will provide stricter enforcement.
    prompt_for_llm = tokenizer_llm.apply_chat_template(
        messages,
        tools=tools_list_for_prompt, 
        add_generation_prompt=True, 
        tokenize=False,
        # Important: Some models might expect tools in a specific part of the prompt.
        # The LogitsProcessor works independently of this, but a good prompt helps.
    )

    inputs = tokenizer_llm(prompt_for_llm, return_tensors="pt", return_dict=True).to(model_llm.device)
    input_len = inputs["input_ids"].shape[-1]

    # Initialize the LogitsProcessor
    # Pass the raw JSON string of tools, tokenizer, and device
    schema_processor = JsonSchemaLogitsProcessor(
        tools_json_string=sample['tools'], 
        tokenizer=tokenizer_llm,
        llm_device=model_llm.device
    )
    
    logits_processor_list = LogitsProcessorList([schema_processor])

    with torch.no_grad():
        outputs = model_llm.generate(
            **inputs,
            max_new_tokens=256,
            num_return_sequences=1,
            do_sample=False, 
            pad_token_id=tokenizer_llm.eos_token_id,
            logits_processor=logits_processor_list # Add the custom processor
        )
    
    output_ids = outputs[0][input_len:]
    output_text = tokenizer_llm.decode(output_ids, skip_special_tokens=True)

    try:
        json_start_index = output_text.find('[')
        json_end_index = output_text.rfind(']') + 1
        
        if json_start_index != -1 and json_end_index != -1 and json_start_index < json_end_index:
            json_string = output_text[json_start_index:json_end_index]
            structured_output = json.loads(json_string)
        else:
            json_start_index = output_text.find('{') # Fallback for single tool call, not in a list
            json_end_index = output_text.rfind('}') + 1
            if json_start_index != -1 and json_end_index != -1 and json_start_index < json_end_index:
                json_string = output_text[json_start_index:json_end_index]
                structured_output = [json.loads(json_string)] # Wrap in list
            else:
                print(f"Warning: Could not find valid JSON list/object in LLM output: {output_text}")
                return [] 
        return structured_output
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from LLM output: {e}")
        print(f"LLM Raw Output: {output_text}")
        return []

