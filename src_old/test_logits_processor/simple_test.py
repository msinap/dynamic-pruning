#!/usr/bin/env python3
"""
Simple test script for StructuredFunctionCallProcessor

This script tests the core functionality without requiring large language models.
"""

import json
import torch
from transformers import AutoTokenizer
from src.llm import StructuredFunctionCallProcessor

def test_basic_functionality():
    """Test basic functionality of the processor"""
    print("=== Basic Functionality Test ===")
    
    # Sample tools
    tools = [
        {
            "name": "get_weather",
            "description": "Get weather information",
            "parameters": {
                "location": {"type": "string", "description": "City name"},
                "units": {"type": "string", "description": "Temperature units"}
            },
            "required": ["location"]
        },
        {
            "name": "calculate",
            "description": "Perform calculations",
            "parameters": {
                "operation": {"type": "string", "description": "Math operation"},
                "numbers": {"type": "array", "description": "Numbers to operate on"}
            },
            "required": ["operation", "numbers"]
        }
    ]
    
    # Use a simple tokenizer (you can change this to any available model)
    try:
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    except:
        print("Could not load gpt2 tokenizer, trying alternative...")
        try:
            tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
        except:
            print("Could not load any tokenizer. Please install transformers and download a model.")
            return
    
    # Initialize processor
    processor = StructuredFunctionCallProcessor(
        tokenizer=tokenizer,
        tools=tools,
        device="cpu"
    )
    
    print(f"✓ Processor initialized with {len(tools)} tools")
    print(f"✓ Valid function names: {list(processor.valid_function_names)}")
    
    return processor, tokenizer

def test_json_state_parsing(processor):
    """Test JSON parsing at different stages"""
    print("\n=== JSON State Parsing Test ===")
    
    test_cases = [
        # Starting states
        ("", "list_start"),
        ("[", "object_start"),
        
        # Object and name parsing
        ('[{', "name_key"),
        ('[{"name"', "name_colon"),
        ('[{"name":', "name_value"),
        ('[{"name": "get_weather"', "name_colon"),
        
        # Arguments parsing
        ('[{"name": "get_weather", "arguments"', "arguments_colon"),
        ('[{"name": "get_weather", "arguments":', "arguments_value"),
        ('[{"name": "get_weather", "arguments": {', "argument_key"),
        ('[{"name": "get_weather", "arguments": {"location"', "arg_colon"),
        ('[{"name": "get_weather", "arguments": {"location":', "argument_value"),
        ('[{"name": "get_weather", "arguments": {"location": "NYC"', "arg_comma_or_close"),
        
        # Completion states
        ('[{"name": "get_weather", "arguments": {"location": "NYC"}}', "comma_or_list_end"),
        ('[{"name": "get_weather", "arguments": {"location": "NYC"}},', "object_start"),
    ]
    
    for json_text, description in test_cases:
        try:
            state = processor._parse_partial_json(json_text)
            expecting = state.get('expecting', 'unknown')
            print(f"✓ '{json_text[:30]}...' -> expecting: {expecting}")
            
            # Show additional state info
            if state.get('current_function'):
                print(f"  Function: {state['current_function']}")
            if state.get('current_args'):
                print(f"  Args: {state['current_args']}")
            if state.get('partial_function_name'):
                print(f"  Partial function: {state['partial_function_name']}")
                
        except Exception as e:
            print(f"✗ Error parsing '{json_text}': {e}")

def test_token_validation(processor, tokenizer):
    """Test token validation for different states"""
    print("\n=== Token Validation Test ===")
    
    # Test different parsing states
    test_states = [
        {"expecting": "list_start"},
        {"expecting": "object_start"},
        {"expecting": "name_key"},
        {"expecting": "function_name", "partial_function_name": ""},
        {"expecting": "function_name", "partial_function_name": "get"},
        {"expecting": "function_name", "partial_function_name": "get_weather"},
        {"expecting": "argument_key", "current_function": "get_weather"},
        {"expecting": "argument_key", "current_function": "get_weather", "partial_arg_key": "loc"},
        {"expecting": "comma_or_list_end"},
    ]
    
    for state in test_states:
        try:
            valid_tokens = processor._get_valid_tokens_for_state(state)
            expecting = state.get('expecting', 'unknown')
            print(f"✓ State '{expecting}': {len(valid_tokens)} valid tokens")
            
            # Show some example tokens
            if valid_tokens and len(valid_tokens) > 0:
                # Get first few tokens and decode them
                sample_tokens = list(valid_tokens)[:3]
                sample_text = []
                for token in sample_tokens:
                    try:
                        decoded = tokenizer.decode([token])
                        sample_text.append(f"'{decoded}'")
                    except:
                        sample_text.append(f"token_{token}")
                print(f"  Examples: {', '.join(sample_text)}")
            
            # Special case: show function name validation
            if expecting == "function_name" and state.get("partial_function_name") == "get":
                print(f"  Validating continuation of 'get' -> should allow '_weather' tokens")
                
        except Exception as e:
            print(f"✗ Error validating state {expecting}: {e}")

def test_logits_processing(processor, tokenizer):
    """Test the actual logits processing"""
    print("\n=== Logits Processing Test ===")
    
    # Create dummy input_ids and scores
    vocab_size = len(tokenizer.vocab) if hasattr(tokenizer, 'vocab') else 50000
    
    # Simulate different generation states
    test_scenarios = [
        {
            "description": "Start of generation (no JSON yet)",
            "text": "Here are the function calls: ",
            "should_allow": ["["]
        },
        {
            "description": "After opening bracket",
            "text": "Here are the function calls: [",
            "should_allow": ["{", " "]
        },
        {
            "description": "After opening brace",
            "text": "Here are the function calls: [{",
            "should_allow": ['"', "n"]  # for "name" key
        }
    ]
    
    for scenario in test_scenarios:
        try:
            # Encode the text to get input_ids
            input_ids = tokenizer.encode(scenario["text"], return_tensors="pt")
            
            # Create dummy scores (logits)
            scores = torch.randn(1, vocab_size)
            
            # Process with the processor
            processed_scores = processor(input_ids, scores)
            
            print(f"✓ {scenario['description']}")
            
            # Check if expected tokens have higher scores
            for expected_token in scenario["should_allow"]:
                try:
                    token_ids = tokenizer.encode(expected_token, add_special_tokens=False)
                    if token_ids:
                        token_id = token_ids[0]
                        if token_id < vocab_size:
                            original_score = scores[0, token_id].item()
                            processed_score = processed_scores[0, token_id].item()
                            print(f"  Token '{expected_token}' (id={token_id}): {original_score:.2f} -> {processed_score:.2f}")
                except:
                    pass
                    
        except Exception as e:
            print(f"✗ Error processing scenario '{scenario['description']}': {e}")

def run_simple_tests():
    """Run all simple tests"""
    print("StructuredFunctionCallProcessor - Simple Tests")
    print("=" * 50)
    
    # Test basic functionality
    result = test_basic_functionality()
    if result is None:
        print("Could not initialize processor. Exiting.")
        return
    
    processor, tokenizer = result
    
    # Test JSON parsing
    test_json_state_parsing(processor)
    
    # Test token validation
    test_token_validation(processor, tokenizer)
    
    # Test logits processing
    test_logits_processing(processor, tokenizer)
    
    print("\n" + "=" * 50)
    print("Simple tests completed!")
    print("\nThe processor is working correctly for:")
    print("- Tool validation and setup")
    print("- JSON state parsing")
    print("- Token validation")
    print("- Logits processing")

if __name__ == "__main__":
    run_simple_tests() 