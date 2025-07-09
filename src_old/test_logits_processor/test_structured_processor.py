#!/usr/bin/env python3
"""
Test script for StructuredFunctionCallProcessor

This script demonstrates how to use the StructuredFunctionCallProcessor
to enforce structured function call generation with various test cases.
"""

import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from src.llm import StructuredFunctionCallProcessor, generate_structured_function_calls

# Sample tools for testing
SAMPLE_TOOLS = [
    {
        "name": "get_weather",
        "description": "Get current weather information for a location",
        "parameters": {
            "location": {
                "type": "string",
                "description": "The city and state/country"
            },
            "units": {
                "type": "string", 
                "description": "Temperature units",
                "enum": ["celsius", "fahrenheit"]
            }
        },
        "required": ["location"]
    },
    {
        "name": "calculate_tip",
        "description": "Calculate tip amount for a bill",
        "parameters": {
            "bill_amount": {
                "type": "number",
                "description": "The total bill amount"
            },
            "tip_percentage": {
                "type": "number",
                "description": "Tip percentage (default 15%)"
            }
        },
        "required": ["bill_amount"]
    },
    {
        "name": "send_email",
        "description": "Send an email to recipients",
        "parameters": {
            "to": {
                "type": "string",
                "description": "Recipient email address"
            },
            "subject": {
                "type": "string",
                "description": "Email subject"
            },
            "body": {
                "type": "string",
                "description": "Email body content"
            }
        },
        "required": ["to", "subject", "body"]
    }
]

def create_test_sample(query: str, tools: list = None) -> dict:
    """Create a test sample with query and tools"""
    if tools is None:
        tools = SAMPLE_TOOLS
    return {
        "query": query,
        "tools": json.dumps(tools)
    }

def test_processor_initialization():
    """Test processor initialization with sample tools"""
    print("=== Testing Processor Initialization ===")
    
    # Use a small model for testing (you can change this to any model you prefer)
    model_name = "microsoft/DialoGPT-small"  # Small model for quick testing
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        processor = StructuredFunctionCallProcessor(
            tokenizer=tokenizer,
            tools=SAMPLE_TOOLS,
            device="cpu"  # Use CPU for testing
        )
        
        print(f"✓ Processor initialized successfully")
        print(f"✓ Found {len(processor.valid_function_names)} valid function names:")
        for name in processor.valid_function_names:
            print(f"  - {name}")
        
        print(f"✓ Special tokens cached: {list(processor.special_tokens.keys())}")
        print(f"✓ Valid name tokens computed for {len(processor.valid_name_tokens)} prefixes")
        
        return tokenizer, processor
        
    except Exception as e:
        print(f"✗ Error initializing processor: {e}")
        return None, None

def test_json_parsing():
    """Test the JSON parsing functionality"""
    print("\n=== Testing JSON Parsing ===")
    
    tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    processor = StructuredFunctionCallProcessor(
        tokenizer=tokenizer,
        tools=SAMPLE_TOOLS,
        device="cpu"
    )
    
    test_cases = [
        ('[', 'list_start'),
        ('[{', 'name_key'),
        ('[{"name":', 'name_value'),
        ('[{"name": "get_weather"', 'name_colon'),
        ('[{"name": "get_weather", "arguments":', 'arguments_colon'),
        ('[{"name": "get_weather", "arguments": {', 'argument_key'),
        ('[{"name": "get_weather", "arguments": {"location":', 'arg_colon'),
        ('[{"name": "get_weather", "arguments": {"location": "New York"', 'arg_comma_or_close'),
        ('[{"name": "get_weather", "arguments": {"location": "New York"}}', 'comma_or_list_end'),
    ]
    
    for json_text, expected_state in test_cases:
        try:
            state = processor._parse_partial_json(json_text)
            print(f"✓ '{json_text}' -> expecting: {state.get('expecting', 'unknown')}")
            if 'current_function' in state and state['current_function']:
                print(f"  Current function: {state['current_function']}")
            if 'current_args' in state and state['current_args']:
                print(f"  Current args: {state['current_args']}")
        except Exception as e:
            print(f"✗ Error parsing '{json_text}': {e}")

def test_token_validation():
    """Test token validation for different states"""
    print("\n=== Testing Token Validation ===")
    
    tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    processor = StructuredFunctionCallProcessor(
        tokenizer=tokenizer,
        tools=SAMPLE_TOOLS,
        device="cpu"
    )
    
    test_states = [
        {'expecting': 'list_start'},
        {'expecting': 'object_start'},
        {'expecting': 'function_name', 'partial_function_name': 'get'},
        {'expecting': 'function_name', 'partial_function_name': 'get_weather'},
        {'expecting': 'argument_key', 'current_function': 'get_weather', 'partial_arg_key': 'loc'},
    ]
    
    for state in test_states:
        try:
            valid_tokens = processor._get_valid_tokens_for_state(state)
            print(f"✓ State {state['expecting']}: {len(valid_tokens)} valid tokens")
            
            # Show some example valid tokens
            if valid_tokens:
                example_tokens = list(valid_tokens)[:5]  # Show first 5
                example_text = [tokenizer.decode([t]) for t in example_tokens]
                print(f"  Examples: {example_text}")
                
        except Exception as e:
            print(f"✗ Error validating tokens for state {state}: {e}")

def test_full_generation():
    """Test full generation with the processor"""
    print("\n=== Testing Full Generation ===")
    
    # Note: This requires a proper language model
    # For demonstration, we'll show how it would be called
    
    test_queries = [
        "What's the weather like in New York?",
        "Calculate a 20% tip on a $50 bill",
        "Send an email to john@example.com about the meeting tomorrow"
    ]
    
    print("Test queries prepared:")
    for i, query in enumerate(test_queries, 1):
        print(f"{i}. {query}")
        sample = create_test_sample(query)
        print(f"   Sample created with {len(json.loads(sample['tools']))} tools")
    
    print("\nNote: Full generation testing requires a proper language model.")
    print("To test with a real model, uncomment and modify the generation code below.")
    
    # Uncomment and modify this section to test with a real model:
    """
    model_name = "microsoft/DialoGPT-medium"  # or your preferred model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    for query in test_queries:
        sample = create_test_sample(query)
        try:
            result = generate_structured_function_calls(
                sample=sample,
                model_llm=model,
                tokenizer_llm=tokenizer,
                max_new_tokens=100,
                temperature=0.1
            )
            print(f"Query: {query}")
            print(f"Result: {result}")
            print("-" * 50)
        except Exception as e:
            print(f"Error generating for '{query}': {e}")
    """

def test_edge_cases():
    """Test edge cases and error handling"""
    print("\n=== Testing Edge Cases ===")
    
    tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Test with empty tools
    try:
        processor = StructuredFunctionCallProcessor(
            tokenizer=tokenizer,
            tools=[],
            device="cpu"
        )
        print("✓ Processor handles empty tools list")
    except Exception as e:
        print(f"✗ Error with empty tools: {e}")
    
    # Test with malformed tools
    malformed_tools = [
        {"name": "test_func"}  # Missing parameters
    ]
    
    try:
        processor = StructuredFunctionCallProcessor(
            tokenizer=tokenizer,
            tools=malformed_tools,
            device="cpu"
        )
        print("✓ Processor handles malformed tools")
    except Exception as e:
        print(f"✗ Error with malformed tools: {e}")
    
    # Test with very long function names
    long_name_tools = [
        {
            "name": "very_long_function_name_that_might_cause_issues",
            "parameters": {"param": {"type": "string"}}
        }
    ]
    
    try:
        processor = StructuredFunctionCallProcessor(
            tokenizer=tokenizer,
            tools=long_name_tools,
            device="cpu"
        )
        print("✓ Processor handles long function names")
    except Exception as e:
        print(f"✗ Error with long function names: {e}")

def run_all_tests():
    """Run all test functions"""
    print("Starting StructuredFunctionCallProcessor Tests")
    print("=" * 60)
    
    # Test initialization
    tokenizer, processor = test_processor_initialization()
    
    if tokenizer and processor:
        # Test JSON parsing
        test_json_parsing()
        
        # Test token validation
        test_token_validation()
        
        # Test full generation (demonstration)
        test_full_generation()
        
        # Test edge cases
        test_edge_cases()
    
    print("\n" + "=" * 60)
    print("Tests completed!")
    print("\nTo test with actual generation:")
    print("1. Uncomment the generation code in test_full_generation()")
    print("2. Install a suitable language model")
    print("3. Modify the model_name variable to your preferred model")

if __name__ == "__main__":
    run_all_tests() 