#!/usr/bin/env python3
"""
Test script for the prefix-validated generation function.
"""

import sys
import os

# Add parent directory to path to import from src
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from src.llm import (
    load_llm, 
    generate_prompt_str, 
    tokenize_for_llm,
    generate_llm_output,
    generate_with_prefix_validation,
    is_valid_output_prefix
)
import json
import torch


def test_prefix_generation():
    """Test the prefix-validated generation function."""
    
    print("Testing Prefix-Validated Generation")
    print("=" * 80)
    
    # Example configuration
    CONFIG = {
        "llm_model_name": "NousResearch/Hermes-3-Llama-3.2-3B",  # Small model for testing
    }
    
    print(f"\nLoading model: {CONFIG['llm_model_name']}...")
    try:
        model, tokenizer = load_llm(CONFIG)
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("Note: This test requires a GPU and the model to be available.")
        return
    
    # Test samples with tools
    test_samples = [
        {
            'query': "What's the weather in San Francisco and New York?",
            'tools': json.dumps([
                {
                    "name": "get_weather",
                    "description": "Get weather information for a location",
                    "parameters": {
                        "location": {
                            "description": "The city and state, e.g. San Francisco, CA",
                            "type": "str",
                            "required": True
                        },
                        "unit": {
                            "description": "Temperature unit",
                            "type": "str",
                            "required": False,
                            "default": "celsius"
                        }
                    }
                }
            ])
        },
        {
            'query': "Search for information about Python programming and machine learning.",
            'tools': json.dumps([
                {
                    "name": "search_web",
                    "description": "Search the web for information",
                    "parameters": {
                        "query": {
                            "description": "Search query",
                            "type": "str",
                            "required": True
                        },
                        "num_results": {
                            "description": "Number of results to return",
                            "type": "int",
                            "required": False,
                            "default": 10
                        }
                    }
                }
            ])
        }
    ]
    
    for i, sample in enumerate(test_samples):
        print(f"\n{'='*80}")
        print(f"Test Case {i+1}")
        print(f"{'='*80}")
        print(f"Query: {sample['query']}")
        
        tools = json.loads(sample['tools'])
        print(f"Available tools: {[tool['name'] for tool in tools]}")
        
        # Tokenize input
        inputs = tokenize_for_llm(sample, tokenizer, model.device)
        
        print("\n1. Standard Generation (may produce invalid prefixes):")
        print("-" * 40)
        try:
            standard_output = generate_llm_output(sample, model, tokenizer)
            print(f"Output: {standard_output}")
            
            # Check all prefixes
            invalid_count = 0
            for j in range(len(standard_output) + 1):
                prefix = standard_output[:j]
                if not is_valid_output_prefix(tools, prefix):
                    invalid_count += 1
            
            if invalid_count > 0:
                print(f"⚠️  Found {invalid_count} invalid prefixes")
            else:
                print("✅ All prefixes are valid")
                
        except Exception as e:
            print(f"Error in standard generation: {e}")
        
        print("\n2. Prefix-Validated Generation (guarantees valid prefixes):")
        print("-" * 40)
        try:
            validated_output = generate_with_prefix_validation(
                model=model,
                tokenizer=tokenizer,
                tools=tools,
                input_ids=inputs["input_ids"],
                max_new_tokens=150,
                temperature=0.1
            )
            print(f"Output: {validated_output}")
            
            # Verify all prefixes are valid
            all_valid = True
            for j in range(len(validated_output) + 1):
                prefix = validated_output[:j]
                if not is_valid_output_prefix(tools, prefix):
                    all_valid = False
                    print(f"❌ Invalid prefix at position {j}: {repr(prefix)}")
                    break
            
            if all_valid:
                print("✅ All prefixes are valid (as guaranteed)")
                
            # Try to parse the final output
            try:
                parsed = json.loads(validated_output.strip())
                print(f"✅ Valid JSON output: {len(parsed)} function calls")
            except:
                print("⚠️  Output is not complete JSON (may have stopped early)")
                
        except Exception as e:
            print(f"Error in prefix-validated generation: {e}")
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print("The prefix-validated generation ensures that every intermediate")
    print("state during generation is a valid prefix of a function call output.")
    print("This prevents the model from generating invalid JSON structures.")


def compare_generation_methods():
    """Compare different generation methods on a sample."""
    print("\n" + "="*80)
    print("GENERATION METHOD COMPARISON")
    print("="*80)
    
    # Simple example that often causes issues
    sample = {
        'query': "Get the weather for cities: ['New York', 'Los Angeles']",
        'tools': json.dumps([{
            "name": "get_weather",
            "description": "Get weather",
            "parameters": {
                "city": {"type": "str", "required": True}
            }
        }])
    }
    
    tools = json.loads(sample['tools'])
    
    # Test different prefixes
    test_prefixes = [
        '[{"name": "get_weather", "arguments": {"city": "New York"}}, {"name": "get_weather", "arguments": {"city": "Los Angeles"}}]',
        '[{"name": "get_weather", "arguments": {"city": ["New York", "Los Angeles"]}}]',  # Invalid: array instead of string
        '[["get_weather", "New York"], ["get_weather", "Los Angeles"]]',  # Invalid format
    ]
    
    print("\nValidating different output formats:")
    for prefix in test_prefixes:
        is_valid = is_valid_output_prefix(tools, prefix)
        print(f"\n{'✅' if is_valid else '❌'} {repr(prefix[:60])}...")
        if not is_valid:
            # Find first invalid prefix
            for i in range(len(prefix)):
                if not is_valid_output_prefix(tools, prefix[:i+1]):
                    print(f"   First invalid at position {i}: {repr(prefix[:i+1])}")
                    break


if __name__ == "__main__":
    print("Choose test mode:")
    print("1. Test prefix generation with model (requires GPU)")
    print("2. Compare generation methods (no model needed)")
    
    choice = input("\nEnter choice (1 or 2): ").strip()
    
    if choice == "1":
        test_prefix_generation()
    elif choice == "2":
        compare_generation_methods()
    else:
        print("Invalid choice. Running comparison test...")
        compare_generation_methods() 