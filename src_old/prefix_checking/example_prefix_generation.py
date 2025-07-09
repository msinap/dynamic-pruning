#!/usr/bin/env python3
"""
Simple example of using prefix-validated generation.
"""

import sys
import os

# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from src.llm import generate_with_prefix_validation, is_valid_output_prefix
import json


def demonstrate_prefix_validation():
    """Demonstrate how prefix validation ensures valid outputs."""
    
    print("Prefix-Validated Generation Example")
    print("=" * 60)
    
    # Define tools
    tools = [
        {
            "name": "calculate",
            "description": "Perform a calculation",
            "parameters": {
                "expression": {"type": "str", "required": True},
                "precision": {"type": "int", "required": False}
            }
        },
        {
            "name": "convert_unit",
            "description": "Convert between units",
            "parameters": {
                "value": {"type": "float", "required": True},
                "from_unit": {"type": "str", "required": True},
                "to_unit": {"type": "str", "required": True}
            }
        }
    ]
    
    print("\nAvailable tools:")
    for tool in tools:
        print(f"  - {tool['name']}: {tool['description']}")
    
    print("\n" + "-" * 60)
    print("Demonstrating prefix validation:")
    print("-" * 60)
    
    # Example prefixes showing the generation process
    generation_steps = [
        "",
        "[",
        "[{",
        '[{"',
        '[{"name"',
        '[{"name":',
        '[{"name": "',
        '[{"name": "calc',
        '[{"name": "calculate"',
        '[{"name": "calculate",',
        '[{"name": "calculate", "',
        '[{"name": "calculate", "arguments"',
        '[{"name": "calculate", "arguments":',
        '[{"name": "calculate", "arguments": {',
        '[{"name": "calculate", "arguments": {"',
        '[{"name": "calculate", "arguments": {"expression"',
        '[{"name": "calculate", "arguments": {"expression":',
        '[{"name": "calculate", "arguments": {"expression": "',
        '[{"name": "calculate", "arguments": {"expression": "2+2"',
        '[{"name": "calculate", "arguments": {"expression": "2+2"}',
        '[{"name": "calculate", "arguments": {"expression": "2+2"}}',
        '[{"name": "calculate", "arguments": {"expression": "2+2"}}]',
    ]
    
    print("\nSimulating token-by-token generation:")
    for i, prefix in enumerate(generation_steps):
        is_valid = is_valid_output_prefix(tools, prefix)
        status = "✅" if is_valid else "❌"
        
        # Show what was added
        if i > 0:
            added = prefix[len(generation_steps[i-1]):]
            print(f"\nStep {i}: Adding '{added}'")
        else:
            print(f"\nStep {i}: Starting with empty string")
            
        print(f"{status} Current prefix: {repr(prefix)}")
        print(f"   Valid: {is_valid}")
    
    print("\n" + "-" * 60)
    print("Invalid prefix examples:")
    print("-" * 60)
    
    invalid_examples = [
        ('[{"name": "invalid_function"}]', "Unknown function name"),
        ('[{"invalid_key": "value"}]', "Invalid object key"),
        ('[{"name": "calculate", "arguments": {"wrong_param": "value"}}]', "Invalid parameter"),
        ('[[', "Nested arrays not allowed"),
        ('[{"name": "calculate", "arguments": {}}]', "Missing required parameter"),
    ]
    
    for prefix, reason in invalid_examples:
        is_valid = is_valid_output_prefix(tools, prefix)
        print(f"\n❌ {repr(prefix)}")
        print(f"   Reason: {reason}")
        
        # Find where it becomes invalid
        for i in range(len(prefix)):
            if not is_valid_output_prefix(tools, prefix[:i+1]):
                print(f"   First invalid at position {i}: {repr(prefix[:i+1])}")
                break
    
    print("\n" + "=" * 60)
    print("Key Benefits of Prefix-Validated Generation:")
    print("=" * 60)
    print("1. Guarantees syntactically valid JSON at every step")
    print("2. Ensures function names match available tools")
    print("3. Validates parameter names and types")
    print("4. Prevents common JSON formatting errors")
    print("5. Can stop generation early if no valid continuation exists")


def show_usage_example():
    """Show how to use the generation function in practice."""
    
    print("\n" + "=" * 60)
    print("Usage Example (Pseudo-code)")
    print("=" * 60)
    
    example_code = '''
# Load your model and tokenizer
model, tokenizer = load_llm(config)

# Define available tools
tools = [
    {
        "name": "get_weather",
        "parameters": {
            "location": {"type": "str", "required": True}
        }
    }
]

# Prepare input
sample = {
    "query": "What's the weather in Paris?",
    "tools": json.dumps(tools)
}
inputs = tokenize_for_llm(sample, tokenizer, model.device)

# Generate with prefix validation
output = generate_with_prefix_validation(
    model=model,
    tokenizer=tokenizer,
    tools=tools,
    input_ids=inputs["input_ids"],
    max_new_tokens=150,
    temperature=0.1
)

print(f"Generated output: {output}")
# Output: [{"name": "get_weather", "arguments": {"location": "Paris"}}]
'''
    
    print(example_code)


if __name__ == "__main__":
    demonstrate_prefix_validation()
    show_usage_example() 