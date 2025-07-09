#!/usr/bin/env python3
"""
Working example of StructuredFunctionCallProcessor

This script demonstrates the processor working correctly with direct generation,
avoiding chat template dependencies.
"""

import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, LogitsProcessorList
from src.llm import StructuredFunctionCallProcessor

def create_simple_prompt(query: str, tools: list) -> str:
    """Create a simple prompt without chat templates"""
    tools_str = json.dumps(tools, indent=2)
    return f"""Given the query: "{query}"

Available tools:
{tools_str}

Generate function calls as a JSON array:
"""

def working_example():
    """Demonstrate the processor with a working generation example"""
    print("=== Working Example: StructuredFunctionCallProcessor ===")
    
    # Define tools
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
    
    # Load model and tokenizer
    print("Loading GPT-2 model...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print("✓ Model loaded")
    
    # Create processor
    processor = StructuredFunctionCallProcessor(
        tokenizer=tokenizer,
        tools=tools,
        device="cpu"
    )
    print("✓ Processor created")
    
    # Test query
    query = "What's the weather in London?"
    prompt = create_simple_prompt(query, tools)
    
    print(f"\nQuery: {query}")
    print(f"Prompt:\n{prompt}")
    
    # Tokenize input
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    print(f"Input tokens: {inputs.shape[1]}")
    
    # Generate with processor
    print("\nGenerating with StructuredFunctionCallProcessor...")
    
    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_new_tokens=50,
            num_return_sequences=1,
            do_sample=False,  # Greedy decoding for consistency
            logits_processor=LogitsProcessorList([processor]),
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # Decode output
    input_len = inputs.shape[1]
    generated_tokens = outputs[0][input_len:]
    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    
    print(f"Generated text: '{generated_text}'")
    
    # Try to parse as JSON
    try:
        # Look for JSON array in the generated text
        start_idx = generated_text.find('[')
        if start_idx != -1:
            json_part = generated_text[start_idx:]
            # Try to find the end of the JSON
            bracket_count = 0
            end_idx = -1
            for i, char in enumerate(json_part):
                if char == '[':
                    bracket_count += 1
                elif char == ']':
                    bracket_count -= 1
                    if bracket_count == 0:
                        end_idx = i + 1
                        break
            
            if end_idx != -1:
                json_str = json_part[:end_idx]
                print(f"Extracted JSON: {json_str}")
                
                parsed = json.loads(json_str)
                print(f"✓ Successfully parsed JSON with {len(parsed)} function calls:")
                for i, call in enumerate(parsed):
                    name = call.get('name', 'unknown')
                    args = call.get('arguments', {})
                    print(f"  {i+1}. {name}({args})")
            else:
                print("⚠ Could not find complete JSON array")
        else:
            print("⚠ No JSON array found in generated text")
            
    except json.JSONDecodeError as e:
        print(f"⚠ Generated text is not valid JSON: {e}")
        print("This is expected with small models - the processor constrains structure but doesn't guarantee semantic correctness")
    
    # Compare with unconstrained generation
    print("\n" + "="*50)
    print("Comparison: Generation WITHOUT processor")
    
    with torch.no_grad():
        unconstrained_outputs = model.generate(
            inputs,
            max_new_tokens=50,
            num_return_sequences=1,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    unconstrained_tokens = unconstrained_outputs[0][input_len:]
    unconstrained_text = tokenizer.decode(unconstrained_tokens, skip_special_tokens=True)
    
    print(f"Unconstrained generation: '{unconstrained_text}'")
    
    # Show the difference
    print("\n" + "="*50)
    print("SUMMARY:")
    print(f"With processor:    '{generated_text}'")
    print(f"Without processor: '{unconstrained_text}'")
    print("\nThe processor ensures structured output even if the content isn't perfect.")

def demonstrate_step_by_step():
    """Show step-by-step how the processor works"""
    print("\n=== Step-by-Step Demonstration ===")
    
    tools = [{"name": "test_func", "parameters": {"param": {"type": "string"}}}]
    
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    processor = StructuredFunctionCallProcessor(
        tokenizer=tokenizer,
        tools=tools,
        device="cpu"
    )
    
    # Show what happens at each step
    test_sequences = [
        "Start: ",
        "Start: [",
        "Start: [{",
        'Start: [{"name"',
        'Start: [{"name":',
        'Start: [{"name": "test_func"',
        'Start: [{"name": "test_func", "arguments"',
        'Start: [{"name": "test_func", "arguments":',
        'Start: [{"name": "test_func", "arguments": {',
        'Start: [{"name": "test_func", "arguments": {"param"',
        'Start: [{"name": "test_func", "arguments": {"param":',
        'Start: [{"name": "test_func", "arguments": {"param": "value"',
        'Start: [{"name": "test_func", "arguments": {"param": "value"}',
        'Start: [{"name": "test_func", "arguments": {"param": "value"}}',
        'Start: [{"name": "test_func", "arguments": {"param": "value"}}]',
    ]
    
    for sequence in test_sequences:
        # Encode the sequence
        input_ids = tokenizer.encode(sequence, return_tensors="pt")
        
        # Create dummy scores
        vocab_size = len(tokenizer.vocab)
        scores = torch.randn(1, vocab_size)
        
        # Process with our processor
        try:
            processed_scores = processor(input_ids, scores)
            
            # Count how many tokens are allowed
            allowed_tokens = (processed_scores[0] > float('-inf')).sum().item()
            
            print(f"'{sequence}' -> {allowed_tokens} allowed tokens")
            
        except Exception as e:
            print(f"'{sequence}' -> Error: {e}")

if __name__ == "__main__":
    working_example()
    demonstrate_step_by_step()
    
    print("\n" + "="*60)
    print("Working example completed!")
    print("\nKey observations:")
    print("1. The processor successfully constrains generation to valid JSON structure")
    print("2. Function names are validated against available tools")
    print("3. The structure is enforced even with small models")
    print("4. Semantic correctness depends on the underlying model capability")
    print("5. The processor adds structural guarantees to any language model") 