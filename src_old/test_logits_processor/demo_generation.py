#!/usr/bin/env python3
"""
Demonstration script for StructuredFunctionCallProcessor with generation

This script shows how to use the processor in a real generation scenario.
"""

import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, LogitsProcessorList
from src.llm import StructuredFunctionCallProcessor, generate_structured_function_calls

# Sample tools for demonstration
DEMO_TOOLS = [
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
        "name": "search_web",
        "description": "Search the web for information",
        "parameters": {
            "query": {
                "type": "string",
                "description": "Search query"
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum number of results"
            }
        },
        "required": ["query"]
    },
    {
        "name": "send_message",
        "description": "Send a message to someone",
        "parameters": {
            "recipient": {
                "type": "string",
                "description": "Message recipient"
            },
            "message": {
                "type": "string",
                "description": "Message content"
            },
            "urgent": {
                "type": "boolean",
                "description": "Whether the message is urgent"
            }
        },
        "required": ["recipient", "message"]
    }
]

def create_sample(query: str) -> dict:
    """Create a sample with query and tools"""
    return {
        "query": query,
        "tools": json.dumps(DEMO_TOOLS)
    }

def demo_with_small_model():
    """Demonstrate with a small model (for testing)"""
    print("=== Demo with Small Model ===")
    
    # Use a small model for demonstration
    model_name = "gpt2"  # Small and widely available
    
    try:
        print(f"Loading {model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        print("✓ Model and tokenizer loaded")
        
        # Create processor
        processor = StructuredFunctionCallProcessor(
            tokenizer=tokenizer,
            tools=DEMO_TOOLS,
            device="cpu"
        )
        print("✓ Processor created")
        
        # Test queries
        test_queries = [
            "What's the weather in Paris?",
            "Search for information about machine learning",
            "Send a message to Alice saying hello"
        ]
        
        for query in test_queries:
            print(f"\nQuery: {query}")
            sample = create_sample(query)
            
            try:
                # Use the generate_structured_function_calls function
                result = generate_structured_function_calls(
                    sample=sample,
                    model_llm=model,
                    tokenizer_llm=tokenizer,
                    max_new_tokens=50,
                    temperature=0.1
                )
                print(f"Generated: {result}")
                
                # Try to parse as JSON to validate structure
                try:
                    parsed = json.loads(result)
                    print(f"✓ Valid JSON with {len(parsed)} function calls")
                    for i, call in enumerate(parsed):
                        print(f"  {i+1}. {call.get('name', 'unknown')}({call.get('arguments', {})})")
                except json.JSONDecodeError:
                    print("⚠ Generated text is not valid JSON (this is expected with small models)")
                    
            except Exception as e:
                print(f"✗ Error generating: {e}")
                
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        print("Note: This demo requires the 'gpt2' model to be available.")

def demo_manual_generation():
    """Demonstrate manual step-by-step generation"""
    print("\n=== Manual Step-by-Step Generation Demo ===")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        processor = StructuredFunctionCallProcessor(
            tokenizer=tokenizer,
            tools=DEMO_TOOLS,
            device="cpu"
        )
        
        # Simulate step-by-step generation
        prompt = "Generate function calls: "
        current_text = prompt
        
        print(f"Starting with: '{prompt}'")
        
        # Simulate generation steps
        generation_steps = [
            "[",
            '{"name": "get_weather", "arguments": {"location": "Paris"}}',
            "]"
        ]
        
        for step, next_token in enumerate(generation_steps):
            print(f"\nStep {step + 1}: Adding '{next_token}'")
            
            # Encode current text
            input_ids = tokenizer.encode(current_text, return_tensors="pt")
            
            # Create dummy scores
            vocab_size = len(tokenizer.vocab) if hasattr(tokenizer, 'vocab') else 50257
            scores = torch.randn(1, vocab_size)
            
            # Process with our processor
            processed_scores = processor(input_ids, scores)
            
            # Check if the next token would be allowed
            next_token_ids = tokenizer.encode(next_token, add_special_tokens=False)
            if next_token_ids:
                token_id = next_token_ids[0]
                if token_id < vocab_size:
                    original_score = scores[0, token_id].item()
                    processed_score = processed_scores[0, token_id].item()
                    
                    if processed_score > float('-inf'):
                        print(f"✓ Token '{next_token}' is ALLOWED (score: {original_score:.2f} -> {processed_score:.2f})")
                    else:
                        print(f"✗ Token '{next_token}' is BLOCKED (score: {original_score:.2f} -> {processed_score:.2f})")
            
            # Add to current text
            current_text += next_token
            print(f"Current text: '{current_text}'")
            
    except Exception as e:
        print(f"✗ Error in manual demo: {e}")

def demo_state_transitions():
    """Demonstrate how the processor handles state transitions"""
    print("\n=== State Transition Demo ===")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        processor = StructuredFunctionCallProcessor(
            tokenizer=tokenizer,
            tools=DEMO_TOOLS,
            device="cpu"
        )
        
        # Show how states transition during generation
        generation_sequence = [
            ("", "Before any JSON"),
            ("[", "After opening bracket"),
            ('[{', "After opening object"),
            ('[{"name"', "After name key"),
            ('[{"name":', "After name colon"),
            ('[{"name": "get_weather"', "After function name"),
            ('[{"name": "get_weather", "arguments"', "After arguments key"),
            ('[{"name": "get_weather", "arguments":', "After arguments colon"),
            ('[{"name": "get_weather", "arguments": {', "After arguments object start"),
            ('[{"name": "get_weather", "arguments": {"location"', "After parameter name"),
            ('[{"name": "get_weather", "arguments": {"location":', "After parameter colon"),
            ('[{"name": "get_weather", "arguments": {"location": "Paris"', "After parameter value"),
            ('[{"name": "get_weather", "arguments": {"location": "Paris"}', "After parameter object end"),
            ('[{"name": "get_weather", "arguments": {"location": "Paris"}}', "After function object end"),
            ('[{"name": "get_weather", "arguments": {"location": "Paris"}}]', "After array end"),
        ]
        
        for json_text, description in generation_sequence:
            print(f"\n{description}:")
            print(f"  JSON: '{json_text}'")
            
            # Parse state
            state = processor._parse_partial_json(json_text)
            expecting = state.get('expecting', 'unknown')
            print(f"  Expecting: {expecting}")
            
            # Get valid tokens
            valid_tokens = processor._get_valid_tokens_for_state(state)
            print(f"  Valid tokens: {len(valid_tokens)}")
            
            # Show some example valid tokens
            if valid_tokens:
                sample_tokens = list(valid_tokens)[:3]
                examples = []
                for token in sample_tokens:
                    try:
                        decoded = tokenizer.decode([token])
                        examples.append(f"'{decoded}'")
                    except:
                        examples.append(f"token_{token}")
                print(f"  Examples: {', '.join(examples)}")
                
    except Exception as e:
        print(f"✗ Error in state transition demo: {e}")

def run_all_demos():
    """Run all demonstration functions"""
    print("StructuredFunctionCallProcessor - Generation Demos")
    print("=" * 60)
    
    # Demo with small model (actual generation)
    demo_with_small_model()
    
    # Manual generation demo
    demo_manual_generation()
    
    # State transition demo
    demo_state_transitions()
    
    print("\n" + "=" * 60)
    print("Demos completed!")
    print("\nKey takeaways:")
    print("1. The processor enforces valid JSON structure during generation")
    print("2. It validates function names against available tools")
    print("3. It ensures proper argument structure")
    print("4. State transitions are handled correctly")
    print("\nFor production use:")
    print("- Use a larger, more capable language model")
    print("- Adjust max_new_tokens and temperature as needed")
    print("- Handle edge cases and error conditions")

if __name__ == "__main__":
    run_all_demos() 