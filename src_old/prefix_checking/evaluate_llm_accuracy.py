#!/usr/bin/env python3
"""
Evaluate LLM accuracy on the filtered dataset.
"""

import sys
import os

# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from src.llm import (
    load_llm,
    tokenize_for_llm,
    generate_llm_output,
    generate_with_prefix_validation,
    is_valid_output_prefix
)
from datasets import load_from_disk
import json
import torch
from tqdm import tqdm
import time
import random
from collections import defaultdict


def evaluate_exact_match(generated, expected):
    """Check if generated output exactly matches expected."""
    try:
        gen_parsed = json.loads(generated.strip())
        exp_parsed = json.loads(expected.strip())
        return gen_parsed == exp_parsed
    except:
        return False


def evaluate_function_match(generated, expected):
    """Check if generated functions match expected (ignoring argument values)."""
    try:
        gen_parsed = json.loads(generated.strip())
        exp_parsed = json.loads(expected.strip())
        
        if len(gen_parsed) != len(exp_parsed):
            return False
            
        for gen_call, exp_call in zip(gen_parsed, exp_parsed):
            if gen_call.get('name') != exp_call.get('name'):
                return False
                
        return True
    except:
        return False


def evaluate_valid_json(output):
    """Check if output is valid JSON."""
    try:
        json.loads(output.strip())
        return True
    except:
        return False


def evaluate_correct_functions(generated, expected):
    """Check if all expected functions are called (order doesn't matter)."""
    try:
        gen_parsed = json.loads(generated.strip())
        exp_parsed = json.loads(expected.strip())
        
        gen_functions = sorted([call.get('name', '') for call in gen_parsed])
        exp_functions = sorted([call.get('name', '') for call in exp_parsed])
        
        return gen_functions == exp_functions
    except:
        return False


def main():
    print("LLM Accuracy Evaluation on Filtered Dataset")
    print("=" * 80)
    
    # Configuration
    CONFIG = {
        "llm_model_name": "Salesforce/xLAM-2-1b-fc-r",  # Updated to correct model
        "num_samples": 50,  # Number of samples to evaluate - reduced for faster evaluation
        "max_new_tokens": 200,
        "batch_size": 1,  # Batch size for generation
    }
    
    # Load model
    print(f"\nLoading model: {CONFIG['llm_model_name']}...")
    try:
        model, tokenizer = load_llm(CONFIG)
        print("✅ Model loaded successfully!")
        print(f"Device: {model.device}")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Load filtered dataset
    print("\nLoading filtered dataset...")
    try:
        dataset = load_from_disk("/workspace/xlam-function-calling-60k-filtered")
        print(f"✅ Dataset loaded successfully! Total samples: {len(dataset)}")
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return
    
    # Sample random subset for evaluation
    num_samples = min(CONFIG["num_samples"], len(dataset))
    sample_indices = random.sample(range(len(dataset)), num_samples)
    
    print(f"\nEvaluating on {num_samples} random samples...")
    print(f"Max new tokens: {CONFIG['max_new_tokens']}")
    
    # Metrics
    metrics = {
        "exact_match": 0,
        "function_match": 0,
        "correct_functions": 0,
        "valid_json": 0,
        "all_prefixes_valid": 0,
        "total_invalid_prefixes": 0,
        "generation_errors": 0,
        "parse_errors": 0,
    }
    
    # Track detailed results
    results_by_num_functions = defaultdict(lambda: {
        "total": 0,
        "exact_match": 0,
        "function_match": 0,
        "correct_functions": 0,
        "valid_json": 0
    })
    
    # Start evaluation
    start_time = time.time()
    
    for idx in tqdm(sample_indices, desc="Evaluating samples"):
        sample = dataset[idx]
        
        # Parse tools and expected output
        try:
            tools = json.loads(sample['tools']) if isinstance(sample['tools'], str) else sample['tools']
            expected_output = sample['answers']
            
            # Count number of expected function calls
            expected_parsed = json.loads(expected_output)
            num_expected_functions = len(expected_parsed)
            results_by_num_functions[num_expected_functions]["total"] += 1
            
        except Exception as e:
            metrics["parse_errors"] += 1
            continue
        
        # Generate output
        try:
            generated_output = generate_llm_output(
                sample, 
                model, 
                tokenizer, 
                max_new_tokens=CONFIG["max_new_tokens"]
            )
            
            # Evaluate metrics
            if evaluate_exact_match(generated_output, expected_output):
                metrics["exact_match"] += 1
                results_by_num_functions[num_expected_functions]["exact_match"] += 1
            
            if evaluate_function_match(generated_output, expected_output):
                metrics["function_match"] += 1
                results_by_num_functions[num_expected_functions]["function_match"] += 1
            
            if evaluate_correct_functions(generated_output, expected_output):
                metrics["correct_functions"] += 1
                results_by_num_functions[num_expected_functions]["correct_functions"] += 1
            
            if evaluate_valid_json(generated_output):
                metrics["valid_json"] += 1
                results_by_num_functions[num_expected_functions]["valid_json"] += 1
            
            # Check prefix validity
            invalid_count = 0
            for i in range(len(generated_output) + 1):
                prefix = generated_output[:i]
                if not is_valid_output_prefix(tools, prefix):
                    invalid_count += 1
            
            metrics["total_invalid_prefixes"] += invalid_count
            if invalid_count == 0:
                metrics["all_prefixes_valid"] += 1
                
        except Exception as e:
            metrics["generation_errors"] += 1
            print(f"\nGeneration error at sample {idx}: {e}")
    
    # Calculate total time
    total_time = time.time() - start_time
    successful_samples = num_samples - metrics["generation_errors"] - metrics["parse_errors"]
    
    # Print results
    print("\n" + "="*80)
    print("EVALUATION RESULTS")
    print("="*80)
    
    print(f"\nModel: {CONFIG['llm_model_name']}")
    print(f"Dataset: Filtered XLAM Function Calling")
    print(f"Samples evaluated: {num_samples}")
    print(f"Successful generations: {successful_samples}")
    print(f"Total time: {total_time:.2f}s ({total_time/num_samples:.2f}s per sample)")
    
    print("\n" + "-"*40)
    print("ACCURACY METRICS")
    print("-"*40)
    
    if successful_samples > 0:
        print(f"Exact Match: {metrics['exact_match']}/{successful_samples} ({metrics['exact_match']/successful_samples*100:.2f}%)")
        print(f"Function Match (same functions, same order): {metrics['function_match']}/{successful_samples} ({metrics['function_match']/successful_samples*100:.2f}%)")
        print(f"Correct Functions (any order): {metrics['correct_functions']}/{successful_samples} ({metrics['correct_functions']/successful_samples*100:.2f}%)")
        print(f"Valid JSON: {metrics['valid_json']}/{successful_samples} ({metrics['valid_json']/successful_samples*100:.2f}%)")
        print(f"All Prefixes Valid: {metrics['all_prefixes_valid']}/{successful_samples} ({metrics['all_prefixes_valid']/successful_samples*100:.2f}%)")
        
        if metrics['total_invalid_prefixes'] > 0:
            print(f"\nTotal Invalid Prefixes: {metrics['total_invalid_prefixes']}")
            print(f"Average Invalid Prefixes per Sample: {metrics['total_invalid_prefixes']/successful_samples:.2f}")
    
    print(f"\nGeneration Errors: {metrics['generation_errors']}")
    print(f"Parse Errors: {metrics['parse_errors']}")
    
    # Results by number of functions
    print("\n" + "-"*40)
    print("ACCURACY BY NUMBER OF FUNCTION CALLS")
    print("-"*40)
    
    for num_functions in sorted(results_by_num_functions.keys()):
        data = results_by_num_functions[num_functions]
        if data["total"] > 0:
            print(f"\n{num_functions} function call(s) ({data['total']} samples):")
            print(f"  Exact Match: {data['exact_match']}/{data['total']} ({data['exact_match']/data['total']*100:.1f}%)")
            print(f"  Function Match: {data['function_match']}/{data['total']} ({data['function_match']/data['total']*100:.1f}%)")
            print(f"  Correct Functions: {data['correct_functions']}/{data['total']} ({data['correct_functions']/data['total']*100:.1f}%)")
            print(f"  Valid JSON: {data['valid_json']}/{data['total']} ({data['valid_json']/data['total']*100:.1f}%)")
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    if successful_samples > 0:
        print(f"\n✅ Overall Exact Match Accuracy: {metrics['exact_match']/successful_samples*100:.2f}%")
        print(f"✅ Valid JSON Rate: {metrics['valid_json']/successful_samples*100:.2f}%")
        print(f"✅ Correct Functions Rate: {metrics['correct_functions']/successful_samples*100:.2f}%")
        
        if metrics['all_prefixes_valid'] < successful_samples:
            print(f"\n⚠️  Samples with invalid prefixes: {successful_samples - metrics['all_prefixes_valid']} ({(successful_samples - metrics['all_prefixes_valid'])/successful_samples*100:.1f}%)")
            print("   Consider using prefix-validated generation for 100% valid outputs.")
    
    print(f"\n📊 Generation speed: {successful_samples/total_time:.1f} samples/second")


if __name__ == "__main__":
    main() 