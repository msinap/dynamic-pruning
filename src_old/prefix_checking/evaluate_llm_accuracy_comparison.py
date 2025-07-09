#!/usr/bin/env python3
"""
Compare standard generation vs prefix-validated generation accuracy.
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


def count_invalid_prefixes(output, tools):
    """Count how many prefixes in the output are invalid."""
    invalid_count = 0
    for i in range(len(output) + 1):
        prefix = output[:i]
        if not is_valid_output_prefix(tools, prefix):
            invalid_count += 1
    return invalid_count


def main():
    print("Comparison: Standard vs Prefix-Validated Generation")
    print("=" * 80)
    
    # Configuration
    CONFIG = {
        "llm_model_name": "Salesforce/xLAM-2-1b-fc-r",
        "num_samples": 50,  # Number of samples to evaluate
        "max_new_tokens": 200,
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
    
    # Metrics for both methods
    metrics = {
        "standard": {
            "exact_match": 0,
            "function_match": 0,
            "correct_functions": 0,
            "valid_json": 0,
            "all_prefixes_valid": 0,
            "total_invalid_prefixes": 0,
            "generation_errors": 0,
            "parse_errors": 0,
            "generation_time": 0,
        },
        "prefix_validated": {
            "exact_match": 0,
            "function_match": 0,
            "correct_functions": 0,
            "valid_json": 0,
            "all_prefixes_valid": 0,
            "total_invalid_prefixes": 0,
            "generation_errors": 0,
            "parse_errors": 0,
            "generation_time": 0,
        }
    }
    
    # Evaluate each sample with both methods
    for idx in tqdm(sample_indices, desc="Evaluating samples"):
        sample = dataset[idx]
        
        # Parse tools and expected output
        try:
            tools = json.loads(sample['tools']) if isinstance(sample['tools'], str) else sample['tools']
            expected_output = sample['answers']
        except Exception as e:
            metrics["standard"]["parse_errors"] += 1
            metrics["prefix_validated"]["parse_errors"] += 1
            continue
        
        # 1. Standard Generation
        try:
            start_time = time.time()
            standard_output = generate_llm_output(
                sample, 
                model, 
                tokenizer, 
                max_new_tokens=CONFIG["max_new_tokens"]
            )
            metrics["standard"]["generation_time"] += time.time() - start_time
            
            # Evaluate standard output
            if evaluate_exact_match(standard_output, expected_output):
                metrics["standard"]["exact_match"] += 1
            
            if evaluate_function_match(standard_output, expected_output):
                metrics["standard"]["function_match"] += 1
            
            if evaluate_correct_functions(standard_output, expected_output):
                metrics["standard"]["correct_functions"] += 1
            
            if evaluate_valid_json(standard_output):
                metrics["standard"]["valid_json"] += 1
            
            # Check prefix validity
            invalid_count = count_invalid_prefixes(standard_output, tools)
            metrics["standard"]["total_invalid_prefixes"] += invalid_count
            if invalid_count == 0:
                metrics["standard"]["all_prefixes_valid"] += 1
                
        except Exception as e:
            metrics["standard"]["generation_errors"] += 1
        
        # 2. Prefix-Validated Generation
        try:
            # Tokenize input
            inputs = tokenize_for_llm(sample, tokenizer, model.device)
            
            start_time = time.time()
            validated_output = generate_with_prefix_validation(
                model=model,
                tokenizer=tokenizer,
                tools=tools,
                input_ids=inputs["input_ids"],
                max_new_tokens=CONFIG["max_new_tokens"],
                temperature=0.1
            )
            metrics["prefix_validated"]["generation_time"] += time.time() - start_time
            
            # Evaluate validated output
            if evaluate_exact_match(validated_output, expected_output):
                metrics["prefix_validated"]["exact_match"] += 1
            
            if evaluate_function_match(validated_output, expected_output):
                metrics["prefix_validated"]["function_match"] += 1
            
            if evaluate_correct_functions(validated_output, expected_output):
                metrics["prefix_validated"]["correct_functions"] += 1
            
            if evaluate_valid_json(validated_output):
                metrics["prefix_validated"]["valid_json"] += 1
            
            # Check prefix validity (should always be 0 for prefix-validated)
            invalid_count = count_invalid_prefixes(validated_output, tools)
            metrics["prefix_validated"]["total_invalid_prefixes"] += invalid_count
            if invalid_count == 0:
                metrics["prefix_validated"]["all_prefixes_valid"] += 1
                
        except Exception as e:
            metrics["prefix_validated"]["generation_errors"] += 1
    
    # Print results
    print("\n" + "="*80)
    print("EVALUATION RESULTS")
    print("="*80)
    
    print(f"\nModel: {CONFIG['llm_model_name']}")
    print(f"Dataset: Filtered XLAM Function Calling")
    print(f"Samples evaluated: {num_samples}")
    
    # Print metrics for each method
    for method_name, method_metrics in metrics.items():
        successful_samples = num_samples - method_metrics["generation_errors"] - method_metrics["parse_errors"]
        avg_time = method_metrics["generation_time"] / successful_samples if successful_samples > 0 else 0
        
        print(f"\n{method_name.upper().replace('_', ' ')} GENERATION:")
        print("-" * 40)
        
        if successful_samples > 0:
            print(f"Exact Match: {method_metrics['exact_match']}/{successful_samples} ({method_metrics['exact_match']/successful_samples*100:.2f}%)")
            print(f"Function Match: {method_metrics['function_match']}/{successful_samples} ({method_metrics['function_match']/successful_samples*100:.2f}%)")
            print(f"Correct Functions: {method_metrics['correct_functions']}/{successful_samples} ({method_metrics['correct_functions']/successful_samples*100:.2f}%)")
            print(f"Valid JSON: {method_metrics['valid_json']}/{successful_samples} ({method_metrics['valid_json']/successful_samples*100:.2f}%)")
            print(f"All Prefixes Valid: {method_metrics['all_prefixes_valid']}/{successful_samples} ({method_metrics['all_prefixes_valid']/successful_samples*100:.2f}%)")
            
            if method_metrics['total_invalid_prefixes'] > 0:
                print(f"Total Invalid Prefixes: {method_metrics['total_invalid_prefixes']}")
                print(f"Avg Invalid Prefixes/Sample: {method_metrics['total_invalid_prefixes']/successful_samples:.2f}")
        
        print(f"Avg Generation Time: {avg_time:.3f}s per sample")
        print(f"Generation Errors: {method_metrics['generation_errors']}")
        print(f"Parse Errors: {method_metrics['parse_errors']}")
    
    # Comparison
    print("\n" + "="*80)
    print("COMPARISON")
    print("="*80)
    
    standard_successful = num_samples - metrics["standard"]["generation_errors"] - metrics["standard"]["parse_errors"]
    validated_successful = num_samples - metrics["prefix_validated"]["generation_errors"] - metrics["prefix_validated"]["parse_errors"]
    
    if standard_successful > 0 and validated_successful > 0:
        print("\nPrefix-Validated vs Standard Generation:")
        
        # Accuracy comparison
        for metric in ["exact_match", "function_match", "correct_functions", "valid_json"]:
            standard_rate = metrics["standard"][metric] / standard_successful * 100
            validated_rate = metrics["prefix_validated"][metric] / validated_successful * 100
            diff = validated_rate - standard_rate
            print(f"{metric.replace('_', ' ').title()}: {validated_rate:.1f}% vs {standard_rate:.1f}% ({diff:+.1f}%)")
        
        # Prefix validity
        print(f"\nPrefix Validity:")
        standard_valid = metrics["standard"]["all_prefixes_valid"] / standard_successful * 100
        validated_valid = metrics["prefix_validated"]["all_prefixes_valid"] / validated_successful * 100
        print(f"Standard: {standard_valid:.1f}% samples with all valid prefixes")
        print(f"Prefix-Validated: {validated_valid:.1f}% samples with all valid prefixes")
        print(f"Invalid prefixes eliminated: {metrics['standard']['total_invalid_prefixes']} → {metrics['prefix_validated']['total_invalid_prefixes']}")
        
        # Time comparison
        standard_time = metrics["standard"]["generation_time"] / standard_successful
        validated_time = metrics["prefix_validated"]["generation_time"] / validated_successful
        time_overhead = validated_time - standard_time
        print(f"\nGeneration Time:")
        print(f"Standard: {standard_time:.3f}s per sample")
        print(f"Prefix-Validated: {validated_time:.3f}s per sample")
        print(f"Overhead: {time_overhead:.3f}s ({time_overhead/standard_time*100:+.1f}%)")


if __name__ == "__main__":
    main() 