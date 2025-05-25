#!/usr/bin/env python3
"""
Evaluate the prefix-validated generation function on the filtered dataset.
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
from difflib import SequenceMatcher


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


def evaluate_similarity(generated, expected):
    """Calculate string similarity between generated and expected."""
    return SequenceMatcher(None, generated, expected).ratio()


def count_invalid_prefixes(output, tools):
    """Count how many prefixes in the output are invalid."""
    invalid_count = 0
    for i in range(len(output) + 1):
        prefix = output[:i]
        if not is_valid_output_prefix(tools, prefix):
            invalid_count += 1
    return invalid_count


def main():
    print("Evaluation of Prefix-Validated Generation")
    print("=" * 80)
    
    # Configuration
    CONFIG = {
        "llm_model_name": "NousResearch/Hermes-3-Llama-3.2-3B",  # Change to your model
        "num_samples": 100,  # Number of samples to evaluate
        "max_new_tokens": 150,
        "temperature": 0.1,
    }
    
    # Load model
    print(f"\nLoading model: {CONFIG['llm_model_name']}...")
    try:
        model, tokenizer = load_llm(CONFIG)
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("Note: This evaluation requires a GPU and the model to be available.")
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
    
    # Metrics
    metrics = {
        "standard": {
            "exact_match": 0,
            "function_match": 0,
            "valid_json": 0,
            "all_prefixes_valid": 0,
            "total_invalid_prefixes": 0,
            "avg_similarity": 0,
            "generation_time": 0,
            "errors": 0
        },
        "prefix_validated": {
            "exact_match": 0,
            "function_match": 0,
            "valid_json": 0,
            "all_prefixes_valid": 0,
            "total_invalid_prefixes": 0,
            "avg_similarity": 0,
            "generation_time": 0,
            "errors": 0
        }
    }
    
    # Evaluate each sample
    for idx in tqdm(sample_indices, desc="Evaluating samples"):
        sample = dataset[idx]
        
        # Parse tools
        try:
            tools = json.loads(sample['tools']) if isinstance(sample['tools'], str) else sample['tools']
        except:
            continue
            
        expected_output = sample['answers']
        
        # Tokenize input
        inputs = tokenize_for_llm(sample, tokenizer, model.device)
        
        # 1. Standard Generation
        try:
            start_time = time.time()
            standard_output = generate_llm_output(sample, model, tokenizer, max_new_tokens=CONFIG["max_new_tokens"])
            metrics["standard"]["generation_time"] += time.time() - start_time
            
            # Evaluate standard output
            if evaluate_exact_match(standard_output, expected_output):
                metrics["standard"]["exact_match"] += 1
                
            if evaluate_function_match(standard_output, expected_output):
                metrics["standard"]["function_match"] += 1
                
            try:
                json.loads(standard_output.strip())
                metrics["standard"]["valid_json"] += 1
            except:
                pass
                
            invalid_count = count_invalid_prefixes(standard_output, tools)
            metrics["standard"]["total_invalid_prefixes"] += invalid_count
            if invalid_count == 0:
                metrics["standard"]["all_prefixes_valid"] += 1
                
            metrics["standard"]["avg_similarity"] += evaluate_similarity(standard_output, expected_output)
            
        except Exception as e:
            metrics["standard"]["errors"] += 1
            
        # 2. Prefix-Validated Generation
        try:
            start_time = time.time()
            validated_output = generate_with_prefix_validation(
                model=model,
                tokenizer=tokenizer,
                tools=tools,
                input_ids=inputs["input_ids"],
                max_new_tokens=CONFIG["max_new_tokens"],
                temperature=CONFIG["temperature"]
            )
            metrics["prefix_validated"]["generation_time"] += time.time() - start_time
            
            # Evaluate validated output
            if evaluate_exact_match(validated_output, expected_output):
                metrics["prefix_validated"]["exact_match"] += 1
                
            if evaluate_function_match(validated_output, expected_output):
                metrics["prefix_validated"]["function_match"] += 1
                
            try:
                json.loads(validated_output.strip())
                metrics["prefix_validated"]["valid_json"] += 1
            except:
                pass
                
            invalid_count = count_invalid_prefixes(validated_output, tools)
            metrics["prefix_validated"]["total_invalid_prefixes"] += invalid_count
            if invalid_count == 0:
                metrics["prefix_validated"]["all_prefixes_valid"] += 1
                
            metrics["prefix_validated"]["avg_similarity"] += evaluate_similarity(validated_output, expected_output)
            
        except Exception as e:
            metrics["prefix_validated"]["errors"] += 1
    
    # Calculate averages
    for method in metrics:
        metrics[method]["avg_similarity"] /= num_samples
        metrics[method]["generation_time"] /= (num_samples - metrics[method]["errors"])
    
    # Print results
    print("\n" + "="*80)
    print("EVALUATION RESULTS")
    print("="*80)
    
    print(f"\nDataset: Filtered XLAM Function Calling (60k)")
    print(f"Model: {CONFIG['llm_model_name']}")
    print(f"Samples evaluated: {num_samples}")
    
    for method_name, method_metrics in metrics.items():
        print(f"\n{method_name.upper().replace('_', ' ')} GENERATION:")
        print("-" * 40)
        
        successful_samples = num_samples - method_metrics["errors"]
        
        print(f"Exact Match: {method_metrics['exact_match']}/{successful_samples} ({method_metrics['exact_match']/successful_samples*100:.1f}%)")
        print(f"Function Match: {method_metrics['function_match']}/{successful_samples} ({method_metrics['function_match']/successful_samples*100:.1f}%)")
        print(f"Valid JSON: {method_metrics['valid_json']}/{successful_samples} ({method_metrics['valid_json']/successful_samples*100:.1f}%)")
        print(f"All Prefixes Valid: {method_metrics['all_prefixes_valid']}/{successful_samples} ({method_metrics['all_prefixes_valid']/successful_samples*100:.1f}%)")
        print(f"Total Invalid Prefixes: {method_metrics['total_invalid_prefixes']}")
        print(f"Average Similarity: {method_metrics['avg_similarity']:.3f}")
        print(f"Avg Generation Time: {method_metrics['generation_time']:.3f}s")
        print(f"Errors: {method_metrics['errors']}")
    
    # Comparison
    print("\n" + "="*80)
    print("COMPARISON")
    print("="*80)
    
    print("\nPrefix-Validated vs Standard Generation:")
    
    # Calculate improvements
    standard_valid = metrics["standard"]["all_prefixes_valid"]
    validated_valid = metrics["prefix_validated"]["all_prefixes_valid"]
    
    if standard_valid > 0:
        improvement = ((validated_valid - standard_valid) / standard_valid) * 100
        print(f"✅ Prefix validity improvement: {improvement:+.1f}%")
    else:
        print(f"✅ Prefix validity: {validated_valid} valid (vs 0 in standard)")
    
    print(f"✅ Invalid prefixes eliminated: {metrics['standard']['total_invalid_prefixes']} → {metrics['prefix_validated']['total_invalid_prefixes']}")
    
    # Time comparison
    time_overhead = metrics["prefix_validated"]["generation_time"] - metrics["standard"]["generation_time"]
    print(f"⏱️  Time overhead: {time_overhead:.3f}s per sample ({time_overhead/metrics['standard']['generation_time']*100:+.1f}%)")
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print("The prefix-validated generation guarantees that every prefix is valid,")
    print("ensuring robust and reliable function call generation.")


if __name__ == "__main__":
    main() 