#!/usr/bin/env python3
"""
Test script that validates all prefixes in the Salesforce/xlam-function-calling-60k dataset.
"""

import sys
import json
from datasets import load_dataset
from tqdm import tqdm
import time

sys.path.append('src')
from llm import is_valid_output_prefix


def test_all_prefixes_for_answer(tools, answer):
    """
    Test all prefixes of an answer string.
    Returns a tuple of (total_prefixes, valid_prefixes, invalid_prefixes, first_invalid_prefix)
    """
    total = 0
    valid = 0
    invalid = 0
    first_invalid = None
    
    # Test all prefixes from empty string to full answer
    for i in range(len(answer) + 1):
        prefix = answer[:i]
        total += 1
        
        if is_valid_output_prefix(tools, prefix):
            valid += 1
        else:
            invalid += 1
            if first_invalid is None:
                first_invalid = prefix
                
    return total, valid, invalid, first_invalid


def main():
    print("Loading Salesforce/xlam-function-calling-60k dataset...")
    
    # Load the dataset
    try:
        dataset = load_dataset("Salesforce/xlam-function-calling-60k", split="train")
        print(f"Dataset loaded successfully! Total samples: {len(dataset)}")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return
    
    # Statistics
    total_samples = 0
    samples_with_invalid_prefixes = 0
    total_prefixes_tested = 0
    total_valid_prefixes = 0
    total_invalid_prefixes = 0
    
    # Track problematic samples
    problematic_samples = []
    
    # Test a subset first (you can change this to test all)
    max_samples = min(5000, len(dataset))  # Test first 5000 samples
    print(f"\nTesting first {max_samples} samples...")
    
    start_time = time.time()
    
    # Process each sample
    for idx, sample in enumerate(tqdm(dataset.select(range(max_samples)), desc="Processing samples")):
        total_samples += 1
        
        # Parse tools
        try:
            if isinstance(sample['tools'], str):
                tools = json.loads(sample['tools'])
            else:
                tools = sample['tools']
        except Exception as e:
            print(f"\nError parsing tools for sample {idx}: {e}")
            continue
            
        # Get the answer
        answer = sample.get('answers', '')
        if not answer:
            continue
            
        # Test all prefixes
        total, valid, invalid, first_invalid = test_all_prefixes_for_answer(tools, answer)
        
        total_prefixes_tested += total
        total_valid_prefixes += valid
        total_invalid_prefixes += invalid
        
        if invalid > 0:
            samples_with_invalid_prefixes += 1
            problematic_samples.append({
                'index': idx,
                'query': sample.get('query', '')[:100] + '...' if len(sample.get('query', '')) > 100 else sample.get('query', ''),
                'answer': answer[:100] + '...' if len(answer) > 100 else answer,
                'first_invalid_prefix': first_invalid,
                'invalid_count': invalid,
                'total_prefixes': total
            })
    
    elapsed_time = time.time() - start_time
    
    # Print results
    print("\n" + "="*80)
    print("VALIDATION RESULTS")
    print("="*80)
    print(f"Total samples tested: {total_samples}")
    print(f"Total prefixes tested: {total_prefixes_tested:,}")
    print(f"Valid prefixes: {total_valid_prefixes:,} ({total_valid_prefixes/total_prefixes_tested*100:.2f}%)")
    print(f"Invalid prefixes: {total_invalid_prefixes:,} ({total_invalid_prefixes/total_prefixes_tested*100:.2f}%)")
    print(f"Samples with invalid prefixes: {samples_with_invalid_prefixes} ({samples_with_invalid_prefixes/total_samples*100:.2f}%)")
    print(f"Time elapsed: {elapsed_time:.2f} seconds")
    print(f"Prefixes per second: {total_prefixes_tested/elapsed_time:.0f}")
    
    # Show some problematic samples
    if problematic_samples:
        print("\n" + "="*80)
        print("PROBLEMATIC SAMPLES (first 10)")
        print("="*80)
        
        for i, sample in enumerate(problematic_samples[:10]):
            print(f"\nSample {i+1} (Index: {sample['index']}):")
            print(f"Query: {sample['query']}")
            print(f"Answer: {sample['answer']}")
            print(f"First invalid prefix: {repr(sample['first_invalid_prefix'])}")
            print(f"Invalid prefixes: {sample['invalid_count']} out of {sample['total_prefixes']}")
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    if samples_with_invalid_prefixes == 0:
        print("✅ All prefixes in all tested samples are valid!")
    else:
        print(f"⚠️  Found {samples_with_invalid_prefixes} samples with invalid prefixes")
        print(f"   This represents {samples_with_invalid_prefixes/total_samples*100:.2f}% of tested samples")
        
    # Option to test all samples
    if max_samples < len(dataset):
        print(f"\nNote: Only tested {max_samples} out of {len(dataset)} total samples.")
        print("To test all samples, modify max_samples in the script.")


if __name__ == "__main__":
    main() 