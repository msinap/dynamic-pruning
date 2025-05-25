#!/usr/bin/env python3
"""
Test script that validates all prefixes in the entire Salesforce/xlam-function-calling-60k dataset.
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
        
        try:
            if is_valid_output_prefix(tools, prefix):
                valid += 1
            else:
                invalid += 1
                if first_invalid is None:
                    first_invalid = prefix
        except Exception as e:
            # Count exceptions as invalid
            invalid += 1
            if first_invalid is None:
                first_invalid = f"ERROR at position {i}: {str(e)}"
                
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
    
    # Track error types
    error_types = {}
    
    print(f"\nTesting ALL {len(dataset)} samples...")
    print("This may take several minutes...\n")
    
    start_time = time.time()
    
    # Process each sample with progress bar
    for idx, sample in enumerate(tqdm(dataset, desc="Processing samples")):
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
            
            # Categorize error type
            if first_invalid and isinstance(first_invalid, str):
                if "[[" in first_invalid and '", [' in first_invalid:
                    error_type = "nested_array_in_string_array"
                elif "[[" in first_invalid:
                    error_type = "nested_array"
                elif '{"' in first_invalid and '":' in first_invalid and not '": "' in first_invalid:
                    error_type = "object_value_without_quotes"
                elif ": t" in first_invalid or ": f" in first_invalid:
                    error_type = "boolean_value"
                else:
                    error_type = "other"
                    
                error_types[error_type] = error_types.get(error_type, 0) + 1
    
    elapsed_time = time.time() - start_time
    
    # Print results
    print("\n" + "="*80)
    print("FULL DATASET VALIDATION RESULTS")
    print("="*80)
    print(f"Total samples tested: {total_samples:,}")
    print(f"Total prefixes tested: {total_prefixes_tested:,}")
    print(f"Valid prefixes: {total_valid_prefixes:,} ({total_valid_prefixes/total_prefixes_tested*100:.2f}%)")
    print(f"Invalid prefixes: {total_invalid_prefixes:,} ({total_invalid_prefixes/total_prefixes_tested*100:.2f}%)")
    print(f"Samples with invalid prefixes: {samples_with_invalid_prefixes:,} ({samples_with_invalid_prefixes/total_samples*100:.2f}%)")
    print(f"Time elapsed: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
    print(f"Samples per second: {total_samples/elapsed_time:.0f}")
    print(f"Prefixes per second: {total_prefixes_tested/elapsed_time:.0f}")
    
    # Show error type breakdown
    if error_types:
        print("\n" + "="*80)
        print("ERROR TYPE BREAKDOWN")
        print("="*80)
        for error_type, count in sorted(error_types.items(), key=lambda x: x[1], reverse=True):
            print(f"{error_type}: {count} samples")
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    accuracy = total_valid_prefixes / total_prefixes_tested * 100
    if accuracy >= 99:
        print(f"✅ Excellent validation accuracy: {accuracy:.2f}%")
    elif accuracy >= 95:
        print(f"✓ Good validation accuracy: {accuracy:.2f}%")
    elif accuracy >= 90:
        print(f"⚠️  Acceptable validation accuracy: {accuracy:.2f}%")
    else:
        print(f"❌ Poor validation accuracy: {accuracy:.2f}%")
        
    print(f"\nThe validator correctly identified {total_valid_prefixes:,} valid prefixes")
    print(f"and {total_invalid_prefixes:,} invalid prefixes across {total_samples:,} samples.")


if __name__ == "__main__":
    main() 