#!/usr/bin/env python3
"""
Verify the filtered dataset and analyze what was removed.
"""

import sys
import os

# Add parent directory to path to import from src
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

try:
    from src.llm import is_valid_output_prefix
except ImportError:
    try:
        from prefix_checking import is_valid_output_prefix
    except ImportError:
        # If running as __main__
        from . import is_valid_output_prefix

import json
from datasets import load_from_disk, load_dataset
from tqdm import tqdm
import random

def verify_sample(sample):
    """Verify that a sample has all valid prefixes."""
    try:
        if isinstance(sample['tools'], str):
            tools = json.loads(sample['tools'])
        else:
            tools = sample['tools']
    except:
        return False, "Failed to parse tools"
    
    answer = sample.get('answers', '')
    if not answer:
        return False, "Empty answer"
    
    # Check all prefixes
    for i in range(len(answer) + 1):
        prefix = answer[:i]
        try:
            if not is_valid_output_prefix(tools, prefix):
                return False, f"Invalid prefix at position {i}"
        except Exception as e:
            return False, f"Exception at position {i}: {str(e)}"
    
    return True, "All prefixes valid"


def analyze_removed_samples(removed_indices_path, original_dataset):
    """Analyze the samples that were removed."""
    with open(removed_indices_path, 'r') as f:
        removed_data = json.load(f)
    
    removed_indices = removed_data['removed_indices']
    
    print("\n" + "="*80)
    print("ANALYSIS OF REMOVED SAMPLES")
    print("="*80)
    
    # Sample some removed entries for analysis
    sample_size = min(10, len(removed_indices))
    sampled_indices = random.sample(removed_indices, sample_size)
    
    error_patterns = {}
    
    for idx in sampled_indices:
        sample = original_dataset[idx]
        answer = sample.get('answers', '')
        
        # Try to identify the error pattern
        if "[[" in answer and '", [' in answer:
            pattern = "nested_array_in_string_array"
        elif "[[" in answer:
            pattern = "nested_array"
        elif '{"' in answer and ": t" in answer or ": f" in answer:
            pattern = "boolean_without_quotes"
        elif len(answer) > 500:
            pattern = "very_long_answer"
        else:
            pattern = "other"
            
        error_patterns[pattern] = error_patterns.get(pattern, 0) + 1
        
        print(f"\nRemoved Sample (Index: {idx}):")
        print(f"Query: {sample['query'][:80]}..." if len(sample['query']) > 80 else f"Query: {sample['query']}")
        print(f"Answer preview: {answer[:100]}..." if len(answer) > 100 else f"Answer: {answer}")
        print(f"Pattern: {pattern}")
    
    print("\n" + "-"*40)
    print("Error Pattern Distribution (from sample):")
    for pattern, count in sorted(error_patterns.items(), key=lambda x: x[1], reverse=True):
        print(f"  {pattern}: {count}")


def main():
    print("Loading filtered dataset...")
    
    # Load the filtered dataset
    try:
        filtered_dataset = load_from_disk("xlam-function-calling-60k-filtered")
        print(f"✅ Filtered dataset loaded successfully! Samples: {len(filtered_dataset)}")
    except Exception as e:
        print(f"❌ Error loading filtered dataset: {e}")
        return
    
    # Load removed indices info
    try:
        with open("removed_sample_indices.json", 'r') as f:
            removed_info = json.load(f)
        print(f"✅ Removed indices info loaded. Total removed: {removed_info['total_removed']}")
    except Exception as e:
        print(f"❌ Error loading removed indices: {e}")
        return
    
    print("\n" + "="*80)
    print("VERIFICATION RESULTS")
    print("="*80)
    
    # Verify a sample of the filtered dataset
    print("\nVerifying random sample of filtered dataset...")
    sample_size = min(100, len(filtered_dataset))
    sampled_indices = random.sample(range(len(filtered_dataset)), sample_size)
    
    valid_count = 0
    invalid_count = 0
    
    for idx in tqdm(sampled_indices, desc="Verifying samples"):
        sample = filtered_dataset[idx]
        is_valid, message = verify_sample(sample)
        
        if is_valid:
            valid_count += 1
        else:
            invalid_count += 1
            print(f"\n❌ Invalid sample found at index {idx}: {message}")
    
    print(f"\nVerification complete:")
    print(f"  Valid samples: {valid_count}/{sample_size} ({valid_count/sample_size*100:.2f}%)")
    print(f"  Invalid samples: {invalid_count}/{sample_size}")
    
    if invalid_count == 0:
        print("\n✅ All verified samples are valid!")
    else:
        print(f"\n⚠️  Found {invalid_count} invalid samples in the filtered dataset!")
    
    # Load original dataset for comparison
    print("\nLoading original dataset for comparison...")
    try:
        original_dataset = load_dataset("Salesforce/xlam-function-calling-60k", split="train")
        analyze_removed_samples("removed_sample_indices.json", original_dataset)
    except Exception as e:
        print(f"Could not load original dataset for analysis: {e}")
    
    # Show statistics
    print("\n" + "="*80)
    print("DATASET STATISTICS")
    print("="*80)
    print(f"Original dataset size: 60,000")
    print(f"Filtered dataset size: {len(filtered_dataset)}")
    print(f"Samples removed: {removed_info['total_removed']}")
    print(f"Removal rate: {removed_info['removal_rate']*100:.2f}%")
    print(f"Retention rate: {(1-removed_info['removal_rate'])*100:.2f}%")
    
    # Show some examples from filtered dataset
    print("\n" + "="*80)
    print("EXAMPLES FROM FILTERED DATASET")
    print("="*80)
    
    for i in range(3):
        sample = filtered_dataset[i]
        print(f"\nExample {i+1}:")
        print(f"Query: {sample['query'][:100]}..." if len(sample['query']) > 100 else f"Query: {sample['query']}")
        print(f"Answer: {sample['answers'][:150]}..." if len(sample['answers']) > 150 else f"Answer: {sample['answers']}")


if __name__ == "__main__":
    main() 