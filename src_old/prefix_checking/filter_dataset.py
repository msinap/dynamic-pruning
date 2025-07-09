#!/usr/bin/env python3
"""
Filter out samples with invalid prefixes from the Salesforce/xlam-function-calling-60k dataset.
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
from datasets import load_dataset, Dataset
from tqdm import tqdm
import time

def has_all_valid_prefixes(tools, answer):
    """
    Check if all prefixes of an answer are valid.
    Returns True if all prefixes are valid, False otherwise.
    """
    # Test all prefixes from empty string to full answer
    for i in range(len(answer) + 1):
        prefix = answer[:i]
        try:
            if not is_valid_output_prefix(tools, prefix):
                return False
        except Exception:
            # Count exceptions as invalid
            return False
    return True


def main():
    print("Loading Salesforce/xlam-function-calling-60k dataset...")
    
    # Load the dataset
    try:
        dataset = load_dataset("Salesforce/xlam-function-calling-60k", split="train")
        print(f"Dataset loaded successfully! Total samples: {len(dataset)}")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return
    
    # Track statistics
    total_samples = len(dataset)
    valid_samples = []
    invalid_samples = []
    parse_errors = 0
    
    print(f"\nFiltering dataset...")
    print("This may take several minutes...\n")
    
    start_time = time.time()
    
    # Process each sample
    for idx, sample in enumerate(tqdm(dataset, desc="Processing samples")):
        # Parse tools
        try:
            if isinstance(sample['tools'], str):
                tools = json.loads(sample['tools'])
            else:
                tools = sample['tools']
        except Exception as e:
            print(f"\nError parsing tools for sample {idx}: {e}")
            parse_errors += 1
            invalid_samples.append(idx)
            continue
            
        # Get the answer
        answer = sample.get('answers', '')
        if not answer:
            invalid_samples.append(idx)
            continue
            
        # Check if all prefixes are valid
        if has_all_valid_prefixes(tools, answer):
            valid_samples.append(sample)
        else:
            invalid_samples.append(idx)
    
    elapsed_time = time.time() - start_time
    
    # Create filtered dataset
    filtered_dataset = Dataset.from_list(valid_samples)
    
    # Print results
    print("\n" + "="*80)
    print("FILTERING RESULTS")
    print("="*80)
    print(f"Total samples processed: {total_samples:,}")
    print(f"Valid samples (kept): {len(valid_samples):,} ({len(valid_samples)/total_samples*100:.2f}%)")
    print(f"Invalid samples (removed): {len(invalid_samples):,} ({len(invalid_samples)/total_samples*100:.2f}%)")
    print(f"Parse errors: {parse_errors}")
    print(f"Time elapsed: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
    print(f"Samples per second: {total_samples/elapsed_time:.0f}")
    
    # Save the filtered dataset
    print("\n" + "="*80)
    print("SAVING FILTERED DATASET")
    print("="*80)
    
    # Save to disk
    output_path = "xlam-function-calling-60k-filtered"
    print(f"Saving filtered dataset to '{output_path}'...")
    filtered_dataset.save_to_disk(output_path)
    print(f"✅ Filtered dataset saved successfully!")
    
    # Also save as JSON for easy inspection
    json_output_path = "xlam-function-calling-60k-filtered.json"
    print(f"\nSaving filtered dataset as JSON to '{json_output_path}'...")
    with open(json_output_path, 'w') as f:
        json.dump(valid_samples, f, indent=2)
    print(f"✅ JSON file saved successfully!")
    
    # Save list of removed sample indices
    removed_indices_path = "removed_sample_indices.json"
    print(f"\nSaving list of removed sample indices to '{removed_indices_path}'...")
    with open(removed_indices_path, 'w') as f:
        json.dump({
            "removed_indices": invalid_samples,
            "total_removed": len(invalid_samples),
            "parse_errors": parse_errors,
            "removal_rate": len(invalid_samples) / total_samples
        }, f, indent=2)
    print(f"✅ Removed indices saved successfully!")
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"✅ Successfully filtered dataset from {total_samples:,} to {len(valid_samples):,} samples")
    print(f"📊 Retention rate: {len(valid_samples)/total_samples*100:.2f}%")
    print(f"📁 Filtered dataset saved to: {output_path}")
    print(f"📄 JSON version saved to: {json_output_path}")
    print(f"📋 Removed indices saved to: {removed_indices_path}")
    
    # Show some examples of filtered samples
    if len(valid_samples) > 0:
        print("\n" + "="*80)
        print("SAMPLE OF FILTERED DATA (first 3)")
        print("="*80)
        
        for i, sample in enumerate(valid_samples[:3]):
            print(f"\nSample {i+1}:")
            print(f"Query: {sample['query'][:100]}..." if len(sample['query']) > 100 else f"Query: {sample['query']}")
            print(f"Answer: {sample['answers'][:100]}..." if len(sample['answers']) > 100 else f"Answer: {sample['answers']}")


if __name__ == "__main__":
    main() 