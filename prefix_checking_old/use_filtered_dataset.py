#!/usr/bin/env python3
"""
Example of how to load and use the filtered dataset.
"""

from datasets import load_from_disk
import json
import random


def main():
    print("Loading filtered dataset...")
    
    # Method 1: Load from disk (Hugging Face format)
    filtered_dataset = load_from_disk("xlam-function-calling-60k-filtered")
    print(f"✅ Loaded {len(filtered_dataset)} samples from disk")
    
    # Method 2: Load from JSON (if you prefer)
    # with open("xlam-function-calling-60k-filtered.json", 'r') as f:
    #     filtered_data = json.load(f)
    # print(f"✅ Loaded {len(filtered_data)} samples from JSON")
    
    print("\n" + "="*80)
    print("DATASET USAGE EXAMPLES")
    print("="*80)
    
    # Example 1: Access a specific sample
    print("\n1. Accessing a specific sample:")
    sample = filtered_dataset[0]
    print(f"   Query: {sample['query']}")
    print(f"   Answer: {sample['answers'][:100]}...")
    
    # Example 2: Random sampling
    print("\n2. Random sampling:")
    random_indices = random.sample(range(len(filtered_dataset)), 3)
    for i, idx in enumerate(random_indices):
        sample = filtered_dataset[idx]
        print(f"   Sample {i+1}: {sample['query'][:60]}...")
    
    # Example 3: Iterate through dataset
    print("\n3. Iterating through dataset (first 5):")
    for i, sample in enumerate(filtered_dataset):
        if i >= 5:
            break
        tools = json.loads(sample['tools']) if isinstance(sample['tools'], str) else sample['tools']
        print(f"   {i+1}. Query has {len(tools)} available tools")
    
    # Example 4: Filter by criteria
    print("\n4. Filtering by criteria:")
    short_answers = [s for s in filtered_dataset if len(s['answers']) < 100]
    print(f"   Found {len(short_answers)} samples with answers < 100 characters")
    
    # Example 5: Convert to different format
    print("\n5. Converting to training format:")
    training_sample = filtered_dataset[0]
    training_format = {
        "instruction": training_sample['query'],
        "tools": json.loads(training_sample['tools']) if isinstance(training_sample['tools'], str) else training_sample['tools'],
        "output": training_sample['answers']
    }
    print(f"   Instruction: {training_format['instruction'][:60]}...")
    print(f"   Tools: {len(training_format['tools'])} available")
    print(f"   Output: {training_format['output'][:60]}...")
    
    # Example 6: Dataset statistics
    print("\n6. Dataset statistics:")
    answer_lengths = [len(s['answers']) for s in filtered_dataset]
    print(f"   Average answer length: {sum(answer_lengths)/len(answer_lengths):.1f} characters")
    print(f"   Min answer length: {min(answer_lengths)} characters")
    print(f"   Max answer length: {max(answer_lengths)} characters")
    
    # Example 7: Save a subset
    print("\n7. Creating and saving a subset:")
    subset_size = 1000
    subset = filtered_dataset.select(range(subset_size))
    print(f"   Created subset with {len(subset)} samples")
    # subset.save_to_disk("xlam-subset-1k")  # Uncomment to actually save
    
    print("\n" + "="*80)
    print("READY TO USE!")
    print("="*80)
    print("The filtered dataset is ready for:")
    print("✓ Training function-calling models")
    print("✓ Evaluating LLM performance")
    print("✓ Fine-tuning existing models")
    print("✓ Creating test sets")
    print("\nAll samples have been validated to ensure every prefix is valid JSON!")


if __name__ == "__main__":
    main() 