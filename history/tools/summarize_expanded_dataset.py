#!/usr/bin/env python3
"""
Create a summary of the expanded dataset
"""
import json

def summarize_expanded_dataset():
    """Create a summary of the expanded dataset."""
    
    with open("training/datasets/expanded_bytelogic_dataset.json", 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print("EXPANDED DATASET SUMMARY")
    print("="*50)
    print(f"Dataset Version: {data['metadata']['version']}")
    print(f"Total Examples: {data['metadata']['total_examples']}")
    print(f"Training Examples: {data['metadata']['train_examples']}")
    print(f"Validation Examples: {data['metadata']['val_examples']}")
    print(f"Test Examples: {data['metadata']['test_examples']}")
    print(f"Features: {', '.join(data['metadata']['features'])}")
    print(f"Compatibility: {data['metadata']['compatibility']}")
    print()
    
    # Analyze categories in the expanded dataset
    train_examples = data.get('train', [])
    categories = {}
    difficulties = {}
    
    for ex in train_examples:
        cat = ex['metadata']['category']
        diff = ex['metadata']['difficulty']
        categories[cat] = categories.get(cat, 0) + 1
        difficulties[diff] = difficulties.get(diff, 0) + 1
    
    print("Training Examples by Category:")
    for cat, count in sorted(categories.items()):
        print(f"  - {cat}: {count}")
    print()
    
    print("Training Examples by Difficulty:")  
    for diff, count in sorted(difficulties.items()):
        print(f"  - {diff}: {count}")
    print()
    
    print("USAGE RECOMMENDATION:")
    print(f"You can now train with the expanded dataset:")
    print(f"  --dataset training/datasets/expanded_bytelogic_dataset.json")
    print(f"This gives you {data['metadata']['train_examples']} training examples")
    print(f"({data['metadata']['train_examples'] - 880} more than the original dataset)")

if __name__ == "__main__":
    summarize_expanded_dataset()