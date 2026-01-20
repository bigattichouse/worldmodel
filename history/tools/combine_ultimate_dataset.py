#!/usr/bin/env python3
"""
Combine all datasets together into one ultimate comprehensive dataset
"""

import json
from pathlib import Path

def load_json_file(file_path):
    """Load examples from JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        return data.get('train', []) if 'train' in data else []

def load_jsonl_file(file_path):
    """Load examples from JSONL file."""
    examples = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    example = json.loads(line)
                    if isinstance(example, dict) and 'input' in example and 'output' in example:
                        examples.append(example)
                except json.JSONDecodeError:
                    continue  # Skip invalid lines
    return examples

def combine_all_datasets():
    """Combine all available datasets into one comprehensive file."""
    datasets_dir = Path("training/datasets")
    all_examples = []
    
    print("Loading all datasets...")
    
    # JSON files
    for json_file in datasets_dir.glob("*.json"):
        if json_file.name.startswith(".") or json_file.name == "large_scale_bytelogic_dataset.json":
            continue  # Skip hidden files and the output file to avoid duplication
            
        print(f"Loading {json_file.name}...")
        try:
            examples = load_json_file(json_file)
            all_examples.extend(examples)
            print(f"  - Added {len(examples)} examples")
        except Exception as e:
            print(f"  - Error loading {json_file.name}: {e}")
    
    # JSONL files
    for jsonl_file in datasets_dir.glob("*.jsonl"):
        if jsonl_file.name.startswith("."):
            continue  # Skip hidden files
            
        print(f"Loading {jsonl_file.name}...")
        try:
            examples = load_jsonl_file(jsonl_file)
            all_examples.extend(examples)
            print(f"  - Added {len(examples)} examples")
        except Exception as e:
            print(f"  - Error loading {jsonl_file.name}: {e}")
    
    print(f"Total examples loaded: {len(all_examples)}")
    return all_examples

def save_ultimate_dataset(examples):
    """Save all examples into the ultimate comprehensive dataset."""
    # Categorize examples
    categories = {}
    for ex in examples:
        cat = ex.get('metadata', {}).get('category', 'miscellaneous')
        if cat not in categories:
            categories[cat] = 0
        categories[cat] += 1
    
    # Create ultimate dataset
    ultimate_dataset = {
        "metadata": {
            "version": "4.0-ultimate",
            "generator": "All Datasets Combined",
            "total_examples": len(examples),
            "train_examples": len(examples),
            "val_examples": 0,
            "test_examples": 0,
            "features": ["relations", "facts", "rules", "queries", "error_detection", 
                        "error_recovery", "syntax_validation", "synthetic_generation",
                        "mathematical_computation", "scaling_up", "rate_calculations", 
                        "ratio_calculations", "unit_conversion", "percentages_ratios"],
            "compatibility": "ByteLogic 2.0 Standard Syntax + Error Handling",
            "datasets_combined": [
                "complete_bytelogic_dataset.json",
                "expanded_bytelogic_dataset.json", 
                "comprehensive_bytelogic_dataset.json",
                "rate_ratio_bytelogic_dataset.json",
                "various JSONL files"
            ],
            "category_distribution": categories
        },
        "train": examples
    }
    
    output_path = "training/datasets/ultimate_bytelogic_dataset.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(ultimate_dataset, f, ensure_ascii=False, indent=2)
    
    print(f"\\nUltimate dataset saved to: {output_path}")
    print(f"Total examples in ultimate dataset: {len(examples)}")
    
    print("\\nCategory distribution:")
    for cat, count in sorted(ultimate_dataset['metadata']['category_distribution'].items()):
        print(f"  - {cat}: {count}")
    
    return len(examples)

def main():
    print("Creating Ultimate Comprehensive Dataset")
    print("="*50)
    
    all_examples = combine_all_datasets()
    total_examples = save_ultimate_dataset(all_examples)
    
    print(f"\\nThe ultimate dataset contains {total_examples} diverse examples!")
    print("\\nExample usage:")
    print("  python3 train_bytelogic_worldmodel.py \\")
    print("    --model ~/workspace/model/Qwen3-0.6B \\")
    print("    --dataset training/datasets/ultimate_bytelogic_dataset.json \\")
    print("    --epochs 5")

if __name__ == "__main__":
    main()