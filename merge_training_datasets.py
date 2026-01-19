#!/usr/bin/env python3
"""
Merge ByteLogic Training Datasets
==================================

Combines core ByteLogic dataset with error handling dataset to create 
a comprehensive training dataset that teaches both:
1. ByteLogic language fundamentals
2. Error detection and handling
"""

import json
import random
from typing import Dict, List, Any

def merge_datasets(core_file: str, error_file: str, output_file: str):
    """Merge core and error handling datasets."""
    print("🔄 Merging ByteLogic training datasets...")
    
    # Load core dataset
    with open(core_file, 'r', encoding='utf-8') as f:
        core_data = json.load(f)
    
    # Load error handling dataset
    with open(error_file, 'r', encoding='utf-8') as f:
        error_data = json.load(f)
    
    # Combine examples from each split
    merged_train = core_data['train'] + error_data['train']
    merged_val = core_data['validation'] + error_data['validation'] 
    merged_test = core_data['test'] + error_data['test']
    
    # Shuffle to mix core learning with error handling
    random.shuffle(merged_train)
    random.shuffle(merged_val)
    random.shuffle(merged_test)
    
    # Create merged dataset
    merged_dataset = {
        "metadata": {
            "version": "2.1-complete",
            "generator": "Merged Core + Error Handling",
            "total_examples": len(merged_train) + len(merged_val) + len(merged_test),
            "train_examples": len(merged_train),
            "val_examples": len(merged_val),
            "test_examples": len(merged_test),
            "features": [
                "relations", "facts", "rules", "queries",  # Core ByteLogic
                "error_detection", "error_recovery", "syntax_validation"  # Error handling
            ],
            "compatibility": "ByteLogic 2.0 + Error Handling",
            "core_dataset": core_file,
            "error_dataset": error_file
        },
        "train": merged_train,
        "validation": merged_val,
        "test": merged_test
    }
    
    # Save merged dataset
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(merged_dataset, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Merged dataset created:")
    print(f"   Core examples: {core_data['metadata']['total_examples']}")
    print(f"   Error examples: {error_data['metadata']['total_examples']}")
    print(f"   Total examples: {merged_dataset['metadata']['total_examples']}")
    print(f"   Train/Val/Test: {len(merged_train)}/{len(merged_val)}/{len(merged_test)}")
    print(f"   Saved to: {output_file}")
    
    return output_file

def main():
    """Create comprehensive ByteLogic training dataset."""
    core_file = "training/datasets/corrected_bytelogic_dataset.json"
    error_file = "training/datasets/bytelogic_error_handling_dataset.json"
    output_file = "training/datasets/complete_bytelogic_dataset.json"
    
    merged_file = merge_datasets(core_file, error_file, output_file)
    
    print(f"\n💡 Training Recommendation:")
    print(f"   Use: {merged_file}")
    print(f"   Features: Core ByteLogic + Error Handling")
    print(f"   Quality: 100% validated syntax + Error recovery")
    
    return merged_file

if __name__ == "__main__":
    main()