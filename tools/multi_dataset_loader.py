#!/usr/bin/env python3
"""
Multi-Dataset Loader for ByteLogic WorldModel Training
====================================================

Dynamically loads all JSON/JSONL datasets from the training directory.
"""

import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Any
import random

def load_json_file(filepath: Path) -> List[Dict[str, Any]]:
    """Load examples from a JSON file."""
    print(f"  Loading {filepath.name}...")
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
        if 'train' in data:
            examples = data['train']
        else:
            # If no train key, assume it's a direct array of examples
            if isinstance(data, list):
                examples = data
            else:
                examples = []
        
        print(f"    - Loaded {len(examples)} examples")
        return examples

def load_jsonl_file(filepath: Path) -> List[Dict[str, Any]]:
    """Load examples from a JSONL file."""
    print(f"  Loading {filepath.name}...")
    examples = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line:
                    try:
                        example = json.loads(line)
                        if isinstance(example, dict) and 'input' in example and 'output' in example:
                            examples.append(example)
                    except json.JSONDecodeError:
                        print(f"    - Warning: Failed to parse line {line_num} in {filepath.name}")
    except Exception as e:
        print(f"    - Error loading {filepath.name}: {e}")
    
    print(f"    - Loaded {len(examples)} examples")
    return examples

def load_all_datasets(datasets_dir: str = "training/datasets") -> List[Dict[str, Any]]:
    """Load all JSON and JSONL dataset files from the specified directory."""
    datasets_path = Path(datasets_dir)
    print(f"Loading all datasets from {datasets_path.absolute()}/")
    print("="*60)
    
    all_examples = []
    total_examples = 0
    
    # Load JSON files
    for json_file in datasets_path.glob("*.json"):
        # Skip backup files, temporary files, and the auto-generated file
        if json_file.name.startswith('.') or json_file.name.startswith('_') or json_file.name.endswith('.backup'):
            print(f"  Skipping {json_file.name} (hidden/temporary file)")
            continue
            
        try:
            examples = load_json_file(json_file)
            all_examples.extend(examples)
            total_examples += len(examples)
            print(f"  Added {len(examples)} examples from {json_file.name}")
        except Exception as e:
            print(f"  Error loading {json_file.name}: {e}")
    
    # Load JSONL files
    for jsonl_file in datasets_path.glob("*.jsonl"):
        # Skip hidden/temporary files
        if jsonl_file.name.startswith('.') or jsonl_file.name.startswith('_'):
            print(f"  Skipping {jsonl_file.name} (hidden/temporary file)")
            continue
            
        try:
            examples = load_jsonl_file(jsonl_file)
            all_examples.extend(examples)
            total_examples += len(examples)
            print(f"  Added {len(examples)} examples from {jsonl_file.name}")
        except Exception as e:
            print(f"  Error loading {jsonl_file.name}: {e}")
    
    print("="*60)
    print(f"Total loaded: {total_examples} examples from {len([f for f in datasets_path.glob('*') if f.suffix in ['.json', '.jsonl']])} dataset files")
    
    # Shuffle examples to mix different datasets together
    print("Shuffling examples to mix different datasets...")
    random.shuffle(all_examples)
    
    return all_examples

def create_dynamic_dataset(split_ratio: float = 0.8) -> Dict[str, Any]:
    """Create a dynamically combined dataset from all available files."""
    all_examples = load_all_datasets()
    
    if not all_examples:
        raise ValueError("No examples loaded from any dataset files!")
    
    # Split into train and validation sets
    split_idx = int(len(all_examples) * split_ratio)
    train_examples = all_examples[:split_idx]
    val_examples = all_examples[split_idx:]
    
    # Create dynamic metadata
    categories = set()
    difficulties = set()
    
    for example in train_examples:
        meta = example.get('metadata', {})
        category = meta.get('category', 'unknown')
        difficulty = meta.get('difficulty', 'unknown')
        categories.add(category)
        difficulties.add(difficulty)
    
    dataset = {
        "metadata": {
            "version": "dynamic_multi_source",
            "generator": "Dynamic Multi-Dataset Loader",
            "total_examples": len(all_examples),
            "train_examples": len(train_examples),
            "val_examples": len(val_examples),
            "split_ratio": split_ratio,
            "dynamic_sources": True,
            "loaded_categories": list(sorted(categories)),
            "loaded_difficulties": list(sorted(difficulties)),
            "source_directories": ["training/datasets"]
        },
        "train": train_examples,
        "val": val_examples
    }
    
    print(f"Created dynamic dataset with {len(train_examples)} train and {len(val_examples)} validation examples")
    print(f"Categories found: {list(sorted(categories))}")
    print(f"Difficulties found: {list(sorted(difficulties))}")
    
    return dataset

class DynamicDatasetLoader:
    """Class to handle dynamic loading of all available datasets."""
    
    def __init__(self, datasets_dir: str = "training/datasets", split_ratio: float = 0.8):
        self.datasets_dir = Path(datasets_dir)
        self.split_ratio = split_ratio
        self.dataset = None
    
    def load(self) -> Dict[str, Any]:
        """Load all datasets dynamically."""
        self.dataset = create_dynamic_dataset(self.split_ratio)
        return self.dataset
    
    def get_info(self) -> Dict[str, Any]:
        """Get information about loaded datasets."""
        if not self.dataset:
            return {"error": "No dataset loaded yet"}
        
        return {
            "total_examples": self.dataset["metadata"]["total_examples"],
            "train_examples": self.dataset["metadata"]["train_examples"],
            "val_examples": self.dataset["metadata"]["val_examples"],
            "categories": self.dataset["metadata"]["loaded_categories"],
            "difficulties": self.dataset["metadata"]["loaded_difficulties"],
            "dynamic_sources": True
        }

def main():
    """Demo of dynamic dataset loading."""
    print("Multi-Dataset Loader Demo")
    print("="*60)
    
    loader = DynamicDatasetLoader()
    
    try:
        dataset = loader.load()
        info = loader.get_info()
        
        print(f"\\nDataset Info:")
        print(f"  Total Examples: {info['total_examples']}")
        print(f"  Train: {info['train_examples']}")
        print(f"  Validation: {info['val_examples']}")
        print(f"  Categories: {len(info['categories'])} - {info['categories']}")
        print(f"  Difficulties: {len(info['difficulties'])} - {info['difficulties']}")
        
        # Show a sample of loaded examples
        print(f"\\nSample examples:")
        for i in range(min(3, len(dataset['train']))):
            example = dataset['train'][i]
            print(f"  {i+1}. {example['input'][:80]}...")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()