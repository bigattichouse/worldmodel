#!/usr/bin/env python3
"""
Combine all training datasets into final comprehensive set.
"""

import json
import random
from pathlib import Path
from datetime import datetime, timezone

def load_dataset(file_path: str) -> list:
    """Load examples from a dataset file."""
    with open(file_path) as f:
        data = json.load(f)
    return data.get('examples', [])

def main():
    """Combine all datasets."""
    print("Combining all training datasets...")
    
    # Load all datasets
    datasets = []
    
    # Original synthetic data (all computational)
    try:
        original = load_dataset('./data/worldmodel_training.json')
        print(f"✅ Loaded {len(original)} examples from original dataset")
        datasets.append(("original", original))
    except FileNotFoundError:
        print("⚠️  Original dataset not found")
    
    # Comprehensive dataset (mixed computational/non-computational)
    try:
        comprehensive = load_dataset('./data/worldmodel_comprehensive_training.json')
        print(f"✅ Loaded {len(comprehensive)} examples from comprehensive dataset")
        datasets.append(("comprehensive", comprehensive))
    except FileNotFoundError:
        print("⚠️  Comprehensive dataset not found")
    
    # Additional synthetic data
    try:
        additional = load_dataset('./data/synthetic_additional.json')
        print(f"✅ Loaded {len(additional)} examples from additional dataset")
        datasets.append(("additional", additional))
    except FileNotFoundError:
        print("⚠️  Additional dataset not found")
    
    # Combine all examples
    all_examples = []
    source_counts = {}
    
    for source_name, examples in datasets:
        all_examples.extend(examples)
        source_counts[source_name] = len(examples)
    
    # Shuffle for better training distribution
    random.shuffle(all_examples)
    
    # Analyze combined dataset
    categories = {}
    computational_count = 0
    non_computational_count = 0
    
    for example in all_examples:
        category = example.get('category', 'unknown')
        categories[category] = categories.get(category, 0) + 1
        
        # Determine if computational based on target output
        target = example.get('target_output', '')
        if '<think>' in target and '<model>' in target and '<requires>' in target:
            computational_count += 1
        else:
            non_computational_count += 1
    
    # Create final dataset
    final_dataset = {
        "metadata": {
            "total_examples": len(all_examples),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "generator_version": "3.0_combined",
            "description": "Combined WorldModel training dataset for instinctive behavior",
            "source_breakdown": source_counts,
            "computational_examples": computational_count,
            "non_computational_examples": non_computational_count,
            "categories": categories
        },
        "examples": all_examples
    }
    
    # Save final dataset
    output_file = Path("./data/worldmodel_final_training.json")
    with open(output_file, 'w') as f:
        json.dump(final_dataset, f, indent=2)
    
    print(f"\\n🎯 Final Dataset Created!")
    print(f"   📊 Total examples: {len(all_examples)}")
    print(f"   🧠 Computational: {computational_count} ({computational_count/len(all_examples)*100:.1f}%)")
    print(f"   💬 Non-computational: {non_computational_count} ({non_computational_count/len(all_examples)*100:.1f}%)")
    print(f"   📝 Categories: {list(categories.keys())}")
    print(f"   💾 Saved to: {output_file}")
    
    print(f"\\n📈 Dataset Breakdown:")
    for source, count in source_counts.items():
        print(f"   {source}: {count} examples")
    
    # Show some example pairs
    print(f"\\n🔍 Sample Examples:")
    
    # Find one computational and one non-computational example
    comp_example = None
    non_comp_example = None
    
    for example in all_examples:
        target = example.get('target_output', '')
        if '<think>' in target and comp_example is None:
            comp_example = example
        elif '<think>' not in target and non_comp_example is None:
            non_comp_example = example
        
        if comp_example and non_comp_example:
            break
    
    if comp_example:
        print(f"\\n🧮 COMPUTATIONAL Example:")
        print(f"   Input: {comp_example['input_text']}")
        print(f"   Output: {comp_example['target_output'][:80]}...")
    
    if non_comp_example:
        print(f"\\n💭 NON-COMPUTATIONAL Example:")
        print(f"   Input: {non_comp_example['input_text']}")
        print(f"   Output: {non_comp_example['target_output'][:80]}...")

if __name__ == "__main__":
    main()