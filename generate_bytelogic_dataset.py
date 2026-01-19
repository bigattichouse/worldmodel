#!/usr/bin/env python3
"""
Generate ByteLogic Training Dataset
==================================

Generate a comprehensive training dataset for ByteLogic computation tokens.
"""

import sys
import os
import json
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def main():
    """Generate the ByteLogic training dataset."""
    print("🚀 Generating ByteLogic Training Dataset...")
    
    try:
        # Import without relative imports
        sys.path.append("src")
        from training.bytelogic_dataset import ByteLogicDatasetGenerator
        
        # Create generator (disable validation since we need to test compilation)
        generator = ByteLogicDatasetGenerator(validate_examples=False)
        print(f"  ✅ Dataset generator initialized")
        
        # Get statistics
        stats = generator.get_statistics()
        print(f"  📊 Base examples: {stats['base_examples']}")
        print(f"     Categories: {', '.join(stats['categories'])}")
        print(f"     Subcategories: {', '.join(stats['subcategories'])}")
        
        # Generate dataset
        output_file = "training/datasets/bytelogic_training_dataset.json"
        os.makedirs("training/datasets", exist_ok=True)
        
        metadata = generator.generate_training_dataset(
            output_file=output_file,
            num_variations_per_example=3,  # 3 variations per example
            train_split=0.8,
            val_split=0.1
        )
        
        print(f"\n✅ Dataset generated successfully!")
        print(f"   File: {output_file}")
        print(f"   Total examples: {metadata['total_examples']}")
        print(f"   Training: {metadata['train_examples']}")
        print(f"   Validation: {metadata['val_examples']}")
        print(f"   Test: {metadata['test_examples']}")
        
        # Generate additional JSONL files for easy streaming
        with open(output_file, 'r') as f:
            dataset = json.load(f)
        
        # Export train set to JSONL
        train_jsonl = "training/datasets/bytelogic_train.jsonl"
        generator.export_examples_to_jsonl(dataset['train'], train_jsonl)
        print(f"   Train JSONL: {train_jsonl}")
        
        # Export validation set to JSONL
        val_jsonl = "training/datasets/bytelogic_val.jsonl"
        generator.export_examples_to_jsonl(dataset['validation'], val_jsonl)
        print(f"   Validation JSONL: {val_jsonl}")
        
        # Show sample examples
        print(f"\n📝 Sample Training Examples:")
        print("=" * 60)
        
        for i, example in enumerate(dataset['train'][:3]):
            print(f"\nExample {i+1} ({example['metadata']['category']}):")
            print(f"Input: {example['input']}")
            print(f"Output: {example['output'][:100]}...")
        
        print("\n🎉 ByteLogic training dataset generation complete!")
        return True
        
    except Exception as e:
        print(f"❌ Dataset generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # Change to worldmodel directory
    os.chdir(Path(__file__).parent)
    
    success = main()
    exit(0 if success else 1)