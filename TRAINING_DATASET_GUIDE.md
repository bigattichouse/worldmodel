#!/usr/bin/env python3
"""
Summary of all available datasets and recommendations
"""

def print_summary():
    print("DATASET OPTIONS SUMMARY")
    print("="*60)
    
    print("""
Available datasets for training:

1. ORIGINAL DATASET:
   - File: training/datasets/complete_bytelogic_dataset.json
   - Training examples: 880
   - Type: Original curated examples
   - Use with: --dataset training/datasets/complete_bytelogic_dataset.json

2. EXPANDED DATASET: 
   - File: training/datasets/expanded_bytelogic_dataset.json
   - Training examples: 1,227 (+40% from original)
   - Type: Original + 347 compatible examples from comprehensive dataset
   - Use with: --dataset training/datasets/expanded_bytelogic_dataset.json

3. COMPREHENSIVE DATASET (Recommended):
   - File: training/datasets/comprehensive_bytelogic_dataset.json
   - Training examples: 1,577 (+80% from original, +28% from expanded)
   - Type: Original + compatible examples + 350 synthetic examples
   - Content: Mathematical, logical, and graph reasoning examples
   - Use with: --dataset training/datasets/comprehensive_bytelogic_dataset.json

RECOMMENDATIONS:

For MAXIMUM DATA (Prevents Overfitting - RECOMMENDED):
  Use the large-scale dataset:
  python3 train_bytelogic_worldmodel.py \\
    --model ~/workspace/model/Qwen3-0.6B \\
    --dataset training/datasets/large_scale_bytelogic_dataset.json \\
    --epochs 5

For COMPREHENSIVE BUT SMALLER DATASET:
  Use the comprehensive dataset:
  python3 train_bytelogic_worldmodel.py \\
    --model ~/workspace/model/Qwen3-0.6B \\
    --dataset training/datasets/comprehensive_bytelogic_dataset.json \\
    --epochs 5

For STABILITY TESTING:
  Use the original or expanded dataset to compare with results

SYNTHETIC EXAMPLE CATEGORIES ADDED:
- Mathematical calculations (addition, subtraction, multiplication, division)
- Logical reasoning (family relations, social networks)
- Graph theory (reachability, connections)
- Compound operations (multi-step calculations)
- Decimal/real number operations
- Large-scale repetition for overfitting prevention

DATASET CHARACTERISTICS:
- All datasets maintain 100% standard ByteLogic syntax compatibility
- All datasets preserve same validation/test splits for consistent evaluation
- Large-scale dataset increases training data by ~1000% (880 -> 8,694 examples) helping prevent overfitting
- Comprehensive dataset increases training data by 80% (880 -> 1,577 examples)
- Added diverse natural language patterns to improve generalization
- Added thousands of mathematical calculation examples for stronger foundations

The large-scale dataset with 8,694 training examples is ideal for
training a robust, generalized model that won't overfit to the small dataset.
The comprehensive dataset with 1,577 training examples is a good middle ground.
""")

if __name__ == "__main__":
    print_summary()