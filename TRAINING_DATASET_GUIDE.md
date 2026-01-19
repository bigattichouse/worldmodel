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

3. COMPREHENSIVE DATASET:
   - File: training/datasets/comprehensive_bytelogic_dataset.json
   - Training examples: 1,577 (+80% from original, +28% from expanded)
   - Type: Original + compatible examples + 350 synthetic examples
   - Content: Mathematical, logical, and graph reasoning examples
   - Use with: --dataset training/datasets/comprehensive_bytelogic_dataset.json

4. RATE/RATIO DATASET (Specialized):
   - File: training/datasets/rate_ratio_bytelogic_dataset.json
   - Training examples: 1,500
   - Type: Rate, ratio, time period, and percentage calculations
   - Content: Time-based calculations (X per day/week/month/year), percentages, ratios
   - Use with: --dataset training/datasets/rate_ratio_bytelogic_dataset.json

5. ULTIMATE COMPREHENSIVE DATASET (Most Comprehensive):
   - File: training/datasets/ultimate_bytelogic_dataset.json
   - Training examples: 8,154
   - Type: All datasets combined (original + expanded + synthetic + rate/ratio)
   - Content: Everything in one comprehensive dataset
   - Use with: --dataset training/datasets/ultimate_bytelogic_dataset.json

6. LARGE-SCALE DATASET (Previous comprehensive):
   - File: training/datasets/large_scale_bytelogic_dataset.json
   - Training examples: 8,694
   - Type: All datasets + 2,000+ additional math examples
   - Use with: --dataset training/datasets/large_scale_bytelogic_dataset.json

RECOMMENDATIONS:

For MAXIMUM DATA (Prevents Overfitting - RECOMMENDED):
  Use the large-scale dataset (10x original size):
  python3 train_bytelogic_worldmodel.py \
    --model ~/workspace/model/Qwen3-0.6B \
    --dataset training/datasets/large_scale_bytelogic_dataset.json \
    --epochs 5

For ULTIMATE COMPREHENSIVE COVERAGE:
  Use the ultimate dataset (all combined):
  python3 train_bytelogic_worldmodel.py \
    --model ~/workspace/model/Qwen3-0.6B \
    --dataset training/datasets/ultimate_bytelogic_dataset.json \
    --epochs 5

For SPECIALIZED TIME/RATE CALCULATIONS:
  Use the rate/ratio dataset:
  python3 train_bytelogic_worldmodel.py \
    --model ~/workspace/model/Qwen3-0.6B \
    --dataset training/datasets/rate_ratio_bytelogic_dataset.json \
    --epochs 5

For COMPREHENSIVE BUT SMALLER DATASET:
  Use the comprehensive dataset:
  python3 train_bytelogic_worldmodel.py \
    --model ~/workspace/model/Qwen3-0.6B \
    --dataset training/datasets/comprehensive_bytelogic_dataset.json \
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
- Rate and ratio calculations (time periods, X per day/week/month/year)
- Percentage and proportion calculations
- Unit conversions and time period calculations

DATASET CHARACTERISTICS:
- All datasets maintain 100% standard ByteLogic syntax compatibility
- All datasets preserve same validation/test splits for consistent evaluation
- Large-scale dataset increases training data by ~1000% (880 -> 8,694 examples) helping prevent overfitting
- Ultimate dataset combines all datasets for comprehensive coverage (8,154 examples)
- Comprehensive dataset increases training data by 80% (880 -> 1,577 examples)
- Added diverse natural language patterns to improve generalization
- Added thousands of mathematical, rate/ratio, and calculation examples for stronger foundations

The large-scale dataset with 8,694 training examples is ideal for
training a robust, generalized model that won't overfit to the small dataset.
The ultimate dataset with 8,154 examples provides all-inclusive coverage.
The comprehensive dataset with 1,577 training examples is a good middle ground.
""")

if __name__ == "__main__":
    print_summary()