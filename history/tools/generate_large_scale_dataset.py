#!/usr/bin/env python3
"""
Generate thousands of mathematical examples and add them to existing datasets
"""

import json
import random
import os
from pathlib import Path

def generate_math_examples(num_examples=2000):
    """Generate thousands of mathematical calculation examples."""
    examples = []
    
    operations = [
        ("add", lambda a, b: a + b, "+", "plus", "sum"),
        ("subtract", lambda a, b: a - b if a >= b else b - a, "-", "minus", "difference"), 
        ("multiply", lambda a, b: a * b, "*", "times", "product"),
        ("divide", lambda a, b: a // b if b != 0 and a % b == 0 else (b // a if b % a == 0 else None), "/", "divided by", "quotient"),
    ]
    
    for i in range(num_examples):
        # Choose a random operation
        op_name, op_func, op_symbol, op_word, op_desc = random.choice(operations)
        
        # Generate appropriate numbers based on operation
        if op_name == "divide":
            # For division, make sure we get clean results
            b = random.randint(1, 20)
            a_mult = random.randint(1, 10)
            a = b * a_mult  # So a / b will be clean
        elif op_name == "subtract":
            # For subtraction, make sure a >= b to avoid negatives for simplicity
            a = random.randint(1, 100)
            b = random.randint(1, min(a, 50))
        else:
            # For add/multiply, just pick random numbers
            a = random.randint(1, 50)
            b = random.randint(1, 50)
        
        # Calculate result
        result = op_func(a, b)
        if result is None:  # Skip invalid divisions
            continue
            
        # Create multiple variations of the question
        question_templates = [
            f"What is {a} {op_symbol} {b}?",
            f"Calculate {a} {op_word} {b}", 
            f"What is the {op_desc} of {a} and {b}?",
            f"Compute {a} {op_symbol} {b}",
        ]
        
        # Choose a random template
        question = random.choice(question_templates)
        
        # Generate ByteLogic code
        byte_logic = f"""REL calculation
REL operand
FACT calculation operation {op_name}
FACT operand first {a}
FACT operand second {b}
SOLVE
QUERY calculation ? ?
"""
        
        output = f"Calculation result: <computation>\\n{byte_logic.strip()}\\n</computation> → {result}"
        
        example = {
            "input": question,
            "output": output,
            "metadata": {
                "id": f"math_gen_{i}",
                "category": "mathematical_computation",
                "subcategory": f"{op_name}_operations",
                "difficulty": "beginner" if i < num_examples // 2 else "intermediate"
            }
        }
        examples.append(example)
        
        # Add some fractional/decimal operations occasionally
        if i % 50 == 0:  # Every 50th example
            # Generate decimal calculation
            a_float = round(random.uniform(1, 20), 1)
            b_float = round(random.uniform(1, 10), 1)
            
            # Only do addition and multiplication for decimals to avoid rounding issues
            float_ops = [("add", a_float + b_float, "plus"), ("multiply", a_float * b_float, "times")]
            float_op, float_result, float_word = random.choice(float_ops)
            
            float_question = f"What is {a_float} {float_word} {b_float}?"
            float_byte_logic = f"""REL decimal_calc
REL operand
FACT decimal_calc operation {float_op}
FACT operand first {a_float}
FACT operand second {b_float}
SOLVE
QUERY decimal_calc ? ?
"""
            
            float_output = f"Decimal calculation result: <computation>\\n{float_byte_logic.strip()}\\n</computation> → {round(float_result, 2)}"
            
            float_example = {
                "input": float_question,
                "output": float_output,
                "metadata": {
                    "id": f"decimal_gen_{i//50}",
                    "category": "mathematical_computation",
                    "subcategory": "decimal_operations",
                    "difficulty": "intermediate"
                }
            }
            examples.append(float_example)
    
    return examples

def load_existing_datasets():
    """Load all existing JSON files from the training directory."""
    datasets_dir = Path("training/datasets")
    all_examples = []
    
    # Load all .json files
    for json_file in datasets_dir.glob("*.json"):
        if json_file.name.startswith("."):  # Skip hidden files
            continue
        if json_file.name == "large_scale_bytelogic_dataset.json":  # Skip the output file
            continue
            
        print(f"Loading {json_file.name}")
        with open(json_file, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
                if "train" in data:
                    all_examples.extend(data["train"])
                    print(f"  - Added {len(data['train'])} training examples")
                else:
                    print(f"  - No 'train' key found, skipping")
            except Exception as e:
                print(f"  - Error loading {json_file.name}: {e}")
    
    # For JSONL files, we'll read them line by line
    for jsonl_file in datasets_dir.glob("*.jsonl"):
        if jsonl_file.name.startswith("."):  # Skip hidden files
            continue
            
        print(f"Loading {jsonl_file.name}")
        try:
            with open(jsonl_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if line:
                        try:
                            example = json.loads(line)
                            if isinstance(example, dict) and 'input' in example and 'output' in example:
                                all_examples.append(example)
                        except json.JSONDecodeError:
                            print(f"  - Failed to parse line {line_num} in {jsonl_file.name}")
            print(f"  - Added examples from {jsonl_file.name}")
        except Exception as e:
            print(f"  - Error loading {jsonl_file.name}: {e}")
    
    print(f"Total loaded examples: {len(all_examples)}")
    return all_examples

def save_combined_dataset(examples):
    """Save the combined dataset with the added synthetic examples."""
    # Sort examples by category for better organization
    categorized_examples = {}
    for ex in examples:
        cat = ex.get('metadata', {}).get('category', 'mixed')
        if cat not in categorized_examples:
            categorized_examples[cat] = []
        categorized_examples[cat].append(ex)
    
    # Flatten back to a single list
    combined_examples = []
    for cat in sorted(categorized_examples.keys()):
        combined_examples.extend(categorized_examples[cat])
    
    # Create dataset with metadata
    combined_dataset = {
        "metadata": {
            "version": "3.0-large_scale",
            "generator": "Original Datasets + Large Scale Mathematical Synthesis",
            "total_examples": len(combined_examples),
            "train_examples": len(combined_examples),
            "val_examples": 0,  # Could split later if needed
            "test_examples": 0,  # Could split later if needed
            "features": ["relations", "facts", "rules", "queries", "error_detection", 
                        "error_recovery", "syntax_validation", "synthetic_generation", 
                        "mathematical_computation", "scaling_up"],
            "compatibility": "ByteLogic 2.0 Standard Syntax + Error Handling",
            "synthetic_math_examples": len([ex for ex in examples if 'math_gen_' in ex.get('metadata', {}).get('id', '') or 'decimal_gen_' in ex.get('metadata', {}).get('id', '')]),
            "original_examples": len([ex for ex in examples if 'math_gen_' not in ex.get('metadata', {}).get('id', '') and 'decimal_gen_' not in ex.get('metadata', {}).get('id', '')])
        },
        "train": combined_examples
    }
    
    output_path = "training/datasets/large_scale_bytelogic_dataset.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(combined_dataset, f, ensure_ascii=False, indent=2)
    
    print(f"\\nLarge scale dataset saved to: {output_path}")
    print(f"Total examples in combined dataset: {len(combined_examples)}")
    print(f"Synthetic math examples added: {combined_dataset['metadata']['synthetic_math_examples']}")
    print(f"Original examples: {combined_dataset['metadata']['original_examples']}")
    
    # Show sample of newly generated examples
    print("\\nSample of generated mathematical examples:")
    math_examples = [ex for ex in combined_examples if 'math_gen_' in ex.get('metadata', {}).get('id', '')][:3]
    for i, ex in enumerate(math_examples):
        print(f"  {i+1}. {ex['input']}")

def main():
    print("Generating Large Scale Mathematical Dataset")
    print("="*60)
    
    print("Loading existing datasets...")
    existing_examples = load_existing_datasets()
    
    print(f"\\nGenerating 2,000 new mathematical examples...")
    new_examples = generate_math_examples(2000)
    print(f"Generated {len(new_examples)} mathematical examples")
    
    print(f"\\nCombining datasets...")
    all_examples = existing_examples + new_examples
    print(f"Total examples after combining: {len(all_examples)}")
    
    save_combined_dataset(all_examples)
    
    print(f"\\nTo use this dataset for training:")
    print(f"  --dataset training/datasets/large_scale_bytelogic_dataset.json")

if __name__ == "__main__":
    main()