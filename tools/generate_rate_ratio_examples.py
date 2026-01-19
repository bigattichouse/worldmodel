#!/usr/bin/env python3
"""
Generate Rate and Ratio Calculation Examples
"""

import json
import random
from pathlib import Path

def generate_rate_ratio_examples():
    """Generate rate and ratio calculation examples."""
    examples = []
    
    # Common contexts for rate calculations
    contexts = [
        "emails", "messages", "visitors", "sales", "downloads", 
        "views", "orders", "requests", "transactions", "posts",
        "calls", "tickets", "reports", "meetings", "updates"
    ]
    
    # Time periods and their conversions
    time_periods = [
        ("day", "week", 7), 
        ("day", "month", 30),
        ("day", "year", 365),
        ("hour", "day", 24),
        ("minute", "hour", 60),
        ("week", "month", 4),
        ("week", "year", 52),
        ("month", "year", 12)
    ]
    
    # Generate examples
    for i in range(1000):
        # Select context and time periods
        context = random.choice(contexts)
        period1, period2, multiplier = random.choice(time_periods)
        
        # Generate rate
        rate = random.randint(1, 50)  # Occurrences per first time period
        
        # Calculate result
        result = rate * multiplier
        
        # Create different question formats
        question_formats = [
            f"If {rate} {context} happen per {period1}, how many {context} happen in a {period2}?",
            f"How many {context} occur in a {period2} if {rate} {context} occur each {period1}?",
            f"In a {period2}, how many {context} will there be if the rate is {rate} per {period1}?",
            f"At a rate of {rate} {context} per {period1}, how many {context} are there in a {period2}?",
        ]
        
        question = random.choice(question_formats)
        
        # Create ByteLogic code
        byte_logic = f"""REL rate_calculation
REL rate
REL time_period
FACT rate {context}_per_{period1} {rate}
FACT time_period {period2}_equals_{period1}s {multiplier}
SOLVE
QUERY rate ? ?
"""
        
        output = f"Rate calculation result: <computation>\\n{byte_logic.strip()}\\n</computation> → {result} {context}"
        
        example = {
            "input": question,
            "output": output,
            "metadata": {
                "id": f"rate_ratio_{i}",
                "category": "mathematical_computation",
                "subcategory": "rate_and_ratio",
                "difficulty": "beginner" if i < 500 else "intermediate"
            }
        }
        examples.append(example)
    
    # Add compound rate examples
    for i in range(250):
        context1 = random.choice(contexts)
        context2 = random.choice([ctx for ctx in contexts if ctx != context1])
        
        # Rates for each context
        rate1 = random.randint(2, 20)  # per hour
        rate2 = random.randint(1, 15)  # per minute
        
        # Time periods
        period1, period2, multiplier = random.choice([("hour", "day", 24), ("minute", "hour", 60)])
        
        # Calculate results
        result1 = rate1 * multiplier
        result2 = rate2 * multiplier
        
        question = f"If {rate1} {context1} happen per {period1} and {rate2} {context2} happen per {period1}, how many total occur in a {period2}?"
        
        # Create more complex ByteLogic code
        byte_logic = f"""REL rate1
REL rate2
REL total
FACT rate1 {context1}_per_{period1} {rate1}
FACT rate2 {context2}_per_{period1} {rate2}
FACT rate1 {period2}_multiplier {multiplier}
SOLVE
QUERY rate1 ? ?
QUERY rate2 ? ?
"""
        
        total_result = result1 + result2
        output = f"Compound rate calculation: <computation>\\n{byte_logic.strip()}\\n</computation> → {result1} {context1} and {result2} {context2} for a total of {total_result}"
        
        example = {
            "input": question,
            "output": output,
            "metadata": {
                "id": f"compound_rate_{i}",
                "category": "mathematical_computation", 
                "subcategory": "compound_rates",
                "difficulty": "intermediate"
            }
        }
        examples.append(example)
    
    # Add percentage/ratio examples
    for i in range(250):
        base_amount = random.randint(10, 1000)
        percentage = random.randint(5, 90)
        
        result = round(base_amount * percentage / 100, 2)
        
        question_types = [
            f"If something increases by {percentage}% from {base_amount}, what is the new amount?",
            f"What is {percentage}% of {base_amount}?",
            f"If {percentage}% of {base_amount} items meet a criteria, how many items is that?",
            f"A value of {base_amount} changes by {percentage}%. What is the new value?"
        ]
        
        question = random.choice(question_types)
        
        byte_logic = f"""REL percentage_calc
REL amount
FACT percentage_calc base {base_amount}
FACT percentage_calc percent {percentage}
SOLVE
QUERY percentage_calc ? ?
"""
        
        output = f"Percentage calculation: <computation>\\n{byte_logic.strip()}\\n</computation> → {result}"
        
        example = {
            "input": question,
            "output": output,
            "metadata": {
                "id": f"percentage_{i}",
                "category": "mathematical_computation",
                "subcategory": "percentages_ratios", 
                "difficulty": "intermediate"
            }
        }
        examples.append(example)
    
    return examples

def save_rate_ratio_dataset(examples):
    """Save rate/ratio examples to a new dataset."""
    dataset = {
        "metadata": {
            "version": "1.0-rate-ratio",
            "generator": "Rate and Ratio Calculation Examples",
            "total_examples": len(examples),
            "train_examples": len(examples),
            "val_examples": 0,
            "test_examples": 0,
            "features": ["mathematical_computation", "rate_calculations", "ratio_calculations", "unit_conversion"],
            "compatibility": "ByteLogic 2.0 Standard Syntax + Error Handling",
            "categories": ["rate_and_ratio", "compound_rates", "percentages_ratios"],
            "example_count": {
                "rate_ratio": sum(1 for ex in examples if "rate_ratio_" in ex['metadata']['id']),
                "compound_rate": sum(1 for ex in examples if "compound_rate_" in ex['metadata']['id']), 
                "percentage": sum(1 for ex in examples if "percentage_" in ex['metadata']['id'])
            }
        },
        "train": examples
    }
    
    output_path = "training/datasets/rate_ratio_bytelogic_dataset.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
    
    print(f"Rate and ratio dataset saved to: {output_path}")
    print(f"Total examples: {len(examples)}")
    print(f"Rate examples: {dataset['metadata']['example_count']['rate_ratio']}")
    print(f"Compound rate examples: {dataset['metadata']['example_count']['compound_rate']}")
    print(f"Percentage examples: {dataset['metadata']['example_count']['percentage']}")
    
    print("\\nSample examples:")
    for i, ex in enumerate(examples[:3]):
        print(f"  {i+1}. {ex['input']}")
    
    return output_path

def main():
    print("Generating Rate and Ratio Calculation Examples")
    print("="*60)
    
    print("Creating 1,500 rate/ratio calculation examples...")
    examples = generate_rate_ratio_examples()
    print(f"Generated {len(examples)} examples")
    
    save_path = save_rate_ratio_dataset(examples)
    
    print(f"\\nTo use this dataset separately:")
    print(f"  --dataset {save_path}")
    
    print("\\nOr to combine with the comprehensive dataset:")
    print("  The main dataset can be combined with this rate/ratio dataset for even more variety")

if __name__ == "__main__":
    main()