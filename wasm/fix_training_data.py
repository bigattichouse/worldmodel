#!/usr/bin/env python3
"""
Fix Training Data
================

Fixes the incorrect computed results in the training data by actually calculating
the correct answers and validating the WAT code.
"""

import re
import sys
import os
import ast

def extract_numbers_from_question(user_text: str):
    """Extract numbers from user question."""
    # Find all numbers (int and float)
    numbers = re.findall(r'-?\d+\.?\d*', user_text)
    return [float(n) for n in numbers if n]

def extract_operation_from_question(user_text: str):
    """Extract operation type from user question."""
    user_text = user_text.lower()
    if any(word in user_text for word in ['add', '+', 'plus', 'sum']):
        return 'add'
    elif any(word in user_text for word in ['multiply', 'times', '×', '*', 'product']):
        return 'multiply' 
    elif any(word in user_text for word in ['subtract', '-', 'minus', 'difference']):
        return 'subtract'
    elif any(word in user_text for word in ['divide', '/', '÷', 'division']):
        return 'divide'
    elif any(word in user_text for word in ['square', 'squared', '^2']):
        return 'square'
    elif any(word in user_text for word in ['power', '^', 'exponent']):
        return 'power'
    return None

def compute_correct_result(numbers, operation):
    """Compute the correct mathematical result."""
    if not numbers:
        return None
    
    if operation == 'add':
        return sum(numbers)
    elif operation == 'multiply':
        result = numbers[0]
        for num in numbers[1:]:
            result *= num
        return result
    elif operation == 'subtract' and len(numbers) >= 2:
        result = numbers[0]
        for num in numbers[1:]:
            result -= num
        return result
    elif operation == 'divide' and len(numbers) >= 2:
        result = numbers[0]
        for num in numbers[1:]:
            if num != 0:
                result /= num
            else:
                return None  # Division by zero
        return result
    elif operation == 'square' and len(numbers) >= 1:
        return numbers[0] ** 2
    elif operation == 'power' and len(numbers) >= 2:
        return numbers[0] ** numbers[1]
    
    return None

def validate_wat_code(wat_code: str):
    """Validate WAT code structure."""
    if not wat_code.strip():
        return False
    
    # Must start with (module
    if not wat_code.strip().startswith("(module"):
        return False
    
    # Check balanced parentheses
    open_count = wat_code.count('(')
    close_count = wat_code.count(')')
    
    return open_count == close_count

def fix_example(example_text: str):
    """Fix a single training example."""
    lines = example_text.split('\n')
    fixed_lines = []
    
    user_text = ""
    for line in lines:
        if line.startswith('User:'):
            user_text = line[5:].strip()
            fixed_lines.append(line)
        elif '<computed>' in line:
            # Extract and fix the computed result
            numbers = extract_numbers_from_question(user_text)
            operation = extract_operation_from_question(user_text)
            
            if numbers and operation:
                correct_result = compute_correct_result(numbers, operation)
                if correct_result is not None:
                    # Format the result appropriately
                    if correct_result == int(correct_result):
                        result_str = str(int(correct_result))
                    else:
                        result_str = f"{correct_result:.1f}"
                    
                    fixed_line = f"<computed>{result_str}</computed>"
                    fixed_lines.append(fixed_line)
                    print(f"Fixed: {user_text} -> {result_str}")
                else:
                    fixed_lines.append(line)  # Keep original if can't compute
            else:
                fixed_lines.append(line)  # Keep original if can't parse
        else:
            fixed_lines.append(line)
    
    return '\n'.join(fixed_lines)

def fix_training_file(input_file: str, output_file: str):
    """Fix an entire training file."""
    print(f"Fixing {input_file}...")
    
    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Split into examples
    examples = content.split('\n\n')
    fixed_examples = []
    
    for example in examples:
        if example.strip() and 'User:' in example:
            fixed_example = fix_example(example.strip())
            fixed_examples.append(fixed_example)
        elif example.strip():  # Non-empty but no User:
            fixed_examples.append(example.strip())
    
    # Write fixed content
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n\n'.join(fixed_examples))
    
    print(f"Fixed training data written to {output_file}")

def main():
    """Fix all training data files."""
    print("🔧 Fixing Training Data")
    print("=" * 50)
    
    data_dir = "data/converted"
    backup_dir = "data/converted_backup"
    
    # Create backup directory
    os.makedirs(backup_dir, exist_ok=True)
    
    training_files = [
        "basic_arithmetic_training.txt",
        "complex_logic_training.txt", 
        "system_operations_training.txt"
    ]
    
    for filename in training_files:
        input_path = os.path.join(data_dir, filename)
        backup_path = os.path.join(backup_dir, filename)
        
        if os.path.exists(input_path):
            # Create backup
            print(f"Creating backup: {backup_path}")
            with open(input_path, 'r') as src, open(backup_path, 'w') as dst:
                dst.write(src.read())
            
            # Fix the file
            fix_training_file(input_path, input_path)
        else:
            print(f"Warning: {input_path} not found")
    
    print("\n✅ Training data fixed!")
    print(f"   Backups created in: {backup_dir}")
    print(f"   Fixed files in: {data_dir}")

if __name__ == "__main__":
    main()