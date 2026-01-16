#!/usr/bin/env python3
"""
Fix WAT Exports
===============

Adds proper function exports to WAT code in training data for real WASM execution.
"""

import re
import os

def fix_wat_exports(content: str) -> str:
    """Fix WAT code to include proper exports."""
    # Pattern to match WAT modules
    wat_pattern = r'<wat_model>\s*\(module[^<]+\)\s*</wat_model>'
    
    def fix_module(match):
        wat_content = match.group(0)
        
        # Extract the inner module content
        module_start = wat_content.find('(module')
        module_end = wat_content.rfind(')')
        module_content = wat_content[module_start:module_end+1]
        
        # Check if it already has an export
        if '(export' in module_content:
            return wat_content  # Already has exports
        
        # Find function definition
        func_match = re.search(r'\(func \$(\w+)', module_content)
        if not func_match:
            return wat_content  # No function found
        
        func_name = func_match.group(1)
        
        # Add export before the closing parenthesis
        # Insert export line before final )
        insert_pos = module_content.rfind(')')
        export_line = f'\n  (export "compute" (func ${func_name}))'
        fixed_module = module_content[:insert_pos] + export_line + module_content[insert_pos:]
        
        # Replace in original content
        return wat_content.replace(module_content, fixed_module)
    
    # Apply fix to all WAT modules
    fixed_content = re.sub(wat_pattern, fix_module, content, flags=re.DOTALL)
    return fixed_content

def fix_training_file(filepath: str):
    """Fix WAT exports in a training file."""
    print(f"Fixing WAT exports in {filepath}...")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    fixed_content = fix_wat_exports(content)
    
    # Count changes
    original_exports = content.count('(export')
    fixed_exports = fixed_content.count('(export')
    changes = fixed_exports - original_exports
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(fixed_content)
    
    print(f"   Added {changes} function exports")

def main():
    """Fix WAT exports in all training files."""
    print("🔧 Adding WAT Function Exports")
    print("=" * 50)
    
    data_dir = "data/converted"
    training_files = [
        "basic_arithmetic_training.txt",
        "complex_logic_training.txt",
        "system_operations_training.txt"
    ]
    
    total_changes = 0
    
    for filename in training_files:
        filepath = os.path.join(data_dir, filename)
        if os.path.exists(filepath):
            fix_training_file(filepath)
        else:
            print(f"Warning: {filepath} not found")
    
    print(f"\n✅ WAT exports fixed!")
    print(f"   Training data now has properly exported WASM functions")

if __name__ == "__main__":
    main()