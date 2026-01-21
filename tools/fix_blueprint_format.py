#!/usr/bin/env python3
"""
Fix records that have incorrect BluePrint format (missing or incorrect tags).
Specifically fixes the common issue where '</king>' appears instead of '</thinking>'.
"""

import json
from pathlib import Path
import re
from collections import defaultdict


def fix_blueprint_format_in_file(file_path: Path) -> int:
    """
    Fix BluePrint format issues in a single file.
    Returns the number of records fixed.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    fixed_lines = []
    records_fixed = 0
    
    for line in lines:
        line = line.strip()
        if not line:
            fixed_lines.append(line + '\n')
            continue
            
        try:
            example = json.loads(line)
            
            # Check if it has the required BluePrint format
            response_key = None
            if 'response' in example:
                response_key = 'response'
            elif 'text' in example:
                response_key = 'text'
            
            if response_key:
                original_response = example[response_key]
                
                # Fix common issue: '</king>' should be '</thinking>'
                fixed_response = original_response.replace('</king>', '</thinking>')
                
                # Additional checks to ensure proper format
                has_thinking = '<thinking>' in fixed_response and '</thinking>' in fixed_response
                has_blueprint = '<blueprint>' in fixed_response and '</blueprint>' in fixed_response
                
                # If still missing proper format, try to reconstruct
                if not (has_thinking and has_blueprint):
                    # This is more complex - for now, just ensure tags are correct
                    pass
                
                if fixed_response != original_response:
                    example[response_key] = fixed_response
                    records_fixed += 1
                
            # Write the (potentially modified) example back
            fixed_lines.append(json.dumps(example) + '\n')
            
        except json.JSONDecodeError:
            # Keep invalid lines as they are
            fixed_lines.append(line + '\n')
    
    # Create backup
    backup_path = file_path.with_suffix(file_path.suffix + '.blueprint_fix')
    with open(backup_path, 'w', encoding='utf-8') as backup_f:
        with open(file_path, 'r', encoding='utf-8') as orig_f:
            backup_f.write(orig_f.read())
    
    # Write fixed content back to file
    with open(file_path, 'w', encoding='utf-8') as f:
        f.writelines(fixed_lines)
    
    return records_fixed


def main():
    datasets_dir = Path('./training/datasets')
    jsonl_files = list(datasets_dir.rglob('*.jsonl'))
    
    total_fixed = 0
    files_fixed = []
    
    print("Fixing BluePrint format issues...")
    
    for file_path in jsonl_files:
        records_in_file = fix_blueprint_format_in_file(file_path)
        if records_in_file > 0:
            total_fixed += records_in_file
            files_fixed.append((file_path.name, records_in_file))
            print(f"Fixed {records_in_file} records in {file_path.name}")
    
    print(f"\nSummary:")
    print(f"Total records fixed: {total_fixed}")
    print(f"Files updated: {len(files_fixed)}")
    
    if files_fixed:
        print(f"Updated files:")
        for filename, count in files_fixed:
            print(f"  - {filename}: {count} records")


if __name__ == "__main__":
    main()