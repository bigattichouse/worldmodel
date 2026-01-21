#!/usr/bin/env python3
"""
Validate JSONL files in the history directory.
"""

import json
from pathlib import Path


def is_valid_jsonl_file(file_path: Path) -> bool:
    """Check if a file is a valid JSONL file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    json.loads(line)
                except json.JSONDecodeError as e:
                    print(f"  Invalid JSON in {file_path.name} at line {line_num}: {e}")
                    return False
    except Exception as e:
        print(f"  Error reading {file_path.name}: {e}")
        return False
    return True


def main():
    history_dir = Path('./history')
    jsonl_files = list(history_dir.rglob('*.jsonl'))
    
    print(f"Validating {len(jsonl_files)} JSONL files in history directory...\n")
    
    valid_files = 0
    invalid_files = 0
    
    for file_path in jsonl_files:
        print(f"Checking: {file_path}")
        if is_valid_jsonl_file(file_path):
            print("  ✓ Valid")
            valid_files += 1
        else:
            print("  ✗ Invalid")
            invalid_files += 1
        print()
    
    print(f"Summary:")
    print(f"  Valid files: {valid_files}")
    print(f"  Invalid files: {invalid_files}")
    print(f"  Total files: {len(jsonl_files)}")


if __name__ == "__main__":
    main()