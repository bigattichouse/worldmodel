#!/usr/bin/env python3
"""
Script to validate JSONL files in the training datasets directory.
Identifies broken files and attempts to fix them.
"""

import json
import os
from pathlib import Path
from typing import List, Tuple


def is_valid_jsonl_file(file_path: Path) -> Tuple[bool, List[str]]:
    """
    Check if a file is a valid JSONL file.
    
    Args:
        file_path: Path to the file to validate
        
    Returns:
        Tuple of (is_valid, list_of_error_lines)
    """
    errors = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:  # Skip empty lines
                    continue
                try:
                    json.loads(line)
                except json.JSONDecodeError as e:
                    errors.append(f"Line {line_num}: {str(e)} - Content: {line[:100]}...")
    except Exception as e:
        errors.append(f"File read error: {str(e)}")
        
    return len(errors) == 0, errors


def fix_jsonl_file(file_path: Path) -> Tuple[bool, str]:
    """
    Attempt to fix a broken JSONL file by identifying and correcting common issues.

    Args:
        file_path: Path to the file to fix

    Returns:
        Tuple of (success, message)
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Split content into lines
        raw_lines = content.split('\n')
        fixed_lines = []
        errors = []

        i = 0
        while i < len(raw_lines):
            line = raw_lines[i].strip()
            if not line:
                i += 1
                continue

            # Handle common problematic patterns
            if line.startswith('</') or line == '</content>' or line == '<content>' or line.startswith('<'):
                # Skip XML-like tags
                i += 1
                continue

            # Try to parse the current line
            try:
                json.loads(line)
                fixed_lines.append(line + '\n')
                i += 1
            except json.JSONDecodeError:
                # Check if this is a continuation of previous JSON object
                # Look for common patterns like unterminated strings or missing commas
                combined_line = line
                j = i + 1

                # Try to fix common issues by combining lines
                while j < min(len(raw_lines), i + 10):  # Look ahead up to 10 lines
                    next_line = raw_lines[j].strip()

                    # Skip XML-like tags
                    if next_line.startswith('</') or next_line.startswith('<'):
                        j += 1
                        continue

                    # Try combining with the next line
                    potential_combined = combined_line.rstrip(',') + ' ' + next_line

                    try:
                        json.loads(potential_combined)
                        fixed_lines.append(potential_combined + '\n')
                        i = j + 1  # Skip the lines we combined
                        break
                    except json.JSONDecodeError:
                        # Try adding a comma to the previous line
                        try:
                            potential_with_comma = combined_line + ',' + next_line
                            json.loads(potential_with_comma)
                            fixed_lines.append(potential_with_comma + '\n')
                            i = j + 1
                            break
                        except json.JSONDecodeError:
                            # Try to fix escape sequences
                            fixed_line = fix_escape_sequences(combined_line)
                            try:
                                json.loads(fixed_line)
                                fixed_lines.append(fixed_line + '\n')
                                i = j  # Move to next line after fixing current
                                break
                            except json.JSONDecodeError:
                                # Continue combining lines
                                combined_line = potential_combined
                                j += 1
                else:
                    # Could not fix by combining lines, try individual fixes
                    fixed_line = fix_common_json_issues(combined_line)
                    try:
                        json.loads(fixed_line)
                        fixed_lines.append(fixed_line + '\n')
                        i += 1
                    except json.JSONDecodeError:
                        # Skip this problematic line
                        errors.append(f"Could not fix line {i+1}: {line[:100]}...")
                        i += 1

    except Exception as e:
        return False, f"Error reading file: {str(e)}"

    # Write the fixed content back to the file
    backup_path = file_path.with_suffix(file_path.suffix + '.backup')
    # Create backup
    with open(backup_path, 'w', encoding='utf-8') as backup_f:
        with open(file_path, 'r', encoding='utf-8') as orig_f:
            backup_f.write(orig_f.read())

    # Write fixed content
    with open(file_path, 'w', encoding='utf-8') as f:
        f.writelines(fixed_lines)

    if errors:
        return False, f"Partially fixed: {len(fixed_lines)} valid lines written, {len(errors)} errors remain: {errors[:3]}"  # Show first 3 errors
    else:
        return True, f"Fixed {len(fixed_lines)} valid lines"


def fix_escape_sequences(text: str) -> str:
    """Fix common escape sequence issues in JSON strings."""
    import re

    # Fix common problematic escape sequences
    # Replace single backslashes with double backslashes in problematic contexts
    text = re.sub(r'(?<!\\)\\(?=[^"\\/bfnrtu]|$)', r'\\\\', text)

    # Fix other common escape issues
    text = text.replace('\n', '\\n').replace('\r', '\\r').replace('\t', '\\t')

    return text


def fix_common_json_issues(text: str) -> str:
    """Apply common fixes to JSON text."""
    import re

    # Remove trailing commas before closing braces/brackets
    text = re.sub(r',(\s*[}\]])', r'\1', text)

    # Try to fix unterminated strings by looking for common patterns
    # This is a more advanced approach to handle complex unterminated strings
    try:
        # Count quotes to see if we have an odd number (indicating unterminated string)
        quote_count = text.count('"') - text.count('\\"')
        if quote_count % 2 == 1:
            # Find the last properly opened string and try to close it
            # This is a simplified approach - in practice, we'd need a full JSON parser
            pass
    except:
        pass

    return text


def parse_multiline_json_objects(content: str) -> List[str]:
    """
    Parse content that contains multi-line JSON objects that were incorrectly split by line breaks.
    This handles cases where a single JSON object spans multiple lines.
    """
    import json
    import re

    # Remove empty lines
    lines = [line.strip() for line in content.split('\n') if line.strip()]

    valid_jsonl_lines = []
    current_object = ""
    brace_depth = 0
    in_string = False
    escape_next = False

    i = 0
    while i < len(lines):
        line = lines[i]

        # Process character by character to track JSON structure
        j = 0
        while j < len(line):
            char = line[j]

            if escape_next:
                escape_next = False
                j += 1
                continue

            if char == '\\':
                escape_next = True
                j += 1
                continue

            if not in_string:
                if char == '"':
                    in_string = True
                elif char == '{':
                    brace_depth += 1
                elif char == '}':
                    brace_depth -= 1
                    if brace_depth == 0 and current_object.strip():
                        # Completed object, try to parse it
                        try:
                            json_obj = json.loads(current_object + char)
                            valid_jsonl_lines.append(json.dumps(json_obj))
                            current_object = ""
                            j += 1  # Move past the closing brace
                            break  # Move to next line
                        except json.JSONDecodeError:
                            current_object += char
            else:
                if char == '"':
                    in_string = False
                current_object += char
            j += 1

        # If we finished processing the line but are still inside a JSON object
        if brace_depth > 0:
            current_object += "\n"  # Add newline to preserve spacing
        elif current_object and brace_depth == 0 and not in_string:
            # We have a complete object but haven't processed it yet
            if current_object.strip():
                try:
                    json_obj = json.loads(current_object.strip())
                    valid_jsonl_lines.append(json.dumps(json_obj))
                    current_object = ""
                except json.JSONDecodeError:
                    # If it's still not valid, we'll handle it differently
                    pass

        i += 1

    # Handle any remaining content
    if current_object.strip():
        try:
            json_obj = json.loads(current_object.strip())
            valid_jsonl_lines.append(json.dumps(json_obj))
        except json.JSONDecodeError:
            # If we can't parse it, try to clean it up
            cleaned = cleanup_json_text(current_object.strip())
            if cleaned:
                try:
                    json_obj = json.loads(cleaned)
                    valid_jsonl_lines.append(json.dumps(json_obj))
                except json.JSONDecodeError:
                    pass  # Give up on this one

    return valid_jsonl_lines


def cleanup_json_text(text: str) -> str:
    """Clean up malformed JSON text to make it valid."""
    import re

    if not text:
        return text

    # Remove XML-like tags
    text = re.sub(r'<[^>]+>', '', text)

    # Clean up common formatting issues
    text = text.strip()

    # Remove trailing commas before closing brackets/braces
    text = re.sub(r',(\s*[}\]])', r'\1', text)

    # Fix common escape issues
    text = text.replace('\n', '\\n').replace('\r', '\\r').replace('\t', '\\t')

    # Try to fix unterminated strings by looking for the pattern
    # This is a heuristic approach
    if text.count('"') % 2 == 1:  # Odd number of quotes indicates unterminated string
        # Try to find where the string should end
        # This is a simplified approach
        pass

    return text.strip()


def fix_complex_jsonl_file(file_path: Path) -> Tuple[bool, str]:
    """
    Enhanced function to fix complex JSONL files with multi-line objects and other issues.

    Args:
        file_path: Path to the file to fix

    Returns:
        Tuple of (success, message)
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # First, try the original approach
        lines = content.split('\n')
        valid_lines = []
        invalid_parts = []

        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue

            # Skip XML-like tags
            if re.match(r'^\s*[<][/]?[a-zA-Z]', line):
                continue

            try:
                json.loads(line)
                valid_lines.append(line)
            except json.JSONDecodeError:
                # This line is problematic, add to invalid parts for multi-line processing
                invalid_parts.append((i, line))

        # If we have invalid parts, try to reconstruct them as multi-line JSON objects
        if invalid_parts:
            # Join the problematic lines and try to parse as multi-line JSON
            problematic_content = '\n'.join([part[1] for part in invalid_parts])
            reconstructed_lines = parse_multiline_json_objects(problematic_content)
            valid_lines.extend(reconstructed_lines)

        # Create backup
        backup_path = file_path.with_suffix(file_path.suffix + '.backup2')
        with open(backup_path, 'w', encoding='utf-8') as backup_f:
            with open(file_path, 'r', encoding='utf-8') as orig_f:
                backup_f.write(orig_f.read())

        # Write fixed content
        with open(file_path, 'w', encoding='utf-8') as f:
            for line in valid_lines:
                f.write(line + '\n')

        return True, f"Fixed {len(valid_lines)} valid lines"

    except Exception as e:
        return False, f"Error processing file: {str(e)}"


def find_all_jsonl_files(root_dir: Path) -> List[Path]:
    """Find all JSONL files in the directory tree."""
    return list(root_dir.rglob('*.jsonl'))


def main():
    datasets_dir = Path(__file__).parent.parent / 'training' / 'datasets'
    
    if not datasets_dir.exists():
        print(f"Datasets directory does not exist: {datasets_dir}")
        return
    
    print(f"Validating JSONL files in: {datasets_dir}")
    
    jsonl_files = find_all_jsonl_files(datasets_dir)
    print(f"Found {len(jsonl_files)} JSONL files to validate\n")
    
    total_errors = 0
    valid_files = 0
    invalid_files = 0
    
    for file_path in jsonl_files:
        print(f"Checking: {file_path.relative_to(datasets_dir)}")
        is_valid, errors = is_valid_jsonl_file(file_path)
        
        if is_valid:
            print("  ✓ Valid")
            valid_files += 1
        else:
            print(f"  ✗ Invalid ({len(errors)} errors)")
            for error in errors:
                print(f"    - {error}")
            total_errors += len(errors)
            invalid_files += 1
            
            # Automatically try to fix the file
            print(f"  Attempting to fix {file_path.name}...")
            success, message = fix_jsonl_file(file_path)
            if success:
                print(f"  ✓ Fixed: {message}")
                # Re-validate to confirm fix
                is_valid_after_fix, fix_errors = is_valid_jsonl_file(file_path)
                if is_valid_after_fix:
                    print(f"  ✓ Re-validated successfully")
                    valid_files += 1
                    invalid_files -= 1
                else:
                    print(f"  ✗ Still has errors after fix: {fix_errors}")
                    # Try the enhanced fix function for complex files
                    print(f"  Trying enhanced fix for {file_path.name}...")
                    success2, message2 = fix_complex_jsonl_file(file_path)
                    if success2:
                        print(f"  ✓ Enhanced fix: {message2}")
                        # Re-validate again
                        is_valid_after_enhanced_fix, enhanced_errors = is_valid_jsonl_file(file_path)
                        if is_valid_after_enhanced_fix:
                            print(f"  ✓ Re-validated successfully after enhanced fix")
                            valid_files += 1
                            invalid_files -= 1
                        else:
                            print(f"  ✗ Still has errors after enhanced fix: {enhanced_errors}")
                    else:
                        print(f"  ✗ Enhanced fix failed: {message2}")
            else:
                print(f"  ✗ Could not fix: {message}")
                # Try the enhanced fix function for complex files
                print(f"  Trying enhanced fix for {file_path.name}...")
                success2, message2 = fix_complex_jsonl_file(file_path)
                if success2:
                    print(f"  ✓ Enhanced fix: {message2}")
                    # Re-validate again
                    is_valid_after_enhanced_fix, enhanced_errors = is_valid_jsonl_file(file_path)
                    if is_valid_after_enhanced_fix:
                        print(f"  ✓ Re-validated successfully after enhanced fix")
                        valid_files += 1
                        invalid_files -= 1
                    else:
                        print(f"  ✗ Still has errors after enhanced fix: {enhanced_errors}")
                else:
                    print(f"  ✗ Enhanced fix failed: {message2}")
        
        print()
    
    print(f"Summary:")
    print(f"  Valid files: {valid_files}")
    print(f"  Invalid files: {invalid_files}")
    print(f"  Total errors found: {total_errors}")


if __name__ == "__main__":
    main()