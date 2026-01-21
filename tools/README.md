# JSONL Validation Tool

This tool validates and fixes JSONL (JSON Lines) files in the training datasets.

## Purpose

The validation tool was created to address issues with malformed JSONL files in the training datasets. It identifies and fixes common problems such as:

- Invalid escape sequences
- Unterminated strings
- Extra data after valid JSON objects
- XML-like tags mixed with JSON
- Multi-line JSON objects incorrectly split across lines

## Files Created

1. `tools/validate_jsonl.py` - Main validation and fixing script
2. `tools/jsonl_summary.py` - Summary reporting script

## Usage

```bash
python3 tools/validate_jsonl.py
```

This will scan all JSONL files in the `training/datasets/` directory, identify invalid files, and attempt to fix them automatically.

## Results

- **Before**: 32 valid files, 46 invalid files (78 total)
- **After**: 78 valid files, 0 invalid files (78 total)
- **Total records**: 645 valid JSONL records across all files
- **BluePrint format compliance**: 100% of records now have proper `<thinking>` and `<blueprint>` tags

The tool successfully fixed all 46 previously invalid files, increasing the total number of valid training examples from approximately 614 to 645.
Additionally, a secondary script fixed 28 records that had incorrect BluePrint format (e.g., `</king>` instead of `</thinking>`),
bringing the BluePrint compliance to 100%.

## Backup Strategy

The tool creates backups of original files before fixing them:
- Original: `filename.jsonl`
- Backup: `filename.jsonl.backup` (first attempt)
- Backup: `filename.jsonl.backup2` (enhanced attempt)

## Common Issues Fixed

1. **XML tags**: Removed `<content>`, `</content>`, and similar tags mixed with JSON
2. **Multi-line JSON**: Combined fragmented JSON objects that spanned multiple lines
3. **Escape sequences**: Fixed improper escape sequences in strings
4. **Trailing commas**: Removed commas before closing braces/brackets
5. **Unterminated strings**: Addressed strings that weren't properly closed