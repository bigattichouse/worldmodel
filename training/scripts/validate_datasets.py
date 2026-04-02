#!/usr/bin/env python3
"""
Validate all training datasets.

Checks every JSONL file in training/datasets/ for:
- Required fields (id, category, difficulty, query, response)
- Matching <code> and <output> blocks (code without output is invalid)
- Non-empty content
- Executability of code blocks (optional, slow)

Usage:
    python training/scripts/validate_datasets.py
    python training/scripts/validate_datasets.py --check-exec  # also re-run code
    python training/scripts/validate_datasets.py --path training/datasets/arithmetic/
"""

import sys
import json
import argparse
import re
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.executor.python_exec import run_once


def check_example(ex: dict, path: Path, lineno: int, check_exec: bool = False) -> list:
    errors = []
    prefix = f"{path.name}:{lineno} [{ex.get('id', '?')}]"

    # Required fields
    for field in ("id", "category", "query", "response"):
        if field not in ex or not str(ex[field]).strip():
            errors.append(f"{prefix}: missing or empty '{field}'")

    response = ex.get("response", "")

    # Count code and output blocks
    code_blocks = re.findall(r"<code>(.*?)</code>", response, re.DOTALL)
    output_blocks = re.findall(r"<output>(.*?)</output>", response, re.DOTALL)

    if len(code_blocks) != len(output_blocks):
        errors.append(
            f"{prefix}: {len(code_blocks)} <code> blocks but {len(output_blocks)} <output> blocks"
        )

    # Each output block should be non-empty
    for i, out in enumerate(output_blocks):
        if not out.strip():
            errors.append(f"{prefix}: <output> block {i+1} is empty")

    # Optional: re-execute code and compare
    if check_exec and code_blocks:
        from src.executor.python_exec import PythonExecutor
        exec_shared = PythonExecutor()
        for i, (code, expected_out) in enumerate(zip(code_blocks, output_blocks)):
            result = exec_shared.run(code.strip())
            actual = result.output_text().strip()
            expected = expected_out.strip()
            if actual != expected:
                errors.append(
                    f"{prefix}: code block {i+1} output mismatch\n"
                    f"  expected: {repr(expected[:80])}\n"
                    f"  actual:   {repr(actual[:80])}"
                )

    return errors


def validate_file(path: Path, check_exec: bool = False) -> tuple:
    """Returns (total_examples, errors_list)."""
    all_errors = []
    total = 0

    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                ex = json.loads(line)
                total += 1
                errors = check_example(ex, path, lineno, check_exec)
                all_errors.extend(errors)
            except json.JSONDecodeError as e:
                all_errors.append(f"{path.name}:{lineno}: JSON decode error: {e}")

    return total, all_errors


def main():
    parser = argparse.ArgumentParser(description="Validate WorldModel training datasets")
    parser.add_argument("--path", type=str, default="training/datasets",
                        help="Path to dataset directory or file")
    parser.add_argument("--check-exec", action="store_true",
                        help="Re-execute code blocks and verify outputs (slow)")
    args = parser.parse_args()

    root = Path(args.path)
    if root.is_file():
        paths = [root]
    else:
        paths = sorted(root.rglob("*.jsonl"))

    if not paths:
        print(f"No JSONL files found in {root}")
        sys.exit(1)

    total_examples = 0
    total_errors = []
    file_summary = []

    for path in paths:
        count, errors = validate_file(path, args.check_exec)
        total_examples += count
        total_errors.extend(errors)
        status = "OK" if not errors else f"{len(errors)} ERROR(s)"
        file_summary.append((path, count, status))

    # Print summary
    print(f"\n{'='*60}")
    print(f"Dataset Validation Summary")
    print(f"{'='*60}")
    for path, count, status in file_summary:
        try:
            rel = str(path.relative_to(Path("training/datasets")))
        except ValueError:
            rel = str(path)
        print(f"  {rel:<50} {count:>4} examples  {status}")

    print(f"{'─'*60}")
    print(f"  Total: {total_examples} examples across {len(paths)} files")

    if total_errors:
        print(f"\n{'='*60}")
        print(f"ERRORS ({len(total_errors)}):")
        for err in total_errors:
            print(f"  {err}")
        sys.exit(1)
    else:
        print(f"\nAll examples valid.")


if __name__ == "__main__":
    main()
