#!/usr/bin/env python3
"""
Convert historical ByteLogic training examples to the think/code/output design format.

The old format used a `<computation>` tag with a custom ByteLogic DSL.
This script:
1. Reads JSONL files from history/training/datasets/
2. Extracts the query and category
3. Re-implements the logic in Python, verifies the output with PythonExecutor
4. Writes new examples to training/datasets/design/basic.jsonl

We focus on convertible categories:
  - mathematical_computation  → Python arithmetic
  - graph_algorithms          → Python BFS/DFS
  - string_processing         → Python string ops
  - loop_constructs           → Python iteration
  - conditional_logic         → Python conditionals

Bytelogic-specific patterns (RULE/SCAN/EMIT, relational algebra) are adapted
to use dicts and sets in Python — the closest equivalent.

Usage:
    python training/scripts/convert_blueprint_to_design.py
    python training/scripts/convert_blueprint_to_design.py --output training/datasets/design/basic.jsonl
    python training/scripts/convert_blueprint_to_design.py --limit 200
"""

import sys
import re
import json
import argparse
import random
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.executor.python_exec import PythonExecutor


def run_code(executor: PythonExecutor, code: str) -> str:
    result = executor.run(code)
    return result.output_text().strip()


# ---------------------------------------------------------------------------
# Pattern extractors: pull structured info from ByteLogic output strings
# ---------------------------------------------------------------------------

def extract_result(output_str: str):
    """Try to extract the final numerical/boolean result from a ByteLogic output."""
    # → 42 at the end
    m = re.search(r'→\s*(.+)$', output_str.strip())
    if m:
        return m.group(1).strip()
    return None


def extract_facts(output_str: str):
    """Extract FACT lines: FACT rel a b → {rel: [(a,b), ...]}}"""
    facts = {}
    for m in re.finditer(r'FACT\s+(\w+)\s+(\S+)(?:\s+(\S+))?', output_str):
        rel, a, b = m.group(1), m.group(2), m.group(3)
        facts.setdefault(rel, []).append((a, b) if b else (a,))
    return facts


def extract_graph_edges(output_str: str):
    """Extract FACT edge a b → [(a,b), ...]"""
    return [(m.group(1), m.group(2))
            for m in re.finditer(r'FACT\s+edge\s+(\w+)\s+(\w+)', output_str)]


# ---------------------------------------------------------------------------
# Converters per category
# ---------------------------------------------------------------------------

def convert_math(ex: dict) -> dict | None:
    """Convert mathematical_computation examples."""
    query = ex["input"]
    output_str = ex["output"]
    result = extract_result(output_str)
    if result is None:
        return None

    # Try to figure out what operation this is
    subcat = ex["metadata"].get("subcategory", "")
    meta_id = ex["metadata"].get("id", "")

    executor = PythonExecutor()

    # Power calculation
    if "power" in subcat or "power" in query.lower():
        m = re.search(r'(\d+)\s+to\s+the\s+power\s+of\s+(\d+)', query.lower())
        if not m:
            m = re.search(r'(\d+)\s*\*\*\s*(\d+)', query)
        if m:
            base, exp = int(m.group(1)), int(m.group(2))
            code = f"""\
base = {base}
exp  = {exp}
result = base ** exp
print(f"{{base}}^{{exp}} = {{result}}")
"""
            out = run_code(executor, code)
            return {
                "id": f"design_{meta_id}",
                "category": "design",
                "difficulty": ex["metadata"].get("difficulty", "basic"),
                "query": query,
                "response": (
                    f"<think>\nCompute {base}^{exp} using exponentiation.\n</think>\n"
                    f"<code>\n{code}</code>\n"
                    f"<output>\n{out}\n</output>\n"
                    f"{base} to the power of {exp} is **{base**exp}**."
                ),
            }

    # Average / mean
    if "average" in subcat or "average" in query.lower() or "mean" in query.lower():
        numbers = [int(x) for x in re.findall(r'\b\d+\b', query) if int(x) < 10000]
        if len(numbers) >= 2:
            code = f"""\
numbers = {numbers}
avg = sum(numbers) / len(numbers)
print(f"Numbers: {{numbers}}")
print(f"Sum: {{sum(numbers)}}")
print(f"Count: {{len(numbers)}}")
print(f"Average: {{avg:.4f}}")
"""
            out = run_code(executor, code)
            return {
                "id": f"design_{meta_id}",
                "category": "design",
                "difficulty": ex["metadata"].get("difficulty", "basic"),
                "query": query,
                "response": (
                    f"<think>\nAverage = sum / count.\n</think>\n"
                    f"<code>\n{code}</code>\n"
                    f"<output>\n{out}\n</output>\n"
                    f"The average is **{sum(numbers)/len(numbers):.4f}**."
                ),
            }

    # Factorial
    if "factorial" in subcat or "factorial" in query.lower():
        m = re.search(r'factorial\s+of\s+(\d+)|(\d+)!|(\d+)\s+factorial', query.lower())
        if m:
            n = int(next(x for x in m.groups() if x))
            code = f"""\
import math
n = {n}
result = math.factorial(n)
# Also show iterative approach
product = 1
for i in range(1, n+1):
    product *= i
print(f"{{n}}! = {{result}}")
print(f"Verification: {{product}}")
"""
            out = run_code(executor, code)
            import math
            return {
                "id": f"design_{meta_id}",
                "category": "design",
                "difficulty": ex["metadata"].get("difficulty", "basic"),
                "query": query,
                "response": (
                    f"<think>\n{n}! = {n} × {n-1} × ... × 1\n</think>\n"
                    f"<code>\n{code}</code>\n"
                    f"<output>\n{out}\n</output>\n"
                    f"{n}! = **{math.factorial(n)}**"
                ),
            }

    return None


def convert_graph(ex: dict) -> dict | None:
    """Convert graph_algorithms examples (reachability, shortest path)."""
    query = ex["input"]
    output_str = ex["output"]
    meta_id = ex["metadata"].get("id", "")
    result = extract_result(output_str)
    edges = extract_graph_edges(output_str)

    if not edges:
        return None

    # Try to extract source/target from query
    subcat = ex["metadata"].get("subcategory", "")
    executor = PythonExecutor()

    if "reachab" in subcat.lower() or "reach" in query.lower():
        # Extract from/to nodes
        m = re.search(r'(?:from|reach)\s+node\s+(\w+)\s+(?:to|from)\s+node\s+(\w+)|'
                      r'reach\s+(\w+)\s+from\s+(\w+)', query.lower())
        if not m:
            return None
        groups = [g for g in m.groups() if g]
        if len(groups) < 2:
            return None
        # Determine src/target order from query
        src = groups[0] if "from" in query.lower().split("node")[0] else groups[1]
        tgt = groups[1] if src == groups[0] else groups[0]
        # Normalise case
        src = src.lower(); tgt = tgt.lower()
        edges_list = [(a.lower(), b.lower()) for a,b in edges]
        edges_code = str(edges_list)
        code = f"""\
from collections import deque

edges = {edges_code}
graph = {{}}
for u, v in edges:
    graph.setdefault(u, []).append(v)
    graph.setdefault(v, []).append(u)

def is_reachable(graph, src, tgt):
    visited = set()
    queue   = deque([src])
    while queue:
        node = queue.popleft()
        if node == tgt:
            return True
        if node not in visited:
            visited.add(node)
            queue.extend(graph.get(node, []))
    return False

src, tgt = {repr(src)}, {repr(tgt)}
result = is_reachable(graph, src, tgt)
print(f"Edges: {{edges}}")
print(f"Is {{tgt!r}} reachable from {{src!r}}? {{result}}")
"""
        out = run_code(executor, code)
        return {
            "id": f"design_{meta_id}",
            "category": "design",
            "difficulty": ex["metadata"].get("difficulty", "basic"),
            "query": query,
            "response": (
                "<think>\n"
                f"Graph reachability: BFS from '{src}', check if '{tgt}' is visited.\n"
                "</think>\n"
                f"<code>\n{code}</code>\n"
                f"<output>\n{out}\n</output>\n"
            ),
        }

    return None


def convert_string(ex: dict) -> dict | None:
    """Convert string_processing examples."""
    query = ex["input"]
    meta_id = ex["metadata"].get("id", "")
    subcat = ex["metadata"].get("subcategory", "")
    executor = PythonExecutor()

    # String reversal
    if "revers" in subcat.lower() or "revers" in query.lower():
        m = re.search(r'reverse\s+(?:the\s+)?(?:string\s+)?["\']?([a-zA-Z0-9_]+)["\']?', query.lower())
        if m:
            s = m.group(1)
            code = f"""\
s = {repr(s)}
rev = s[::-1]
print(f"Original: {{s!r}}")
print(f"Reversed: {{rev!r}}")
"""
            out = run_code(executor, code)
            return {
                "id": f"design_{meta_id}",
                "category": "design",
                "difficulty": ex["metadata"].get("difficulty", "basic"),
                "query": query,
                "response": (
                    "<think>\nString reversal: slice with step -1.\n</think>\n"
                    f"<code>\n{code}</code>\n"
                    f"<output>\n{out}\n</output>\n"
                    f"The reversed string is **{s[::-1]!r}**."
                ),
            }

    # String length
    if "length" in subcat.lower() or "length" in query.lower() or "how many char" in query.lower():
        m = re.search(r'["\']([a-zA-Z0-9 _]+)["\']', query)
        if m:
            s = m.group(1)
            code = f"""\
s = {repr(s)}
print(f"String: {{s!r}}")
print(f"Length: {{len(s)}} characters")
"""
            out = run_code(executor, code)
            return {
                "id": f"design_{meta_id}",
                "category": "design",
                "difficulty": ex["metadata"].get("difficulty", "basic"),
                "query": query,
                "response": (
                    "<think>\nlen() returns the number of characters.\n</think>\n"
                    f"<code>\n{code}</code>\n"
                    f"<output>\n{out}\n</output>\n"
                    f"The string has **{len(s)}** characters."
                ),
            }

    # Palindrome check
    if "palindrome" in query.lower():
        m = re.search(r'["\']([a-zA-Z0-9 _]+)["\']', query)
        if m:
            s = m.group(1)
            code = f"""\
s = {repr(s)}
is_pal = s.lower() == s.lower()[::-1]
print(f"{{s!r}} is{'{}' if is_pal else ' not'} a palindrome")
"""
            out = run_code(executor, code)
            return {
                "id": f"design_{meta_id}",
                "category": "design",
                "difficulty": ex["metadata"].get("difficulty", "basic"),
                "query": query,
                "response": (
                    "<think>\nPalindrome: string equals its own reverse.\n</think>\n"
                    f"<code>\n{code}</code>\n"
                    f"<output>\n{out}\n</output>\n"
                ),
            }

    return None


def convert_loop(ex: dict) -> dict | None:
    """Convert loop_constructs examples (sum, product, range operations)."""
    query = ex["input"]
    meta_id = ex["metadata"].get("id", "")
    subcat = ex["metadata"].get("subcategory", "")
    executor = PythonExecutor()

    # Sum 1 to N
    m = re.search(r'sum\s+(?:of\s+)?(?:numbers\s+)?(?:from\s+)?1\s+to\s+(\d+)', query.lower())
    if m:
        n = int(m.group(1))
        code = f"""\
n = {n}
# Iterative
total = sum(range(1, n+1))
# Gauss formula
gauss = n * (n+1) // 2
print(f"Sum 1 to {{n}}: {{total}}")
print(f"Gauss formula n(n+1)/2: {{gauss}} ✓")
"""
        out = run_code(executor, code)
        return {
            "id": f"design_{meta_id}",
            "category": "design",
            "difficulty": ex["metadata"].get("difficulty", "basic"),
            "query": query,
            "response": (
                "<think>\nSum 1..n iteratively, also verify with Gauss formula n(n+1)/2.\n</think>\n"
                f"<code>\n{code}</code>\n"
                f"<output>\n{out}\n</output>\n"
                f"Sum from 1 to {n} = **{n*(n+1)//2}**."
            ),
        }

    # Sum of even numbers
    m = re.search(r'sum\s+(?:of\s+)?even\s+numbers\s+(?:from\s+)?1\s+to\s+(\d+)', query.lower())
    if m:
        n = int(m.group(1))
        code = f"""\
n = {n}
total = sum(x for x in range(1, n+1) if x % 2 == 0)
evens = [x for x in range(1, n+1) if x % 2 == 0]
print(f"Even numbers from 1 to {{n}}: {{evens}}")
print(f"Sum: {{total}}")
"""
        out = run_code(executor, code)
        return {
            "id": f"design_{meta_id}",
            "category": "design",
            "difficulty": ex["metadata"].get("difficulty", "basic"),
            "query": query,
            "response": (
                "<think>\nFilter even numbers with x % 2 == 0, then sum.\n</think>\n"
                f"<code>\n{code}</code>\n"
                f"<output>\n{out}\n</output>\n"
            ),
        }

    # Product / multiplication
    m = re.search(r'product\s+(?:of\s+)?(?:numbers\s+)?(?:from\s+)?(\d+)\s+to\s+(\d+)', query.lower())
    if m:
        lo, hi = int(m.group(1)), int(m.group(2))
        code = f"""\
import math
lo, hi = {lo}, {hi}
result = math.prod(range(lo, hi+1))
print(f"Product {{lo}} to {{hi}} = {{result}}")
"""
        out = run_code(executor, code)
        return {
            "id": f"design_{meta_id}",
            "category": "design",
            "difficulty": ex["metadata"].get("difficulty", "basic"),
            "query": query,
            "response": (
                "<think>\nProduct of range using math.prod.\n</think>\n"
                f"<code>\n{code}</code>\n"
                f"<output>\n{out}\n</output>\n"
            ),
        }

    return None


CONVERTERS = {
    "mathematical_computation": convert_math,
    "graph_algorithms": convert_graph,
    "string_processing": convert_string,
    "loop_constructs": convert_loop,
}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def load_bytelogic_examples(base_dir: Path) -> list:
    examples = []
    for path in sorted(base_dir.rglob("*.jsonl")):
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    examples.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return examples


def convert_examples(raw: list, limit: int) -> list:
    converted = []
    seen_ids = set()
    random.shuffle(raw)

    for ex in raw:
        if len(converted) >= limit:
            break
        cat = ex.get("metadata", {}).get("category", "")
        converter = CONVERTERS.get(cat)
        if not converter:
            continue
        try:
            result = converter(ex)
        except Exception as e:
            continue
        if result is None:
            continue
        ex_id = result["id"]
        if ex_id in seen_ids:
            continue
        # Verify: response must have at least one code+output pair
        resp = result.get("response", "")
        if "<code>" not in resp or "<output>" not in resp:
            continue
        seen_ids.add(ex_id)
        converted.append(result)
        if len(converted) % 20 == 0:
            print(f"  Converted {len(converted)} examples...")

    return converted


def main():
    parser = argparse.ArgumentParser(
        description="Convert ByteLogic training data to think/code/output format"
    )
    parser.add_argument("--input",  default="history/training/datasets")
    parser.add_argument("--output", default="training/datasets/design/basic.jsonl")
    parser.add_argument("--limit",  type=int, default=200)
    args = parser.parse_args()

    input_dir = Path(args.input)
    out_path  = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading ByteLogic examples from {input_dir}...")
    raw = load_bytelogic_examples(input_dir)
    print(f"Loaded {len(raw)} raw examples")

    print(f"Converting (limit={args.limit})...")
    converted = convert_examples(raw, args.limit)

    with open(out_path, "w", encoding="utf-8") as f:
        for ex in converted:
            f.write(json.dumps(ex) + "\n")

    print(f"\nWrote {len(converted)} converted examples to {out_path}")

    # Show category breakdown
    cats = {}
    for ex in converted:
        # category is "design" for all but subcategory comes from original
        cats[ex.get("category", "?")] = cats.get(ex.get("category", "?"), 0) + 1
    for k, v in sorted(cats.items()):
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
