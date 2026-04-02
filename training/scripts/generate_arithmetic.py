#!/usr/bin/env python3
"""
Generate verified arithmetic training examples.

All <output> blocks are produced by actually running the code,
so they are guaranteed correct. No hallucinated answers.

Usage:
    python training/scripts/generate_arithmetic.py --output training/datasets/arithmetic/basic.jsonl --count 100
"""

import sys
import json
import random
import argparse
from pathlib import Path

# Allow importing src/
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.executor.python_exec import run_once


def execute_and_capture(code: str) -> str:
    """Run code and return the output text."""
    result = run_once(code)
    return result.output_text()


def make_example(ex_id: str, category: str, difficulty: str,
                 query: str, think: str, model_text: str, code: str) -> dict:
    """
    Build a complete example by executing the code to get real output.
    """
    output = execute_and_capture(code)

    parts = []
    if think:
        parts.append(f"<think>\n{think.strip()}\n</think>")
    if model_text:
        parts.append(f"<model>\n{model_text.strip()}\n</model>")
    parts.append(f"<code>\n{code.strip()}\n</code>")
    parts.append(f"<output>\n{output.strip()}\n</output>")

    # Build a natural language conclusion using the output
    # We'll make a simple one — manual examples will have better prose
    conclusion = f"Result: {output.strip()}"

    response = "\n".join(parts) + "\n" + conclusion

    return {
        "id": ex_id,
        "category": category,
        "difficulty": difficulty,
        "query": query,
        "response": response,
    }


# ─── Problem generators ───────────────────────────────────────────────────────

def gen_basic_arithmetic(rng: random.Random) -> dict:
    ops = [
        ("+", "addition", lambda a, b: f"print({a} + {b})"),
        ("-", "subtraction", lambda a, b: f"print({a} - {b})"),
        ("*", "multiplication", lambda a, b: f"print({a} * {b})"),
    ]
    op_sym, op_name, code_fn = rng.choice(ops)
    a = rng.randint(2, 999)
    b = rng.randint(2, 999)
    query = f"What is {a} {op_sym} {b}?"
    think = f"This is a simple {op_name} problem."
    code = code_fn(a, b)
    return query, think, "", code


def gen_percentage(rng: random.Random) -> dict:
    pct = rng.choice([5, 10, 15, 20, 25, 30, 33, 40, 50, 60, 75, 80])
    base = rng.randint(50, 5000)
    # round base to clean number
    base = round(base / 10) * 10
    query = f"What is {pct}% of {base}?"
    think = f"Percentage: multiply {base} by {pct}/100."
    code = f"result = {base} * {pct} / 100\nprint(f'{pct}% of {base} = {{result}}')"
    return query, think, "", code


def gen_compound_interest(rng: random.Random) -> dict:
    principal = rng.choice([500, 1000, 2000, 5000, 10000])
    rate = rng.choice([0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.10])
    years = rng.randint(2, 10)
    rate_pct = int(rate * 100)
    query = f"What is the compound interest on ${principal:,} at {rate_pct}% annual rate for {years} years?"
    think = "Use compound interest formula: A = P(1 + r)^t, interest = A - P."
    code = (
        f"P = {principal}\n"
        f"r = {rate}\n"
        f"t = {years}\n"
        f"A = P * (1 + r) ** t\n"
        f"interest = A - P\n"
        f"print(f'Final amount: ${{A:.2f}}')\n"
        f"print(f'Interest earned: ${{interest:.2f}}')"
    )
    return query, think, "", code


def gen_area(rng: random.Random) -> dict:
    shape = rng.choice(["rectangle", "triangle", "circle"])
    if shape == "rectangle":
        w = rng.randint(2, 50)
        h = rng.randint(2, 50)
        query = f"What is the area of a rectangle with width {w} and height {h}?"
        think = "Area of rectangle = width × height."
        code = f"area = {w} * {h}\nprint(f'Area = {{area}} square units')"
    elif shape == "triangle":
        base = rng.randint(3, 40)
        height = rng.randint(3, 40)
        query = f"What is the area of a triangle with base {base} and height {height}?"
        think = "Area of triangle = (base × height) / 2."
        code = f"area = ({base} * {height}) / 2\nprint(f'Area = {{area}} square units')"
    else:
        r = rng.randint(2, 20)
        query = f"What is the area of a circle with radius {r}?"
        think = "Area of circle = π × r²."
        code = f"import math\narea = math.pi * {r}**2\nprint(f'Area = {{area:.4f}} square units')"
    return query, think, "", code


def gen_unit_conversion(rng: random.Random) -> dict:
    conversions = [
        ("miles", "kilometers", 1.60934, lambda v, c: f"km = {v} * {c}\nprint(f'{{km:.3f}} kilometers')"),
        ("kilograms", "pounds", 2.20462, lambda v, c: f"lbs = {v} * {c}\nprint(f'{{lbs:.3f}} pounds')"),
        ("Celsius", "Fahrenheit", None,
         lambda v, _: f"f = ({v} * 9/5) + 32\nprint(f'{{f:.1f}}°F')"),
        ("feet", "meters", 0.3048, lambda v, c: f"m = {v} * {c}\nprint(f'{{m:.4f}} meters')"),
        ("liters", "gallons", 0.264172, lambda v, c: f"gal = {v} * {c}\nprint(f'{{gal:.4f}} gallons')"),
    ]
    from_unit, to_unit, factor, code_fn = rng.choice(conversions)
    value = rng.randint(1, 200)
    query = f"Convert {value} {from_unit} to {to_unit}."
    think = f"Look up the conversion factor from {from_unit} to {to_unit} and multiply."
    code = code_fn(value, factor)
    return query, think, "", code


def gen_average(rng: random.Random) -> dict:
    n = rng.randint(4, 8)
    values = [rng.randint(10, 100) for _ in range(n)]
    values_str = str(values)
    query = f"What is the average of {values}?"
    think = "Average = sum of values / count of values."
    code = (
        f"values = {values_str}\n"
        f"avg = sum(values) / len(values)\n"
        f"print(f'Sum: {{sum(values)}}')\n"
        f"print(f'Count: {{len(values)}}')\n"
        f"print(f'Average: {{avg:.2f}}')"
    )
    return query, think, "", code


def gen_multistep_percentage(rng: random.Random) -> dict:
    """Multi-step: calculate discounted price, then apply tax."""
    original = rng.choice([100, 200, 500, 800, 1200, 2000])
    discount = rng.choice([10, 15, 20, 25, 30])
    tax = rng.choice([5, 8, 10, 12])
    query = (f"A product costs ${original}. There is a {discount}% discount, "
             f"then {tax}% tax is applied. What is the final price?")
    think = (f"Two steps: first apply the {discount}% discount, "
             f"then apply {tax}% tax to the discounted price.")

    code1 = (
        f"original = {original}\n"
        f"discount_pct = {discount}\n"
        f"discounted = original * (1 - discount_pct / 100)\n"
        f"print(f'After {discount}% discount: ${{discounted:.2f}}')"
    )
    # Execute step 1
    out1 = execute_and_capture(code1)

    code2 = (
        f"tax_pct = {tax}\n"
        f"final = discounted * (1 + tax_pct / 100)\n"
        f"print(f'After {tax}% tax: ${{final:.2f}}')"
    )
    # Execute step 2 with step 1 state
    from src.executor.python_exec import PythonExecutor
    exec_shared = PythonExecutor()
    exec_shared.run(code1)  # prime namespace
    out2 = exec_shared.run(code2).output_text()

    response = (
        f"<think>\n{think}\n</think>\n"
        f"<code>\n{code1.strip()}\n</code>\n"
        f"<output>\n{out1.strip()}\n</output>\n"
        f"<think>\nNow apply the tax to the discounted price.\n</think>\n"
        f"<code>\n{code2.strip()}\n</code>\n"
        f"<output>\n{out2.strip()}\n</output>\n"
        f"The final price after {discount}% discount and {tax}% tax is shown above."
    )

    return {
        "id": None,  # assigned by caller
        "category": "arithmetic",
        "difficulty": "intermediate",
        "query": query,
        "response": response,
    }


# ─── Main ─────────────────────────────────────────────────────────────────────

GENERATORS = [
    ("basic_arithmetic", "basic", gen_basic_arithmetic, 0.25),
    ("percentage", "basic", gen_percentage, 0.15),
    ("compound_interest", "intermediate", gen_compound_interest, 0.15),
    ("area", "basic", gen_area, 0.15),
    ("unit_conversion", "basic", gen_unit_conversion, 0.15),
    ("average", "basic", gen_average, 0.10),
]
# multistep handled separately since it returns a full example


def generate_examples(count: int, seed: int = 42) -> list:
    rng = random.Random(seed)
    examples = []
    idx = 0

    # Allocate ~15% to multistep
    multistep_count = max(1, count // 7)
    single_count = count - multistep_count

    # Single-step examples
    weights = [g[3] for g in GENERATORS]
    total_w = sum(weights)
    normalized = [w / total_w for w in weights]

    for _ in range(single_count):
        # Weighted random selection
        r = rng.random()
        cumulative = 0
        chosen = GENERATORS[0]
        for gen in GENERATORS:
            cumulative += gen[3] / total_w
            if r <= cumulative:
                chosen = gen
                break

        name, difficulty, fn = chosen[0], chosen[1], chosen[2]
        query, think, model_text, code = fn(rng)
        ex = make_example(
            ex_id=f"arith_{idx:04d}",
            category="arithmetic",
            difficulty=difficulty,
            query=query,
            think=think,
            model_text=model_text,
            code=code,
        )
        examples.append(ex)
        idx += 1

    # Multi-step examples
    for _ in range(multistep_count):
        ex = gen_multistep_percentage(rng)
        ex["id"] = f"arith_{idx:04d}"
        examples.append(ex)
        idx += 1

    rng.shuffle(examples)
    return examples


def main():
    parser = argparse.ArgumentParser(description="Generate arithmetic training examples")
    parser.add_argument("--output", required=True, help="Output JSONL file path")
    parser.add_argument("--count", type=int, default=100, help="Number of examples")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Generating {args.count} arithmetic examples...")
    examples = generate_examples(args.count, seed=args.seed)

    with open(output_path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")

    print(f"Written {len(examples)} examples to {output_path}")

    # Quick validation check
    errors = 0
    for ex in examples:
        if "<code>" in ex["response"] and "<output>" not in ex["response"]:
            print(f"WARNING: {ex['id']} has code but no output!")
            errors += 1
    if errors == 0:
        print("All examples validated OK.")


if __name__ == "__main__":
    main()
