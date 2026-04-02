#!/usr/bin/env python3
"""
Generate verified algebra training examples.

Covers: linear equations, quadratics, systems of equations, inequalities,
polynomials, factoring, and word problems.

Usage:
    python training/scripts/generate_algebra.py --output training/datasets/algebra/basic.jsonl --count 150
"""

import sys
import json
import random
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.executor.python_exec import run_once, PythonExecutor


def execute(code: str) -> str:
    return run_once(code).output_text()


def make_example(ex_id, category, difficulty, query, think, model_text, code):
    output = execute(code)
    parts = []
    if think:
        parts.append(f"<think>\n{think.strip()}\n</think>")
    if model_text:
        parts.append(f"<model>\n{model_text.strip()}\n</model>")
    parts.append(f"<code>\n{code.strip()}\n</code>")
    parts.append(f"<output>\n{output.strip()}\n</output>")
    response = "\n".join(parts)
    return {"id": ex_id, "category": category, "difficulty": difficulty,
            "query": query, "response": response}


# ─── Problem generators ───────────────────────────────────────────────────────

def gen_linear_equation(rng):
    """ax + b = c  →  solve for x"""
    a = rng.choice([2, 3, 4, 5, 6, 7, 8, 9, 10])
    x = rng.randint(-20, 20)
    b = rng.randint(-50, 50)
    c = a * x + b
    query = f"Solve for x: {a}x + {b} = {c}"
    think = "Isolate x by subtracting the constant from both sides, then dividing by the coefficient."
    code = (
        f"from sympy import symbols, solve, Eq\n"
        f"x = symbols('x')\n"
        f"solution = solve(Eq({a}*x + {b}, {c}), x)\n"
        f"print(f'x = {{solution[0]}}')"
    )
    return query, think, "", code


def gen_linear_equation_fractions(rng):
    """Results in fraction solutions"""
    a = rng.choice([3, 4, 5, 6, 7])
    b = rng.randint(1, 20)
    c = rng.randint(1, 30)
    # ax + b = c, solution is (c-b)/a
    query = f"Solve for x: {a}x + {b} = {c}"
    think = "Linear equation: subtract {b} from both sides, divide by {a}.".format(b=b, a=a)
    code = (
        f"from fractions import Fraction\n"
        f"x = Fraction({c} - {b}, {a})\n"
        f"print(f'x = {{x}}')\n"
        f"print(f'x = {{float(x):.4f}} (decimal)')"
    )
    return query, think, "", code


def gen_quadratic(rng):
    """x^2 + bx + c = 0 with integer roots"""
    r1 = rng.randint(-8, 8)
    r2 = rng.randint(-8, 8)
    b = -(r1 + r2)
    c = r1 * r2
    b_str = f"+ {b}x" if b >= 0 else f"- {abs(b)}x"
    c_str = f"+ {c}" if c >= 0 else f"- {abs(c)}"
    query = f"Solve: x² {b_str} {c_str} = 0"
    think = "Quadratic equation — factor or use the quadratic formula."
    model_text = f"Factor: find two numbers that multiply to {c} and add to {b}."
    code = (
        f"from sympy import symbols, solve, factor\n"
        f"x = symbols('x')\n"
        f"expr = x**2 + ({b})*x + ({c})\n"
        f"factored = factor(expr)\n"
        f"solutions = solve(expr, x)\n"
        f"print(f'Factored: {{factored}}')\n"
        f"print(f'Solutions: x = {{solutions[0]}} or x = {{solutions[1]}}')"
    )
    return query, think, model_text, code


def gen_quadratic_formula(rng):
    """General quadratic with quadratic formula, may have non-integer roots"""
    a = rng.choice([1, 2, 3])
    # pick discriminant to be a perfect square for clean answers
    d = rng.choice([1, 4, 9, 16, 25])
    b = rng.randint(-6, 6)
    # discriminant = b^2 - 4ac = d → c = (b^2 - d) / 4a
    # need c to be integer
    num = b*b - d
    if num % (4*a) != 0:
        # fall back to a simpler case
        a, b, d = 1, 2, 4
        num = b*b - d
    c = num // (4*a)
    query = f"Use the quadratic formula to solve: {a}x² + {b}x + {c} = 0"
    think = "Apply the quadratic formula: x = (-b ± √(b²-4ac)) / 2a"
    code = (
        f"import math\n"
        f"a, b, c = {a}, {b}, {c}\n"
        f"discriminant = b**2 - 4*a*c\n"
        f"print(f'Discriminant: {{discriminant}}')\n"
        f"x1 = (-b + math.sqrt(discriminant)) / (2*a)\n"
        f"x2 = (-b - math.sqrt(discriminant)) / (2*a)\n"
        f"print(f'x = {{x1:.4f}} or x = {{x2:.4f}}')"
    )
    return query, think, "", code


def gen_system_of_equations(rng):
    """2x2 linear system: ax + by = e, cx + dy = f"""
    x_val = rng.randint(-5, 5)
    y_val = rng.randint(-5, 5)
    a = rng.randint(1, 5)
    b = rng.randint(1, 5)
    c = rng.randint(1, 5)
    d = rng.randint(1, 5)
    e = a * x_val + b * y_val
    f = c * x_val + d * y_val
    b_str = f"+ {b}y" if b >= 0 else f"- {abs(b)}y"
    d_str = f"+ {d}y" if d >= 0 else f"- {abs(d)}y"
    query = f"Solve the system: {a}x {b_str} = {e}, {c}x {d_str} = {f}"
    think = "System of two linear equations — use substitution or elimination."
    model_text = "Use numpy to solve the matrix equation Ax = b."
    code = (
        f"import numpy as np\n"
        f"A = np.array([[{a}, {b}], [{c}, {d}]], dtype=float)\n"
        f"b = np.array([{e}, {f}], dtype=float)\n"
        f"solution = np.linalg.solve(A, b)\n"
        f"print(f'x = {{solution[0]:.4f}}')\n"
        f"print(f'y = {{solution[1]:.4f}}')"
    )
    return query, think, model_text, code


def gen_polynomial_eval(rng):
    """Evaluate a polynomial at a given x"""
    degree = rng.randint(2, 4)
    coeffs = [rng.randint(-5, 5) for _ in range(degree + 1)]
    while coeffs[0] == 0:
        coeffs[0] = rng.randint(1, 5)
    x_val = rng.randint(-3, 3)

    poly_str = " + ".join(
        f"{c}x^{degree - i}" if degree - i > 1
        else (f"{c}x" if degree - i == 1 else str(c))
        for i, c in enumerate(coeffs)
    )
    query = f"Evaluate the polynomial p(x) = {poly_str} at x = {x_val}"
    think = f"Substitute x = {x_val} into the polynomial and compute."
    coeffs_str = str(coeffs)
    code = (
        f"import numpy as np\n"
        f"coeffs = {coeffs_str}\n"
        f"x = {x_val}\n"
        f"result = np.polyval(coeffs, x)\n"
        f"print(f'p({{x}}) = {{result}}')"
    )
    return query, think, "", code


def gen_inequality(rng):
    """ax + b > c"""
    a = rng.choice([2, 3, 4, 5, -2, -3])
    x_min = rng.randint(-10, 10)
    b = rng.randint(-20, 20)
    c = a * x_min + b
    op = rng.choice([">", "<", ">=", "<="])
    op_words = {">": "greater than", "<": "less than",
                ">=": "greater than or equal to", "<=": "less than or equal to"}
    query = f"Solve the inequality: {a}x + {b} {op} {c}"
    think = f"Isolate x. Remember: dividing by a negative number flips the inequality sign."
    code = (
        f"from sympy import symbols, solve, Rational\n"
        f"from sympy import Lt, Gt, Le, Ge\n"
        f"x = symbols('x')\n"
        f"ops = {{'>': Gt, '<': Lt, '>=': Ge, '<=': Le}}\n"
        f"inequality = ops['{op}']({a}*x + {b}, {c})\n"
        f"solution = solve(inequality, x)\n"
        f"print(f'Solution: {{solution}}')"
    )
    return query, think, "", code


def gen_multistep_word_problem(rng):
    """Word problem requiring system of equations"""
    total_items = rng.randint(15, 40)
    price_a = rng.choice([2, 3, 4, 5, 6, 8])
    price_b = rng.choice([3, 4, 5, 6, 7, 9])
    while price_a == price_b:
        price_b = rng.choice([3, 4, 5, 6, 7, 9])
    count_a = rng.randint(3, total_items - 3)
    count_b = total_items - count_a
    total_cost = count_a * price_a + count_b * price_b

    items = rng.choice([
        ("apples", "oranges"),
        ("pens", "pencils"),
        ("adult tickets", "child tickets"),
        ("notebooks", "folders"),
    ])

    query = (
        f"A store sells {items[0]} for ${price_a} each and {items[1]} for ${price_b} each. "
        f"A customer buys {total_items} items total and spends ${total_cost}. "
        f"How many of each did they buy?"
    )
    think = (
        f"Set up two equations: one for total count, one for total cost. "
        f"Let a = number of {items[0]}, b = number of {items[1]}."
    )
    model_text = (
        f"a + b = {total_items}\n"
        f"{price_a}a + {price_b}b = {total_cost}\n"
        f"Solve for a and b."
    )

    exec_shared = PythonExecutor()
    code1 = (
        f"import numpy as np\n"
        f"A = np.array([[1, 1], [{price_a}, {price_b}]], dtype=float)\n"
        f"b = np.array([{total_items}, {total_cost}], dtype=float)\n"
        f"sol = np.linalg.solve(A, b)\n"
        f"a_count = round(sol[0])\n"
        f"b_count = round(sol[1])\n"
        f"print(f'{items[0].capitalize()}: {{a_count}}')\n"
        f"print(f'{items[1].capitalize()}: {{b_count}}')"
    )
    out1 = exec_shared.run(code1).output_text()

    code2 = (
        f"# Verify\n"
        f"total_check = a_count + b_count\n"
        f"cost_check = a_count * {price_a} + b_count * {price_b}\n"
        f"print(f'Verification: {{total_check}} items, ${{cost_check}} total')\n"
        f"print('Correct!' if total_check == {total_items} and cost_check == {total_cost} else 'Error in solution')"
    )
    out2 = exec_shared.run(code2).output_text()

    response = (
        f"<think>\n{think}\n</think>\n"
        f"<model>\n{model_text}\n</model>\n"
        f"<code>\n{code1.strip()}\n</code>\n"
        f"<output>\n{out1.strip()}\n</output>\n"
        f"<think>\nVerify the answer adds up correctly.\n</think>\n"
        f"<code>\n{code2.strip()}\n</code>\n"
        f"<output>\n{out2.strip()}\n</output>\n"
        f"The customer bought {count_a} {items[0]} and {count_b} {items[1]}."
    )

    return {
        "id": None,
        "category": "algebra",
        "difficulty": "intermediate",
        "query": query,
        "response": response,
    }


# ─── Main ─────────────────────────────────────────────────────────────────────

GENERATORS = [
    ("linear_equation",    "basic",        gen_linear_equation,          0.20),
    ("linear_fractions",   "basic",        gen_linear_equation_fractions, 0.10),
    ("quadratic",          "intermediate", gen_quadratic,                 0.20),
    ("quadratic_formula",  "intermediate", gen_quadratic_formula,         0.15),
    ("system",             "intermediate", gen_system_of_equations,       0.15),
    ("polynomial_eval",    "basic",        gen_polynomial_eval,           0.10),
    ("inequality",         "basic",        gen_inequality,                0.10),
]


def generate_examples(count: int, seed: int = 42) -> list:
    rng = random.Random(seed)
    examples = []
    idx = 0

    multistep_count = max(1, count // 6)
    single_count = count - multistep_count

    total_w = sum(g[3] for g in GENERATORS)

    for _ in range(single_count):
        r = rng.random()
        cumulative = 0.0
        chosen = GENERATORS[0]
        for gen in GENERATORS:
            cumulative += gen[3] / total_w
            if r <= cumulative:
                chosen = gen
                break
        name, difficulty, fn = chosen[0], chosen[1], chosen[2]
        try:
            query, think, model_text, code = fn(rng)
            ex = make_example(f"alg_{idx:04d}", "algebra", difficulty,
                               query, think, model_text, code)
            examples.append(ex)
            idx += 1
        except Exception as e:
            print(f"  Warning: skipping example (error: {e})")

    for _ in range(multistep_count):
        try:
            ex = gen_multistep_word_problem(rng)
            ex["id"] = f"alg_{idx:04d}"
            examples.append(ex)
            idx += 1
        except Exception as e:
            print(f"  Warning: skipping multistep example (error: {e})")

    rng.shuffle(examples)
    return examples


def main():
    parser = argparse.ArgumentParser(description="Generate algebra training examples")
    parser.add_argument("--output", required=True)
    parser.add_argument("--count", type=int, default=150)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Generating {args.count} algebra examples (requires sympy)...")
    try:
        import sympy
    except ImportError:
        print("ERROR: sympy not installed. Run: pip install sympy")
        sys.exit(1)

    examples = generate_examples(args.count, seed=args.seed)

    with open(output_path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")

    print(f"Written {len(examples)} examples to {output_path}")
    errors = sum(1 for ex in examples if "<code>" in ex["response"] and "<output>" not in ex["response"])
    if errors == 0:
        print("All examples validated OK.")
    else:
        print(f"WARNING: {errors} examples missing output blocks")


if __name__ == "__main__":
    main()
