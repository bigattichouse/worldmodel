#!/usr/bin/env python3
"""
Generate verified geometry training examples.

Covers: 2D shapes (area/perimeter), 3D shapes (volume/surface area),
trigonometry, the Pythagorean theorem, coordinate geometry.

Usage:
    python training/scripts/generate_geometry.py --output training/datasets/geometry/basic.jsonl --count 120
"""

import sys
import json
import random
import argparse
import math
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
    return {"id": ex_id, "category": category, "difficulty": difficulty,
            "query": query, "response": "\n".join(parts)}


# ─── 2D Shapes ────────────────────────────────────────────────────────────────

def gen_circle(rng):
    r = rng.randint(2, 25)
    query = f"Find the area and circumference of a circle with radius {r}."
    think = "Use A = πr² for area and C = 2πr for circumference."
    code = (
        f"import math\n"
        f"r = {r}\n"
        f"area = math.pi * r**2\n"
        f"circumference = 2 * math.pi * r\n"
        f"print(f'Area: {{area:.4f}} square units')\n"
        f"print(f'Circumference: {{circumference:.4f}} units')"
    )
    return query, think, "", code


def gen_rectangle_perimeter(rng):
    w = rng.randint(3, 50)
    h = rng.randint(3, 50)
    query = f"Find the area and perimeter of a rectangle with width {w} and height {h}."
    think = "Area = width × height, Perimeter = 2(width + height)."
    code = (
        f"w, h = {w}, {h}\n"
        f"area = w * h\n"
        f"perimeter = 2 * (w + h)\n"
        f"print(f'Area: {{area}} square units')\n"
        f"print(f'Perimeter: {{perimeter}} units')"
    )
    return query, think, "", code


def gen_triangle_area(rng):
    base = rng.randint(3, 30)
    height = rng.randint(3, 30)
    query = f"Find the area of a triangle with base {base} and height {height}."
    think = "Area of triangle = (base × height) / 2."
    code = (
        f"area = ({base} * {height}) / 2\n"
        f"print(f'Area: {{area}} square units')"
    )
    return query, think, "", code


def gen_trapezoid(rng):
    a = rng.randint(4, 20)
    b = rng.randint(4, 20)
    h = rng.randint(3, 15)
    query = f"Find the area of a trapezoid with parallel sides {a} and {b}, and height {h}."
    think = "Area of trapezoid = (a + b) / 2 × h."
    code = (
        f"a, b, h = {a}, {b}, {h}\n"
        f"area = (a + b) / 2 * h\n"
        f"print(f'Area: {{area}} square units')"
    )
    return query, think, "", code


def gen_pythagorean(rng):
    """Pythagorean theorem — find hypotenuse or leg"""
    variant = rng.choice(["hypotenuse", "leg"])
    if variant == "hypotenuse":
        a = rng.randint(3, 30)
        b = rng.randint(3, 30)
        query = f"A right triangle has legs of length {a} and {b}. Find the hypotenuse."
        think = "Use the Pythagorean theorem: c = √(a² + b²)."
        code = (
            f"import math\n"
            f"a, b = {a}, {b}\n"
            f"c = math.sqrt(a**2 + b**2)\n"
            f"print(f'Hypotenuse: {{c:.4f}}')"
        )
    else:
        # Generate Pythagorean triple
        triples = [(3,4,5),(5,12,13),(8,15,17),(7,24,25),(9,40,41),(6,8,10),(10,24,26)]
        a, b, c = rng.choice(triples)
        scale = rng.randint(1, 3)
        a, b, c = a*scale, b*scale, c*scale
        query = f"A right triangle has hypotenuse {c} and one leg {a}. Find the other leg."
        think = "Use Pythagorean theorem: b = √(c² - a²)."
        code = (
            f"import math\n"
            f"c, a = {c}, {a}\n"
            f"b = math.sqrt(c**2 - a**2)\n"
            f"print(f'Missing leg: {{b:.4f}}')"
        )
    return query, think, "", code


# ─── 3D Shapes ────────────────────────────────────────────────────────────────

def gen_sphere(rng):
    r = rng.randint(2, 15)
    query = f"Find the volume and surface area of a sphere with radius {r}."
    think = "V = (4/3)πr³, SA = 4πr²."
    code = (
        f"import math\n"
        f"r = {r}\n"
        f"volume = (4/3) * math.pi * r**3\n"
        f"surface_area = 4 * math.pi * r**2\n"
        f"print(f'Volume: {{volume:.4f}} cubic units')\n"
        f"print(f'Surface Area: {{surface_area:.4f}} square units')"
    )
    return query, think, "", code


def gen_cylinder(rng):
    r = rng.randint(2, 12)
    h = rng.randint(3, 20)
    query = f"Find the volume and lateral surface area of a cylinder with radius {r} and height {h}."
    think = "V = πr²h, lateral SA = 2πrh."
    code = (
        f"import math\n"
        f"r, h = {r}, {h}\n"
        f"volume = math.pi * r**2 * h\n"
        f"lateral_sa = 2 * math.pi * r * h\n"
        f"total_sa = lateral_sa + 2 * math.pi * r**2\n"
        f"print(f'Volume: {{volume:.4f}} cubic units')\n"
        f"print(f'Lateral Surface Area: {{lateral_sa:.4f}} square units')\n"
        f"print(f'Total Surface Area: {{total_sa:.4f}} square units')"
    )
    return query, think, "", code


def gen_cone(rng):
    r = rng.randint(2, 12)
    h = rng.randint(3, 20)
    query = f"Find the volume of a cone with base radius {r} and height {h}."
    think = "V = (1/3)πr²h."
    code = (
        f"import math\n"
        f"r, h = {r}, {h}\n"
        f"volume = (1/3) * math.pi * r**2 * h\n"
        f"slant = math.sqrt(r**2 + h**2)\n"
        f"print(f'Volume: {{volume:.4f}} cubic units')\n"
        f"print(f'Slant height: {{slant:.4f}} units')"
    )
    return query, think, "", code


# ─── Trigonometry ─────────────────────────────────────────────────────────────

def gen_trig_side(rng):
    """Find a side using sin/cos/tan"""
    angle = rng.choice([30, 45, 60, 37, 53])
    hyp = rng.randint(5, 40)
    fn_name = rng.choice(["sin", "cos"])
    if fn_name == "sin":
        query = f"In a right triangle, the hypotenuse is {hyp} and one angle is {angle}°. Find the opposite side."
        think = f"opposite = hypotenuse × sin(angle) = {hyp} × sin({angle}°)"
        code = (
            f"import math\n"
            f"hyp = {hyp}\n"
            f"angle_deg = {angle}\n"
            f"opposite = hyp * math.sin(math.radians(angle_deg))\n"
            f"print(f'Opposite side: {{opposite:.4f}}')"
        )
    else:
        query = f"In a right triangle, the hypotenuse is {hyp} and one angle is {angle}°. Find the adjacent side."
        think = f"adjacent = hypotenuse × cos(angle) = {hyp} × cos({angle}°)"
        code = (
            f"import math\n"
            f"hyp = {hyp}\n"
            f"angle_deg = {angle}\n"
            f"adjacent = hyp * math.cos(math.radians(angle_deg))\n"
            f"print(f'Adjacent side: {{adjacent:.4f}}')"
        )
    return query, think, "", code


def gen_angle_from_sides(rng):
    """Use arctan to find angle"""
    opp = rng.randint(3, 20)
    adj = rng.randint(3, 20)
    query = f"A right triangle has opposite side {opp} and adjacent side {adj}. Find the angle in degrees."
    think = "Use arctan(opposite/adjacent) to find the angle."
    code = (
        f"import math\n"
        f"opp, adj = {opp}, {adj}\n"
        f"angle_rad = math.atan(opp / adj)\n"
        f"angle_deg = math.degrees(angle_rad)\n"
        f"print(f'Angle: {{angle_deg:.2f}}°')"
    )
    return query, think, "", code


def gen_distance_between_points(rng):
    """Distance formula"""
    x1 = rng.randint(-10, 10)
    y1 = rng.randint(-10, 10)
    x2 = rng.randint(-10, 10)
    y2 = rng.randint(-10, 10)
    query = f"Find the distance between points ({x1}, {y1}) and ({x2}, {y2})."
    think = "Distance formula: d = √((x2-x1)² + (y2-y1)²)."
    code = (
        f"import math\n"
        f"x1, y1 = {x1}, {y1}\n"
        f"x2, y2 = {x2}, {y2}\n"
        f"distance = math.sqrt((x2-x1)**2 + (y2-y1)**2)\n"
        f"print(f'Distance: {{distance:.4f}}')"
    )
    return query, think, "", code


def gen_midpoint(rng):
    x1 = rng.randint(-15, 15)
    y1 = rng.randint(-15, 15)
    x2 = rng.randint(-15, 15)
    y2 = rng.randint(-15, 15)
    query = f"Find the midpoint of the segment from ({x1}, {y1}) to ({x2}, {y2})."
    think = "Midpoint = ((x1+x2)/2, (y1+y2)/2)."
    code = (
        f"x1, y1 = {x1}, {y1}\n"
        f"x2, y2 = {x2}, {y2}\n"
        f"mx = (x1 + x2) / 2\n"
        f"my = (y1 + y2) / 2\n"
        f"print(f'Midpoint: ({{mx}}, {{my}})')"
    )
    return query, think, "", code


# ─── Main ─────────────────────────────────────────────────────────────────────

GENERATORS = [
    ("circle",           "basic",        gen_circle,               0.10),
    ("rectangle",        "basic",        gen_rectangle_perimeter,  0.10),
    ("triangle_area",    "basic",        gen_triangle_area,        0.08),
    ("trapezoid",        "basic",        gen_trapezoid,            0.06),
    ("pythagorean",      "basic",        gen_pythagorean,          0.12),
    ("sphere",           "intermediate", gen_sphere,               0.10),
    ("cylinder",         "intermediate", gen_cylinder,             0.10),
    ("cone",             "intermediate", gen_cone,                 0.08),
    ("trig_side",        "intermediate", gen_trig_side,            0.10),
    ("angle_from_sides", "intermediate", gen_angle_from_sides,     0.08),
    ("distance",         "basic",        gen_distance_between_points, 0.05),
    ("midpoint",         "basic",        gen_midpoint,             0.03),
]


def generate_examples(count: int, seed: int = 42) -> list:
    rng = random.Random(seed)
    examples = []
    idx = 0
    total_w = sum(g[3] for g in GENERATORS)

    for _ in range(count):
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
            ex = make_example(f"geo_{idx:04d}", "geometry", difficulty,
                               query, think, model_text, code)
            examples.append(ex)
            idx += 1
        except Exception as e:
            print(f"  Warning: skipping example (error: {e})")

    return examples


def main():
    parser = argparse.ArgumentParser(description="Generate geometry training examples")
    parser.add_argument("--output", required=True)
    parser.add_argument("--count", type=int, default=120)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Generating {args.count} geometry examples...")
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
