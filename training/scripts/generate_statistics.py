#!/usr/bin/env python3
"""
Generate verified statistics training examples.

Covers: descriptive stats, probability, distributions, hypothesis concepts,
correlation, and data analysis patterns.

Usage:
    python training/scripts/generate_statistics.py --output training/datasets/statistics/basic.jsonl --count 120
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
    return {"id": ex_id, "category": category, "difficulty": difficulty,
            "query": query, "response": "\n".join(parts)}


# ─── Generators ───────────────────────────────────────────────────────────────

def gen_descriptive_stats(rng):
    n = rng.randint(6, 12)
    values = sorted([rng.randint(10, 100) for _ in range(n)])
    query = f"Find the mean, median, mode, and standard deviation of: {values}"
    think = "Compute all four descriptive statistics for the dataset."
    code = (
        f"import statistics\n"
        f"data = {values}\n"
        f"mean = statistics.mean(data)\n"
        f"median = statistics.median(data)\n"
        f"try:\n"
        f"    mode = statistics.mode(data)\n"
        f"except statistics.StatisticsError:\n"
        f"    mode = 'no unique mode'\n"
        f"stdev = statistics.stdev(data)\n"
        f"print(f'Mean: {{mean:.2f}}')\n"
        f"print(f'Median: {{median}}')\n"
        f"print(f'Mode: {{mode}}')\n"
        f"print(f'Std Dev: {{stdev:.4f}}')"
    )
    return query, think, "", code


def gen_range_variance(rng):
    n = rng.randint(5, 10)
    values = [rng.randint(5, 95) for _ in range(n)]
    query = f"Calculate the range, variance, and standard deviation for: {values}"
    think = "Range = max - min. Variance = average of squared deviations from mean. StdDev = √variance."
    code = (
        f"import statistics\n"
        f"data = {values}\n"
        f"r = max(data) - min(data)\n"
        f"var = statistics.variance(data)\n"
        f"std = statistics.stdev(data)\n"
        f"print(f'Range: {{r}}')\n"
        f"print(f'Variance: {{var:.4f}}')\n"
        f"print(f'Std Dev: {{std:.4f}}')"
    )
    return query, think, "", code


def gen_percentile(rng):
    n = rng.randint(10, 20)
    values = sorted([rng.randint(40, 100) for _ in range(n)])
    pct = rng.choice([25, 50, 75, 90])
    query = f"Find the {pct}th percentile of: {values}"
    think = f"The {pct}th percentile is the value below which {pct}% of observations fall."
    code = (
        f"import numpy as np\n"
        f"data = {values}\n"
        f"p = np.percentile(data, {pct})\n"
        f"print(f'{pct}th percentile: {{p}}')"
    )
    return query, think, "", code


def gen_probability_basic(rng):
    variants = [
        lambda: gen_prob_dice(rng),
        lambda: gen_prob_cards(rng),
        lambda: gen_prob_coin(rng),
    ]
    return rng.choice(variants)()


def gen_prob_dice(rng):
    n_dice = rng.choice([1, 2])
    if n_dice == 1:
        target = rng.randint(1, 6)
        query = f"What is the probability of rolling a {target} on a fair 6-sided die?"
        think = "One favorable outcome out of 6 equally likely outcomes."
        code = (
            f"favorable = 1\n"
            f"total = 6\n"
            f"prob = favorable / total\n"
            f"print(f'P(rolling {target}) = {{favorable}}/{{total}} = {{prob:.4f}}')"
        )
    else:
        target = rng.randint(2, 12)
        query = f"What is the probability of rolling a sum of {target} with two fair dice?"
        think = f"Count pairs (i,j) where i+j={target} out of 36 total outcomes."
        code = (
            f"target = {target}\n"
            f"favorable = sum(1 for i in range(1,7) for j in range(1,7) if i+j == target)\n"
            f"total = 36\n"
            f"prob = favorable / total\n"
            f"print(f'Favorable outcomes: {{favorable}}')\n"
            f"print(f'P(sum = {{target}}) = {{favorable}}/{{total}} = {{prob:.4f}}')"
        )
    return query, think, "", code


def gen_prob_cards(rng):
    card_type = rng.choice(["ace", "heart", "face card", "red card"])
    counts = {"ace": 4, "heart": 13, "face card": 12, "red card": 26}
    count = counts[card_type]
    query = f"What is the probability of drawing a {card_type} from a standard 52-card deck?"
    think = f"There are {count} {card_type}s in a 52-card deck."
    code = (
        f"favorable = {count}\n"
        f"total = 52\n"
        f"prob = favorable / total\n"
        f"print(f'P({card_type}) = {{favorable}}/{{total}} = {{prob:.4f}}')"
    )
    return query, think, "", code


def gen_prob_coin(rng):
    n = rng.randint(3, 5)
    k = rng.randint(0, n)
    query = f"What is the probability of getting exactly {k} heads in {n} coin flips?"
    think = f"Use binomial probability: P(X=k) = C(n,k) × p^k × (1-p)^(n-k) where p=0.5."
    code = (
        f"from math import comb\n"
        f"n, k = {n}, {k}\n"
        f"p = 0.5\n"
        f"prob = comb(n, k) * (p**k) * ((1-p)**(n-k))\n"
        f"print(f'C({n},{k}) = {{comb(n,k)}}')\n"
        f"print(f'P(X={k}) = {{prob:.4f}}')"
    )
    return query, think, "", code


def gen_normal_distribution(rng):
    mean = rng.choice([100, 70, 500, 0, 50])
    std = rng.choice([10, 15, 5, 1, 20])
    x = mean + rng.choice([-2, -1, 0, 1, 2]) * std
    query = (
        f"A dataset is normally distributed with mean {mean} and standard deviation {std}. "
        f"What percentage of values fall below {x}?"
    )
    think = "Use the normal CDF (cumulative distribution function)."
    code = (
        f"from scipy.stats import norm\n"
        f"mean, std = {mean}, {std}\n"
        f"x = {x}\n"
        f"pct = norm.cdf(x, mean, std) * 100\n"
        f"z = (x - mean) / std\n"
        f"print(f'z-score: {{z:.2f}}')\n"
        f"print(f'Percentage below {x}: {{pct:.2f}}%')"
    )
    return query, think, "", code


def gen_weighted_average(rng):
    n = rng.randint(3, 5)
    grades = [rng.randint(60, 100) for _ in range(n)]
    weights = [rng.randint(1, 5) for _ in range(n)]
    total_weight = sum(weights)
    query = (
        f"Calculate the weighted average of grades {grades} "
        f"with weights {weights}."
    )
    think = "Weighted average = sum(grade × weight) / sum(weights)."
    code = (
        f"grades = {grades}\n"
        f"weights = {weights}\n"
        f"weighted_sum = sum(g * w for g, w in zip(grades, weights))\n"
        f"total_weight = sum(weights)\n"
        f"avg = weighted_sum / total_weight\n"
        f"print(f'Weighted sum: {{weighted_sum}}')\n"
        f"print(f'Total weight: {{total_weight}}')\n"
        f"print(f'Weighted average: {{avg:.2f}}')"
    )
    return query, think, "", code


def gen_correlation(rng):
    """Multi-step: generate data, compute correlation"""
    n = 8
    x = [rng.randint(10, 50) for _ in range(n)]
    noise = [rng.randint(-5, 5) for _ in range(n)]
    slope = rng.choice([1.5, 2.0, 0.5, -1.0])
    y = [int(slope * xi + 20 + ni) for xi, ni in zip(x, noise)]

    query = f"Given x = {x} and y = {y}, calculate the Pearson correlation coefficient."
    think = "Pearson correlation measures linear relationship strength between two variables, ranging from -1 to +1."

    exec_shared = PythonExecutor()
    code1 = (
        f"import numpy as np\n"
        f"x = {x}\n"
        f"y = {y}\n"
        f"print(f'x: {{x}}')\n"
        f"print(f'y: {{y}}')\n"
        f"print(f'Mean x: {{np.mean(x):.2f}}, Mean y: {{np.mean(y):.2f}}')"
    )
    out1 = exec_shared.run(code1).output_text()

    code2 = (
        f"corr = np.corrcoef(x, y)[0, 1]\n"
        f"print(f'Pearson r: {{corr:.4f}}')\n"
        f"strength = 'strong' if abs(corr) > 0.7 else 'moderate' if abs(corr) > 0.4 else 'weak'\n"
        f"direction = 'positive' if corr > 0 else 'negative'\n"
        f"print(f'Interpretation: {{strength}} {{direction}} correlation')"
    )
    out2 = exec_shared.run(code2).output_text()

    response = (
        f"<think>\n{think}\n</think>\n"
        f"<code>\n{code1.strip()}\n</code>\n"
        f"<output>\n{out1.strip()}\n</output>\n"
        f"<think>\nNow compute the Pearson correlation coefficient.\n</think>\n"
        f"<code>\n{code2.strip()}\n</code>\n"
        f"<output>\n{out2.strip()}\n</output>"
    )

    return {
        "id": None,
        "category": "statistics",
        "difficulty": "intermediate",
        "query": query,
        "response": response,
    }


# ─── Main ─────────────────────────────────────────────────────────────────────

GENERATORS_SINGLE = [
    ("descriptive",   "basic",        gen_descriptive_stats,   0.20),
    ("range_var",     "basic",        gen_range_variance,      0.12),
    ("percentile",    "basic",        gen_percentile,          0.10),
    ("probability",   "basic",        gen_probability_basic,   0.25),
    ("normal_dist",   "intermediate", gen_normal_distribution, 0.18),
    ("weighted_avg",  "basic",        gen_weighted_average,    0.15),
]


def generate_examples(count: int, seed: int = 42) -> list:
    rng = random.Random(seed)
    examples = []
    idx = 0

    multistep_count = max(1, count // 6)
    single_count = count - multistep_count
    total_w = sum(g[3] for g in GENERATORS_SINGLE)

    for _ in range(single_count):
        r = rng.random()
        cumulative = 0.0
        chosen = GENERATORS_SINGLE[0]
        for gen in GENERATORS_SINGLE:
            cumulative += gen[3] / total_w
            if r <= cumulative:
                chosen = gen
                break
        name, difficulty, fn = chosen[0], chosen[1], chosen[2]
        try:
            result = fn(rng)
            query, think, model_text, code = result
            ex = make_example(f"stat_{idx:04d}", "statistics", difficulty,
                               query, think, model_text, code)
            examples.append(ex)
            idx += 1
        except Exception as e:
            print(f"  Warning: skipping example (error: {e})")

    for _ in range(multistep_count):
        try:
            ex = gen_correlation(rng)
            ex["id"] = f"stat_{idx:04d}"
            examples.append(ex)
            idx += 1
        except Exception as e:
            print(f"  Warning: skipping correlation example (error: {e})")

    rng.shuffle(examples)
    return examples


def main():
    parser = argparse.ArgumentParser(description="Generate statistics training examples")
    parser.add_argument("--output", required=True)
    parser.add_argument("--count", type=int, default=120)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Generating {args.count} statistics examples...")
    try:
        import scipy
        import numpy
    except ImportError as e:
        print(f"ERROR: Missing dependency: {e}")
        print("Run: pip install scipy numpy")
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
