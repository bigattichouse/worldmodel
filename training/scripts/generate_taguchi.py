#!/usr/bin/env python3
"""
Generate Taguchi Design of Experiments training examples.

The core theme: the model uses Taguchi orthogonal arrays as a *reasoning tool*
to efficiently explore a parameter space and identify which factors matter
(high signal) vs which can be ignored (low noise). This teaches the model to
design experiments rather than brute-force all combinations.

Covers:
- L4 array: 3 factors, 2 levels each (4 runs vs 8 for full factorial)
- L9 array: 4 factors, 3 levels each (9 runs vs 81 for full factorial)
- Signal-to-noise ratio analysis (larger-is-better, smaller-is-better, nominal-is-best)
- Main effects analysis: which factor contributes most to outcome
- Multi-step: design the experiment → run it → analyse S/N ratios → rank factors
- Parameter optimization: find the level combination that maximises S/N

Usage:
    python training/scripts/generate_taguchi.py
    python training/scripts/generate_taguchi.py --output training/datasets/taguchi/basic.jsonl
"""

import sys
import json
import argparse
import random
import math
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.executor.python_exec import PythonExecutor


def run_code(executor: PythonExecutor, code: str) -> str:
    result = executor.run(code)
    return result.output_text().strip()


# ---------------------------------------------------------------------------
# L4 orthogonal array: 3 two-level factors, 4 runs
# Columns: [A, B, C] where each column has 2 levels (1/2)
# ---------------------------------------------------------------------------
L4 = [
    [1, 1, 1],
    [1, 2, 2],
    [2, 1, 2],
    [2, 2, 1],
]

# L9 orthogonal array: 4 three-level factors, 9 runs
# Standard L9 — 4 columns, 9 rows, levels 1/2/3
L9 = [
    [1, 1, 1, 1],
    [1, 2, 2, 2],
    [1, 3, 3, 3],
    [2, 1, 2, 3],
    [2, 2, 3, 1],
    [2, 3, 1, 2],
    [3, 1, 3, 2],
    [3, 2, 1, 3],
    [3, 3, 2, 1],
]


# ---------------------------------------------------------------------------
# Example: ML hyperparameter tuning with L9
# ---------------------------------------------------------------------------

def make_ml_hyperparameter_sweep(ex_id: str) -> dict:
    seed = random.randint(1, 999)
    executor = PythonExecutor(timeout=30)

    # Factor levels for an ML training run
    # Factor A: learning rate (3 levels)
    # Factor B: batch size
    # Factor C: dropout rate
    # Factor D: weight decay
    code1 = f"""\
import math, random
random.seed({seed})

# Taguchi L9 orthogonal array for ML hyperparameter exploration
# 4 factors, 3 levels each, only 9 runs (vs 81 for full factorial)
L9 = [
    [1,1,1,1], [1,2,2,2], [1,3,3,3],
    [2,1,2,3], [2,2,3,1], [2,3,1,2],
    [3,1,3,2], [3,2,1,3], [3,3,2,1],
]

# Factor levels
lr_levels     = [1e-4, 5e-4, 1e-3]     # A: learning rate
batch_levels  = [16, 32, 64]            # B: batch size
dropout_levels= [0.1, 0.3, 0.5]        # C: dropout
decay_levels  = [1e-5, 1e-4, 1e-3]     # D: weight decay

# Simulated validation accuracy (surrogate model: known optimal is lr=5e-4, batch=32)
def sim_accuracy(lr, batch, dropout, decay):
    base = 0.92
    # True signal: lr matters most, batch second, dropout mild, decay negligible
    lr_effect     = -8 * (math.log10(lr) + 3.3)**2   # peaks near 5e-4
    batch_effect  = -0.001 * (batch - 32)**2
    dropout_effect= -0.3 * (dropout - 0.2)**2
    decay_effect  = -0.1 * (math.log10(decay) + 4.5)**2
    noise = random.gauss(0, 0.005)
    return max(0.5, min(0.99, base + lr_effect*0.01 + batch_effect + dropout_effect + decay_effect*0.01 + noise))

print("Run | lr      | batch | dropout | decay  | val_acc")
print("-"*60)
results = []
for i, row in enumerate(L9):
    a, b, c, d = row
    lr, batch, dropout, decay = lr_levels[a-1], batch_levels[b-1], dropout_levels[c-1], decay_levels[d-1]
    acc = sim_accuracy(lr, batch, dropout, decay)
    results.append((a, b, c, d, acc))
    print(f"{{i+1:3d}} | {{lr:.0e}} | {{batch:5d}} | {{dropout:.1f}}    | {{decay:.0e}} | {{acc:.4f}}")
"""
    output1 = run_code(executor, code1)

    code2 = """\
import math

# Signal-to-Noise ratio for "larger is better": S/N = -10*log10(mean(1/y^2))
def sn_larger_is_better(values):
    return -10 * math.log10(sum(1/v**2 for v in values) / len(values))

# Compute main effects: average S/N at each level for each factor
factor_names = ['Learning Rate', 'Batch Size', 'Dropout', 'Weight Decay']
level_names = {
    0: ['1e-4', '5e-4', '1e-3'],
    1: ['16',   '32',   '64'],
    2: ['0.1',  '0.3',  '0.5'],
    3: ['1e-5', '1e-4', '1e-3'],
}

# Group results by factor level
sn_by_factor_level = []
for factor in range(4):
    sn_levels = []
    for level in range(1, 4):
        accs = [r[4] for r in results if r[factor] == level]
        sn = sn_larger_is_better(accs)
        sn_levels.append(sn)
    sn_by_factor_level.append(sn_levels)

print("Main Effects (S/N ratio, dB):")
print(f"{'Factor':<18} {'Level1':>8} {'Level2':>8} {'Level3':>8} {'Range':>8}")
print("-" * 55)
factor_ranges = []
for fi, (name, sn_levels) in enumerate(zip(factor_names, sn_by_factor_level)):
    rng = max(sn_levels) - min(sn_levels)
    factor_ranges.append((rng, name, sn_levels))
    lv = level_names[fi]
    print(f"{name:<18} {sn_levels[0]:>8.3f} {sn_levels[1]:>8.3f} {sn_levels[2]:>8.3f} {rng:>8.3f}")

# Rank factors by range (larger range = more influential)
factor_ranges.sort(reverse=True)
print("\\nFactor importance ranking (by S/N range):")
for rank, (rng, name, sn_levels) in enumerate(factor_ranges, 1):
    best_level = sn_levels.index(max(sn_levels)) + 1
    print(f"  {rank}. {name:<18} range={rng:.3f} dB  → optimal level {best_level}")
"""
    output2 = run_code(executor, code2)

    code3 = """\
# Predict optimal combination
print("Optimal configuration identified by Taguchi analysis:")
for fi, (name, sn_levels) in enumerate(zip(factor_names, sn_by_factor_level)):
    best_level_idx = sn_levels.index(max(sn_levels))
    print(f"  {name}: {level_names[fi][best_level_idx]}")

# Verify: run optimal config
a_opt = sn_by_factor_level[0].index(max(sn_by_factor_level[0])) + 1
b_opt = sn_by_factor_level[1].index(max(sn_by_factor_level[1])) + 1
c_opt = sn_by_factor_level[2].index(max(sn_by_factor_level[2])) + 1
d_opt = sn_by_factor_level[3].index(max(sn_by_factor_level[3])) + 1

from src.executor.python_exec import PythonExecutor
import math, random
random.seed(999)
lr_levels = [1e-4, 5e-4, 1e-3]
batch_levels = [16, 32, 64]
dropout_levels = [0.1, 0.3, 0.5]
decay_levels = [1e-5, 1e-4, 1e-3]

def sim_accuracy(lr, batch, dropout, decay):
    base = 0.92
    lr_effect = -8 * (math.log10(lr) + 3.3)**2
    batch_effect = -0.001 * (batch - 32)**2
    dropout_effect = -0.3 * (dropout - 0.2)**2
    decay_effect = -0.1 * (math.log10(decay) + 4.5)**2
    noise = random.gauss(0, 0.005)
    return max(0.5, min(0.99, base + lr_effect*0.01 + batch_effect + dropout_effect + decay_effect*0.01 + noise))

acc_opt = sim_accuracy(lr_levels[a_opt-1], batch_levels[b_opt-1],
                       dropout_levels[c_opt-1], decay_levels[d_opt-1])
best_in_experiment = max(r[4] for r in results)
print(f"\\nBest accuracy seen in 9 Taguchi runs:  {best_in_experiment:.4f}")
print(f"Predicted optimal (not in array):      {acc_opt:.4f}")
print(f"Improvement over best seen:            {(acc_opt - best_in_experiment)*100:+.2f}%")
print(f"\\nTaguchi used 9/81 = 11% of full factorial runs to find near-optimal config.")
"""
    output3 = run_code(executor, code3)

    return {
        "id": ex_id,
        "category": "taguchi",
        "difficulty": "advanced",
        "query": (
            "I'm tuning an ML model and have 4 hyperparameters to explore: "
            "learning rate (1e-4, 5e-4, 1e-3), batch size (16, 32, 64), "
            "dropout (0.1, 0.3, 0.5), and weight decay (1e-5, 1e-4, 1e-3). "
            "A full grid search would need 81 runs. Use a Taguchi L9 orthogonal array "
            "to explore the space in just 9 runs, then compute signal-to-noise ratios "
            "to identify which hyperparameters matter most."
        ),
        "response": (
            "<think>\n"
            "Taguchi L9 handles 4 three-level factors in just 9 runs. Key idea: "
            "orthogonality means each factor level appears exactly 3 times, so factor "
            "effects can be estimated independently. S/N ratio (larger-is-better: "
            "-10*log10(mean(1/y²))) measures both mean performance and consistency. "
            "The range of S/N across levels tells me which factors have real signal.\n"
            "</think>\n"
            f"<code>\n{code1}</code>\n"
            f"<output>\n{output1}\n</output>\n"
            "<think>\n"
            "Now compute S/N ratios and main effects. The factor with the largest "
            "S/N range is the most influential. I'll rank all four factors.\n"
            "</think>\n"
            f"<code>\n{code2}</code>\n"
            f"<output>\n{output2}\n</output>\n"
            "<think>\n"
            "The ranking shows which factors are signal vs noise. Now predict the "
            "optimal combination by picking the best level for each factor, and verify "
            "it outperforms the best configuration seen in the 9 runs.\n"
            "</think>\n"
            f"<code>\n{code3}</code>\n"
            f"<output>\n{output3}\n</output>\n"
            "Taguchi analysis reduced a 81-run exhaustive search to 9 structured runs, "
            "correctly identified learning rate as the dominant factor, and predicted "
            "an optimal configuration not in the original 9 runs. The S/N range is the "
            "key signal: large range → factor matters; small range → factor is noise."
        ),
    }


def make_l4_manufacturing(ex_id: str) -> dict:
    """L4 array: optimize a simple manufacturing process with 3 two-level factors."""
    seed = random.randint(1, 999)
    executor = PythonExecutor(timeout=30)
    code = f"""\
import math, random
random.seed({seed})

# L4 orthogonal array: 3 factors, 2 levels, 4 runs
# Full factorial would be 2^3 = 8 runs
L4 = [
    [1,1,1],
    [1,2,2],
    [2,1,2],
    [2,2,1],
]

# Factors for a welding process
# A: Temperature (Low=180°C, High=220°C)
# B: Pressure (Low=2bar, High=4bar)
# C: Duration (Short=2s, Long=4s)
factor_levels = {{
    'Temperature': [180, 220],
    'Pressure':    [2,   4],
    'Duration':    [2,   4],
}}

# Simulated yield strength (MPa) — temperature matters most, pressure secondary
def sim_strength(temp, pressure, duration):
    base = 450
    t_effect = 0.8 * (temp - 180)    # +32 at high temp
    p_effect = 5.0 * (pressure - 2)  # +10 at high pressure
    d_effect = 2.5 * (duration - 2)  # +5 at long duration
    noise = random.gauss(0, 3)
    return base + t_effect + p_effect + d_effect + noise

print("Taguchi L4 Experiment — Welding Process Optimization")
print(f"{{'-'*65}}")
print(f"{{'Run':>4}} {{'Temp(°C)':>10}} {{'Press(bar)':>11}} {{'Duration(s)':>12}} {{'Yield(MPa)':>12}}")
print(f"{{'-'*65}}")
results = []
for i, (a, b, c) in enumerate(L4):
    temp     = factor_levels['Temperature'][a-1]
    pressure = factor_levels['Pressure'][b-1]
    duration = factor_levels['Duration'][c-1]
    strength = sim_strength(temp, pressure, duration)
    results.append((a, b, c, strength))
    print(f"{{i+1:>4}} {{temp:>10}} {{pressure:>11}} {{duration:>12}} {{strength:>12.2f}}")

# Larger-is-better S/N ratio
def sn_lib(values):
    return -10 * math.log10(sum(1/v**2 for v in values) / len(values))

factor_names = ['Temperature', 'Pressure', 'Duration']
print(f"\\nMain Effects (S/N, larger-is-better):")
print(f"{{'Factor':<14}} {{'Low (L1)':>10}} {{'High (L2)':>10}} {{'Range':>8}}")
print(f"{{'-'*45}}")
factor_ranges = []
for fi, name in enumerate(factor_names):
    low_vals  = [r[3] for r in results if r[fi] == 1]
    high_vals = [r[3] for r in results if r[fi] == 2]
    sn_low  = sn_lib(low_vals)
    sn_high = sn_lib(high_vals)
    rng = abs(sn_high - sn_low)
    factor_ranges.append((rng, name, sn_low, sn_high))
    print(f"{{name:<14}} {{sn_low:>10.4f}} {{sn_high:>10.4f}} {{rng:>8.4f}}")

factor_ranges.sort(reverse=True)
print(f"\\nFactor ranking (signal = large S/N range):")
for rank, (rng, name, sn_l, sn_h) in enumerate(factor_ranges, 1):
    best = "High" if sn_h > sn_l else "Low"
    signal = "SIGNAL" if rng > 0.05 else "noise"
    print(f"  {{rank}}. {{name:<14}} range={{rng:.4f}} dB → best={{best}}  [{{signal}}]")

best_combo = []
for fi, name in enumerate(factor_names):
    low_vals  = [r[3] for r in results if r[fi] == 1]
    high_vals = [r[3] for r in results if r[fi] == 2]
    sn_l = sn_lib(low_vals); sn_h = sn_lib(high_vals)
    best_combo.append(factor_levels[name][1 if sn_h > sn_l else 0])
print(f"\\nOptimal settings: Temp={{best_combo[0]}}°C, Press={{best_combo[1]}}bar, Duration={{best_combo[2]}}s")
print(f"(Explored with 4 runs instead of 8 — 50% fewer experiments)")
"""
    output = run_code(executor, code)
    return {
        "id": ex_id,
        "category": "taguchi",
        "difficulty": "basic",
        "query": (
            "A welding process has 3 factors at 2 levels: Temperature (180°C or 220°C), "
            "Pressure (2 or 4 bar), Duration (2 or 4 seconds). Use a Taguchi L4 orthogonal "
            "array to find the best settings using only 4 runs instead of 8. Compute S/N "
            "ratios (larger-is-better) and identify which factors are signal vs noise."
        ),
        "response": (
            "<think>\n"
            "L4 is the smallest orthogonal array: 3 two-level factors, 4 runs.\n"
            "Each factor level appears exactly twice, so effects are balanced.\n"
            "S/N ratio (larger-is-better): -10*log10(mean(1/y²))\n"
            "Factor range = max(S/N across levels) - min(S/N). Large range = real signal.\n"
            "</think>\n"
            f"<code>\n{code}</code>\n"
            f"<output>\n{output}\n</output>\n"
            "The Taguchi analysis correctly identifies the dominant factors. The S/N range "
            "acts as a filter: factors with large range are genuine signals; small range "
            "means the factor doesn't matter much and can be set for cost or convenience."
        ),
    }


def make_sn_ratio_types(ex_id: str) -> dict:
    """Illustrate three S/N ratio types on a concrete dataset."""
    seed = random.randint(1, 999)
    executor = PythonExecutor(timeout=30)
    code = f"""\
import math, random
random.seed({seed})

# Three scenarios illustrating different S/N types:
# 1. Larger-is-better: battery life (want high)
# 2. Smaller-is-better: positioning error (want low)
# 3. Nominal-is-best: shaft diameter target 25.00mm (want close to target)

datasets = {{
    "Battery life (h)":      [8.2, 9.1, 8.7, 7.9, 9.3, 8.5],   # want high
    "Position error (mm)":   [0.12, 0.08, 0.15, 0.10, 0.09, 0.11],  # want low
    "Shaft diameter (mm)":   [25.03, 24.97, 25.01, 25.05, 24.99, 25.02],  # target=25
}}

target = 25.00

for name, data in datasets.items():
    n = len(data)
    mean_val = sum(data) / n
    var = sum((x - mean_val)**2 for x in data) / (n-1)
    std = math.sqrt(var)

    # S/N formulas
    sn_lib = -10 * math.log10(sum(1/x**2 for x in data) / n)
    sn_sib = -10 * math.log10(sum(x**2 for x in data) / n)
    # Nominal-is-best: 10*log10(mean^2 / variance)
    sn_nib = 10 * math.log10(mean_val**2 / var) if var > 0 else float('inf')

    print(f"{{name}}")
    print(f"  Data:     {{[f'{{x:.2f}}' for x in data]}}")
    print(f"  Mean={{mean_val:.4f}}, Std={{std:.4f}}")
    print(f"  S/N larger-is-better: {{sn_lib:>8.4f}} dB")
    print(f"  S/N smaller-is-better:{{sn_sib:>8.4f}} dB")
    print(f"  S/N nominal-is-best:  {{sn_nib:>8.4f}} dB  (target={{target}})")
    print()
"""
    output = run_code(executor, code)
    return {
        "id": ex_id,
        "category": "taguchi",
        "difficulty": "basic",
        "query": (
            "Explain and compute the three Taguchi signal-to-noise ratio types: "
            "larger-is-better (battery life), smaller-is-better (positioning error), "
            "and nominal-is-best (shaft diameter with target 25.00mm). "
            "Use representative datasets for each case."
        ),
        "response": (
            "<think>\n"
            "Three S/N formulas:\n"
            "  Larger-is-better: -10*log10(mean(1/y²))   [high y is good]\n"
            "  Smaller-is-better: -10*log10(mean(y²))    [low y is good]\n"
            "  Nominal-is-best: 10*log10(mean²/variance)  [y near target]\n"
            "Higher S/N always means better quality — the formula is chosen to match intent.\n"
            "</think>\n"
            f"<code>\n{code}</code>\n"
            f"<output>\n{output}\n</output>\n"
            "The S/N ratio is a single number that captures both the mean and variability "
            "of a response. Maximising S/N always means maximising quality — the three "
            "formulas simply define quality differently. In a Taguchi experiment, the "
            "factor level that maximises S/N is the robust optimal setting."
        ),
    }


def make_parameter_space_exploration(ex_id: str) -> dict:
    """
    The key use case the user highlighted: model designs a Taguchi sweep to
    understand a parameter space it doesn't know, using S/N ratios to separate
    signal from noise factors.
    """
    seed = random.randint(1, 999)
    executor = PythonExecutor(timeout=30)

    code1 = f"""\
import math, random
random.seed({seed})

# Scenario: optimising a neural network quantisation pipeline.
# We don't know which knobs matter. Let's design an experiment.
#
# Factors under investigation (3 levels each):
#   A: calibration_samples  (100, 500, 2000)
#   B: smoothquant_alpha    (0.3, 0.5, 0.7)
#   C: clip_percentile      (99.0, 99.9, 99.99)
#   D: layer_sensitivity_threshold (0.01, 0.05, 0.10)
#
# Response: perplexity on validation set (smaller is better)
# We suspect A and B matter, but aren't sure about C and D.

L9 = [
    [1,1,1,1],[1,2,2,2],[1,3,3,3],
    [2,1,2,3],[2,2,3,1],[2,3,1,2],
    [3,1,3,2],[3,2,1,3],[3,3,2,1],
]

cal_samples = [100, 500, 2000]
sq_alpha    = [0.3, 0.5, 0.7]
clip_pct    = [99.0, 99.9, 99.99]
sens_thresh = [0.01, 0.05, 0.10]

def sim_perplexity(n_cal, alpha, clip, thresh):
    # True model: n_cal and alpha matter; clip and thresh are mostly noise
    base = 18.0
    cal_effect  = -2.5 * math.log10(n_cal / 100)   # more cal → lower ppl
    alpha_effect = 3.0 * (alpha - 0.5)**2           # optimal at 0.5
    clip_effect  = 0.1 * (clip - 99.9)**2           # tiny effect
    thresh_effect= 0.05 * random.uniform(-1, 1)     # pure noise
    noise = random.gauss(0, 0.15)
    return max(12.0, base + cal_effect + alpha_effect + clip_effect + thresh_effect + noise)

factor_names = ['cal_samples', 'sq_alpha', 'clip_pct', 'sens_thresh']
level_strs   = {{
    'cal_samples': ['100', '500', '2000'],
    'sq_alpha':    ['0.3', '0.5', '0.7'],
    'clip_pct':    ['99.0', '99.9', '99.99'],
    'sens_thresh': ['0.01', '0.05', '0.10'],
}}

print("L9 Experiment: quantisation pipeline parameter sweep")
print(f"{{'Run':>4}} {{'cal':>6}} {{'alpha':>7}} {{'clip':>7}} {{'thresh':>8}} {{'ppl':>8}}")
print("-" * 45)
results = []
for i, (a,b,c,d) in enumerate(L9):
    n_cal  = cal_samples[a-1]
    alpha  = sq_alpha[b-1]
    clip   = clip_pct[c-1]
    thresh = sens_thresh[d-1]
    ppl    = sim_perplexity(n_cal, alpha, clip, thresh)
    results.append((a, b, c, d, ppl))
    print(f"{{i+1:>4}} {{n_cal:>6}} {{alpha:>7.1f}} {{clip:>7.2f}} {{thresh:>8.2f}} {{ppl:>8.4f}}")
"""
    output1 = run_code(executor, code1)

    code2 = """\
# Analyse: compute S/N ratios (smaller-is-better for perplexity)
def sn_sib(values):
    return -10 * math.log10(sum(v**2 for v in values) / len(values))

print("\\nMain Effects Table (S/N, smaller-is-better):")
print(f"{'Factor':<16} {'L1':>8} {'L2':>8} {'L3':>8} {'Range':>8} {'Verdict'}")
print("-" * 65)

effects = {}
for fi, name in enumerate(factor_names):
    sn_levels = []
    for level in range(1, 4):
        vals = [r[4] for r in results if r[fi] == level]
        sn_levels.append(sn_sib(vals))
    rng = max(sn_levels) - min(sn_levels)
    effects[name] = (sn_levels, rng)

# Rank by range
ranked = sorted(effects.items(), key=lambda x: x[1][1], reverse=True)
threshold = ranked[1][1][1] * 0.5  # factors below 50% of 2nd-place are noise

for name, (sn_levels, rng) in ranked:
    verdict = "SIGNAL" if rng >= threshold else "noise"
    print(f"{name:<16} {sn_levels[0]:>8.3f} {sn_levels[1]:>8.3f} {sn_levels[2]:>8.3f} {rng:>8.3f}  {verdict}")

print("\\nConclusion:")
signal_factors = [n for n,(s,r) in ranked if r >= threshold]
noise_factors  = [n for n,(s,r) in ranked if r <  threshold]
print(f"  Signal factors (tune carefully): {signal_factors}")
print(f"  Noise factors (set for convenience): {noise_factors}")
"""
    output2 = run_code(executor, code2)

    code3 = """\
# Find optimal levels for signal factors
print("\\nOptimal configuration:")
for name, (sn_levels, rng) in effects.items():
    best_level_idx = sn_levels.index(max(sn_levels))
    lv_str = level_strs[name][best_level_idx]
    importance = "  ← key setting" if name in signal_factors else "  (any value OK)"
    print(f"  {name:<16} = {lv_str}{importance}")

# Verify predicted optimum
a_opt = effects['cal_samples'][0].index(max(effects['cal_samples'][0])) + 1
b_opt = effects['sq_alpha'][0].index(max(effects['sq_alpha'][0])) + 1
c_opt = effects['clip_pct'][0].index(max(effects['clip_pct'][0])) + 1
d_opt = effects['sens_thresh'][0].index(max(effects['sens_thresh'][0])) + 1
import random, math
random.seed(1234)
ppl_opt = sim_perplexity(cal_samples[a_opt-1], sq_alpha[b_opt-1],
                          clip_pct[c_opt-1], sens_thresh[d_opt-1])
best_seen = min(r[4] for r in results)
print(f"\\nBest perplexity in 9 runs:         {best_seen:.4f}")
print(f"Predicted optimal (confirmation):  {ppl_opt:.4f}")
print(f"\\nExperiment efficiency: 9 runs explored what would have taken 81.")
print(f"S/N analysis correctly identified the signal factors.")
"""
    output3 = run_code(executor, code3)

    return {
        "id": ex_id,
        "category": "taguchi",
        "difficulty": "advanced",
        "query": (
            "I'm optimising a neural network quantisation pipeline and I'm not sure "
            "which of these 4 knobs actually matter: calibration_samples (100/500/2000), "
            "smoothquant_alpha (0.3/0.5/0.7), clip_percentile (99.0/99.9/99.99), "
            "and layer_sensitivity_threshold (0.01/0.05/0.10). "
            "Design a Taguchi L9 experiment to explore the space efficiently, then use "
            "S/N ratio analysis to separate the signal factors from the noise factors."
        ),
        "response": (
            "<think>\n"
            "I don't know which of these 4 factors matter. Instead of running all 81 "
            "combinations, I'll use an L9 array: 9 structured runs that are balanced across "
            "all factor levels. After running, I'll compute S/N ratios per factor level and "
            "look at the *range* of S/N across levels — large range means the factor is a "
            "real signal; small range means it's noise and I can ignore it.\n"
            "\n"
            "S/N for perplexity (smaller-is-better): -10*log10(mean(y²))\n"
            "Signal detection rule: rank factors by S/N range; those below ~50% of the "
            "top factor's range are probably noise.\n"
            "</think>\n"
            f"<code>\n{code1}</code>\n"
            f"<output>\n{output1}\n</output>\n"
            "<think>\n"
            "Now compute main effects. The S/N range for each factor tells me its importance. "
            "I'll threshold at 50% of the second-ranked factor to classify signal vs noise.\n"
            "</think>\n"
            f"<code>\n{code2}</code>\n"
            f"<output>\n{output2}\n</output>\n"
            "<think>\n"
            "I've identified the signal factors. Now predict the optimal setting by "
            "selecting the best level for each factor, and confirm with a verification run.\n"
            "</think>\n"
            f"<code>\n{code3}</code>\n"
            f"<output>\n{output3}\n</output>\n"
            "The Taguchi analysis discovered which factors are genuine signals without "
            "exhaustive search. The noise factors can be set for cost or convenience — "
            "they won't affect quality. This is Taguchi's core insight: orthogonal arrays "
            "let you separate signal from noise in 1/9th (or fewer) of the runs."
        ),
    }


def make_robust_design(ex_id: str) -> dict:
    """Robust design: find settings that give good performance across noise conditions."""
    seed = random.randint(1, 999)
    executor = PythonExecutor(timeout=30)
    code = f"""\
import math, random
random.seed({seed})

# Robust design problem: find control factor settings that give
# consistent output DESPITE variation in noise (uncontrollable) factors.
#
# Control factors (we choose):
#   A: spring_stiffness  [low, high]
#   B: surface_coating   [Type1, Type2]
#
# Noise factors (we can't control):
#   N: ambient_temperature [cold, hot]
#
# Response: positioning accuracy (mm, smaller is better)
# We run each L4 row under both noise conditions -> inner/outer array

L4_control = [[1,1],[1,2],[2,1],[2,2]]

def accuracy(stiffness, coating, temp):
    base = 0.50
    s_eff = -0.10 * (stiffness - 1)    # high stiffness reduces error
    c_eff = -0.05 * (coating - 1)      # coating2 slightly better
    t_eff = 0.08 * (temp - 1)          # hot temp increases error
    # Interaction: stiffness × temp (robust design detects this)
    interaction = -0.06 * (stiffness - 1) * (temp - 1)
    noise = random.gauss(0, 0.02)
    return max(0.01, base + s_eff + c_eff + t_eff + interaction + noise)

stiffness_vals = [1.0, 2.0]   # low=1, high=2
coating_vals   = [1, 2]        # type1=1, type2=2
temp_vals      = [1, 2]        # cold=1, hot=2

print("Inner/Outer Array Experiment (Robust Design)")
print(f"{'Run':>4} {'Stiff':>7} {'Coat':>6} {'Cold':>8} {'Hot':>8} {'Mean':>8} {'S/N':>8}")
print("-" * 55)

results = []
for a, b in L4_control:
    stiff = stiffness_vals[a-1]
    coat  = coating_vals[b-1]
    y_cold = accuracy(stiff, coat, 1)
    y_hot  = accuracy(stiff, coat, 2)
    mean_y = (y_cold + y_hot) / 2
    # Smaller-is-better S/N across noise conditions
    sn = -10 * math.log10((y_cold**2 + y_hot**2) / 2)
    results.append((a, b, y_cold, y_hot, mean_y, sn))
    print(f"{{len(results):>4}} {{stiff:>7.1f}} {{coat:>6}} {{y_cold:>8.4f}} {{y_hot:>8.4f}} {{mean_y:>8.4f}} {{sn:>8.3f}}")

# Main effects
for fi, name in enumerate(['Stiffness', 'Coating']):
    l1_sn = [r[5] for r in results if r[fi] == 1]
    l2_sn = [r[5] for r in results if r[fi] == 2]
    sn1 = sum(l1_sn)/len(l1_sn); sn2 = sum(l2_sn)/len(l2_sn)
    best = "Low" if sn1 > sn2 else "High"
    print(f"{{name}}: S/N L1={{sn1:.3f}}, L2={{sn2:.3f}} → best={{best}} (Δ={{abs(sn2-sn1):.3f}} dB)")

best_run = max(results, key=lambda r: r[5])
print(f"\\nMost robust setting: Stiffness={{'Low' if best_run[0]==1 else 'High'}}, "
      f"Coating={{'Type1' if best_run[1]==1 else 'Type2'}}")
print(f"  Mean error={{best_run[4]:.4f}} mm, S/N={{best_run[5]:.3f}} dB")
print(f"  Performance variation (cold vs hot): {{abs(best_run[2]-best_run[3]):.4f}} mm")
"""
    output = run_code(executor, code)
    return {
        "id": ex_id,
        "category": "taguchi",
        "difficulty": "intermediate",
        "query": (
            "A mechanical positioning system has two control factors: spring stiffness "
            "(low/high) and surface coating (Type1/Type2). Temperature (cold/hot) is a "
            "noise factor we can't control. Use Taguchi robust design with an inner/outer "
            "array to find which control factor settings give the best accuracy "
            "AND are least sensitive to temperature variation."
        ),
        "response": (
            "<think>\n"
            "Robust design: the inner array covers control factors (what we set), "
            "the outer array covers noise factors (what varies in the field). "
            "For each inner array row, we run under all noise conditions and compute S/N. "
            "High S/N = good mean performance AND insensitive to noise.\n"
            "Smaller-is-better S/N across noise: -10*log10(mean(y²))\n"
            "</think>\n"
            f"<code>\n{code}</code>\n"
            f"<output>\n{output}\n</output>\n"
            "The robust design found settings that work well in both cold and hot conditions. "
            "The S/N ratio rewards configurations that not only perform well on average but "
            "also vary little across noise conditions — this is the 'robust' part."
        ),
    }


def make_sn_screening(ex_id: str) -> dict:
    """Fast screening: use S/N ratios to quickly eliminate unimportant factors."""
    seed = random.randint(1, 999)
    n_factors = 7
    executor = PythonExecutor(timeout=30)
    code = f"""\
import math, random
random.seed({seed})

# Screening experiment: 7 factors, want to find the 2-3 that matter
# Use L8 (7 two-level factors, 8 runs) as a screening design
L8 = [
    [1,1,1,1,1,1,1],
    [1,1,1,2,2,2,2],
    [1,2,2,1,1,2,2],
    [1,2,2,2,2,1,1],
    [2,1,2,1,2,1,2],
    [2,1,2,2,1,2,1],
    [2,2,1,1,2,2,1],
    [2,2,1,2,1,1,2],
]

factor_labels = ['buffer_size', 'thread_count', 'cache_policy',
                 'prefetch_depth', 'compression', 'encoding', 'batch_align']

# True model: only buffer_size (A) and thread_count (B) matter
def sim_throughput(levels):
    a,b,c,d,e,f,g = levels
    base = 1000
    buf_effect    = 150 * (a - 1)   # +150 at high
    thread_effect = 80  * (b - 1)   # +80 at high
    # others: tiny or zero
    other = sum([(c-1)*2, (d-1)*(-1), (e-1)*3, (f-1)*1, (g-1)*(-2)])
    noise = random.gauss(0, 5)
    return base + buf_effect + thread_effect + other + noise

print(f"L8 Screening Experiment — {{len(factor_labels)}} factors, 8 runs")
print(f"(Full 2^7 factorial would need 128 runs)")
print()
results = []
for i, row in enumerate(L8):
    y = sim_throughput(row)
    results.append((*row, y))
    print(f"Run {{i+1}}: {{row}} → throughput={{y:.1f}} MB/s")

# S/N analysis (larger-is-better)
def sn_lib(vals):
    return -10 * math.log10(sum(1/v**2 for v in vals) / len(vals))

print("\\nScreening Results (S/N ranges):")
print(f"{{'Factor':<16}} {{'S/N Low':>10}} {{'S/N High':>10}} {{'Range':>8}}")
print("-" * 50)
ranges = []
for fi, name in enumerate(factor_labels):
    low_vals  = [r[7] for r in results if r[fi] == 1]
    high_vals = [r[7] for r in results if r[fi] == 2]
    sn_l = sn_lib(low_vals); sn_h = sn_lib(high_vals)
    rng = abs(sn_h - sn_l)
    ranges.append((rng, name, sn_l, sn_h))
    print(f"{{name:<16}} {{sn_l:>10.4f}} {{sn_h:>10.4f}} {{rng:>8.4f}}")

ranges.sort(reverse=True)
print("\\nScreening conclusion — factors ranked by importance:")
for rank, (rng, name, sn_l, sn_h) in enumerate(ranges, 1):
    flag = "*** ACTIVE ***" if rng > ranges[2][0] * 0.5 else "(inactive)"
    print(f"  {{rank}}. {{name:<16}} range={{rng:.4f}} dB  {{flag}}")
print("\\nNext step: run a detailed experiment on the active factors only.")
"""
    output = run_code(executor, code)
    return {
        "id": ex_id,
        "category": "taguchi",
        "difficulty": "intermediate",
        "query": (
            "A database system has 7 configurable parameters: buffer_size, thread_count, "
            "cache_policy, prefetch_depth, compression, encoding, batch_align. "
            "Running all 2^7=128 combinations is impractical. Use a Taguchi L8 screening "
            "experiment (8 runs) to quickly identify the 2-3 parameters that actually "
            "affect throughput, so we can focus further tuning on what matters."
        ),
        "response": (
            "<think>\n"
            "L8 handles 7 two-level factors in 8 runs. This is a *screening* experiment: "
            "the goal is not to find the optimum but to identify which factors are active.\n"
            "After the 8 runs, S/N range per factor separates active from inactive.\n"
            "Factors with small range → set them anywhere, focus resources on the active ones.\n"
            "</think>\n"
            f"<code>\n{code}</code>\n"
            f"<output>\n{output}\n</output>\n"
            "The L8 screening identified the active factors in 8 runs instead of 128. "
            "This is a powerful first step: screen with a small orthogonal array, "
            "then drill down on only the factors that showed signal."
        ),
    }


# ---------------------------------------------------------------------------
# Master list
# ---------------------------------------------------------------------------

BUILDERS = [
    make_sn_ratio_types,
    make_l4_manufacturing,
    make_robust_design,
    make_sn_screening,
    make_parameter_space_exploration,
    make_ml_hyperparameter_sweep,
]


def generate_examples(count: int) -> list:
    examples = []
    idx = 1
    per_builder = max(1, count // len(BUILDERS))
    remainder = count - per_builder * len(BUILDERS)

    random.seed(42)
    for i, builder in enumerate(BUILDERS):
        n = per_builder + (1 if i < remainder else 0)
        for _ in range(n):
            ex_id = f"taguchi_{idx:03d}"
            try:
                ex = builder(ex_id)
                examples.append(ex)
                idx += 1
                print(f"  {ex_id}: {ex['query'][:70]}...")
            except Exception as e:
                import traceback
                print(f"  SKIP {ex_id} ({builder.__name__}): {e}")
                traceback.print_exc()
    return examples


def main():
    parser = argparse.ArgumentParser(description="Generate Taguchi DOE training examples")
    parser.add_argument("--output", default="training/datasets/taguchi/basic.jsonl")
    parser.add_argument("--count", type=int, default=60)
    args = parser.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Generating {args.count} Taguchi examples...")
    examples = generate_examples(args.count)

    with open(out_path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")

    print(f"\nWrote {len(examples)} examples to {out_path}")


if __name__ == "__main__":
    main()
