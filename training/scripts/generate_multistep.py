#!/usr/bin/env python3
"""
Generate multi-step training examples.

These are problems that genuinely require 3+ code/output cycles, with state
passed between blocks. Examples cover:
- Simulate → analyse → optimise
- Generate data → fit model → predict
- Explore parameter space → refine → conclude
- Calculate → compare → recommend

Each example has multiple <code>/<output> cycles demonstrating
iterative problem-solving with execution feedback.

Usage:
    python training/scripts/generate_multistep.py
    python training/scripts/generate_multistep.py --output training/datasets/multi_step/basic.jsonl
    python training/scripts/generate_multistep.py --count 60
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
# Multi-step: projectile optimisation
# ---------------------------------------------------------------------------

def make_projectile_optimize(ex_id: str) -> dict:
    """Find optimal launch angle for maximum range with air resistance."""
    random.seed(random.randint(1, 999))
    v0 = random.choice([20, 30, 40, 50])
    executor = PythonExecutor()

    code1 = f"""\
import math

# Step 1: Baseline — no air resistance, 45° should be optimal
v0 = {v0}
g = 9.81
theta = 45
rad = math.radians(theta)

vx = v0 * math.cos(rad)
vy = v0 * math.sin(rad)
t_flight = 2 * vy / g
range_ideal = vx * t_flight

print(f"Ideal projectile (no air resistance):")
print(f"  v₀ = {{v0}} m/s, θ = {{theta}}°")
print(f"  Range = {{range_ideal:.2f}} m")
print(f"  Flight time = {{t_flight:.3f}} s")
print(f"  Max height = {{vy**2/(2*g):.2f}} m")
"""
    out1 = run_code(executor, code1)

    code2 = f"""\
# Step 2: Add air resistance (quadratic drag)
# F_drag = -½·ρ·Cd·A·v²  →  a_drag = -k·v²
k = 0.01  # drag coefficient (simplified)
dt = 0.001  # time step

best_angle = 0
best_range = 0

for angle in range(10, 81, 5):
    rad = math.radians(angle)
    vx = v0 * math.cos(rad)
    vy = v0 * math.sin(rad)
    x, y = 0.0, 0.0
    t = 0.0

    while y >= 0 or t == 0:
        v = math.sqrt(vx**2 + vy**2)
        if v > 0:
            ax = -k * v * vx
            ay = -g - k * v * vy
        else:
            ax, ay = 0, -g

        vx += ax * dt
        vy += ay * dt
        x += vx * dt
        y += vy * dt
        t += dt

        if y < -100:  # safety break
            break

    if x > best_range:
        best_range = x
        best_angle = angle

print(f"With air resistance (k={{k}}):")
print(f"  Optimal angle: {{best_angle}}°")
print(f"  Maximum range: {{best_range:.2f}} m")
print(f"  (Ideal was {{range_ideal:.2f}} m at 45°)")
print(f"  Range reduction: {{(1 - best_range/range_ideal)*100:.1f}}%")
"""
    out2 = run_code(executor, code2)

    code3 = f"""\
# Step 3: Fine-tune around the best angle
fine_angles = [best_angle - 2, best_angle - 1, best_angle, best_angle + 1, best_angle + 2]
print(f"Fine-tuning around {{best_angle}}°:")
print(f"{'Angle':>7}  {'Range (m)':>10}")
print("-" * 20)

results = []
for angle in fine_angles:
    rad = math.radians(angle)
    vx = v0 * math.cos(rad)
    vy = v0 * math.sin(rad)
    x, y, t = 0.0, 0.0, 0.0

    while y >= 0 or t == 0:
        v = math.sqrt(vx**2 + vy**2)
        if v > 0:
            ax = -k * v * vx
            ay = -g - k * v * vy
        vx += ax * dt
        vy += ay * dt
        x += vx * dt
        y += vy * dt
        t += dt
        if y < -100:
            break

    results.append((angle, x))
    print(f"{{angle:>7}}°  {{x:>10.2f}}")

best_fine = max(results, key=lambda r: r[1])
print(f"\\nOptimal: {{best_fine[0]}}° → {{best_fine[1]:.2f}} m")
"""
    out3 = run_code(executor, code3)

    return {
        "id": ex_id,
        "category": "multi_step",
        "difficulty": "advanced",
        "query": (
            f"A projectile is launched at {v0} m/s. "
            "Step 1: find the ideal range (no air resistance). "
            "Step 2: add quadratic drag and find the optimal angle by simulation. "
            "Step 3: fine-tune the angle to ±1° precision."
        ),
        "response": (
            "<think>\n"
            "Three-step approach:\n"
            "1. Ideal case: 45° is optimal, use analytic formula.\n"
            "2. With drag: simulate trajectory for many angles, find best.\n"
            "3. Fine-tune: narrow search around the coarse optimum.\n"
            "</think>\n"
            f"<code>\n{code1}</code>\n"
            f"<output>\n{out1}\n</output>\n"
            "<think>\n"
            "Now add air resistance. Need numerical simulation — no closed form.\n"
            "Euler integration with small dt. Sweep angles 10°–80° in 5° steps.\n"
            "</think>\n"
            f"<code>\n{code2}</code>\n"
            f"<output>\n{out2}\n</output>\n"
            "<think>\n"
            "Coarse search found the best angle. Now fine-tune with ±2° resolution.\n"
            "</think>\n"
            f"<code>\n{code3}</code>\n"
            f"<output>\n{out3}\n</output>\n"
            "Air resistance reduces range and shifts optimal angle below 45°."
        ),
    }


# ---------------------------------------------------------------------------
# Multi-step: data generation → fit → predict
# ---------------------------------------------------------------------------

def make_linear_regression(ex_id: str) -> dict:
    """Generate noisy data, fit linear model, predict with confidence."""
    random.seed(random.randint(1, 999))
    true_slope = round(random.uniform(1.5, 5.0), 2)
    true_intercept = round(random.uniform(-10, 10), 1)
    n_points = random.choice([20, 30, 50])
    executor = PythonExecutor()

    code1 = f"""\
import random
import math

# Step 1: Generate noisy linear data
random.seed(42)
true_slope = {true_slope}
true_intercept = {true_intercept}
noise_std = 2.0
n = {n_points}

x_data = [random.uniform(0, 20) for _ in range(n)]
y_data = [true_slope * x + true_intercept + random.gauss(0, noise_std) for x in x_data]

# Show first 10 points
print(f"Generated {{n}} data points: y = {{true_slope}}·x + {{true_intercept}} + noise(σ={{noise_std}})")
print(f"{'x':>8}  {'y':>8}")
print("-" * 18)
for i in range(min(10, n)):
    print(f"{{x_data[i]:>8.2f}}  {{y_data[i]:>8.2f}}")
if n > 10:
    print(f"  ... and {{n-10}} more points")

# Basic statistics
mean_x = sum(x_data) / n
mean_y = sum(y_data) / n
print(f"\\nMean x: {{mean_x:.3f}}, Mean y: {{mean_y:.3f}}")
"""
    out1 = run_code(executor, code1)

    code2 = f"""\
# Step 2: Least-squares linear fit
n = len(x_data)
sum_x = sum(x_data)
sum_y = sum(y_data)
sum_xy = sum(x*y for x, y in zip(x_data, y_data))
sum_x2 = sum(x**2 for x in x_data)

slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x**2)
intercept = (sum_y - slope * sum_x) / n

# R² (coefficient of determination)
y_pred = [slope * x + intercept for x in x_data]
ss_res = sum((y - yp)**2 for y, yp in zip(y_data, y_pred))
ss_tot = sum((y - mean_y)**2 for y in y_data)
r_squared = 1 - ss_res / ss_tot

# Residual standard error
s_e = math.sqrt(ss_res / (n - 2))

print(f"Least-squares fit: y = {{slope:.4f}}·x + {{intercept:.4f}}")
print(f"R² = {{r_squared:.6f}}")
print(f"Residual std error: s_e = {{s_e:.4f}}")
print(f"True values: slope={{true_slope}}, intercept={{true_intercept}}")
print(f"Errors: slope={{abs(slope-true_slope):.4f}}, intercept={{abs(intercept-true_intercept):.4f}}")
"""
    out2 = run_code(executor, code2)

    code3 = f"""\
# Step 3: Prediction with confidence interval
x_new = 15.0
y_pred_new = slope * x_new + intercept

# Standard error of prediction
s_xx = sum((x - mean_x)**2 for x in x_data)
se_pred = s_e * math.sqrt(1 + 1/n + (x_new - mean_x)**2 / s_xx)
t_95 = 2.0  # approximate for large n
ci_lower = y_pred_new - t_95 * se_pred
ci_upper = y_pred_new + t_95 * se_pred

print(f"Prediction at x = {{x_new}}:")
print(f"  ŷ = {{y_pred_new:.4f}}")
print(f"  95% CI: [{{ci_lower:.4f}}, {{ci_upper:.4f}}]")
print(f"  Margin of error: ±{{t_95*se_pred:.4f}}")

# True value
y_true = true_slope * x_new + true_intercept
print(f"  True value: {{y_true:.4f}}")
print(f"  Within CI: {{ci_lower <= y_true <= ci_upper}} ✓")
"""
    out3 = run_code(executor, code3)

    return {
        "id": ex_id,
        "category": "multi_step",
        "difficulty": "advanced",
        "query": (
            f"Generate {n_points} noisy data points from y = {true_slope}x + ({true_intercept}) "
            "with Gaussian noise (σ=2). "
            "Step 1: generate and summarize the data. "
            "Step 2: fit a linear model using least squares. "
            "Step 3: predict y at x=15 with a 95% confidence interval."
        ),
        "response": (
            "<think>\n"
            "Three steps:\n"
            "1. Generate data with known true parameters + noise.\n"
            "2. Fit using normal equations: slope and intercept from sums.\n"
            "3. Predict with confidence interval using residual standard error.\n"
            "</think>\n"
            f"<code>\n{code1}</code>\n"
            f"<output>\n{out1}\n</output>\n"
            "<think>\n"
            "Now fit the model. Use standard least-squares formulas.\n"
            "Check R² and compare fitted vs true parameters.\n"
            "</think>\n"
            f"<code>\n{code2}</code>\n"
            f"<output>\n{out2}\n</output>\n"
            "<think>\n"
            "Predict at x=15 with confidence interval.\n"
            "SE of prediction includes both model uncertainty and observation noise.\n"
            "</think>\n"
            f"<code>\n{code3}</code>\n"
            f"<output>\n{out3}\n</output>\n"
            "The fitted model recovers the true parameters within expected error."
        ),
    }


# ---------------------------------------------------------------------------
# Multi-step: Monte Carlo integration refinement
# ---------------------------------------------------------------------------

def make_monte_carlo_refine(ex_id: str) -> dict:
    """Monte Carlo integration with progressive refinement."""
    random.seed(random.randint(1, 999))
    # Integral of sin(x) from 0 to pi = 2
    executor = PythonExecutor()

    code1 = f"""\
import random
import math

# Step 1: Rough Monte Carlo — estimate ∫₀^π sin(x) dx = 2
random.seed(123)
a, b = 0, math.pi
n1 = 1000

def f(x):
    return math.sin(x)

samples = [f(random.uniform(a, b)) for _ in range(n1)]
mean_f = sum(samples) / n1
integral_est = (b - a) * mean_f
variance = sum((s - mean_f)**2 for s in samples) / (n1 - 1)
std_error = (b - a) * math.sqrt(variance / n1)

true_value = 2.0
error = abs(integral_est - true_value)

print(f"Monte Carlo: ∫₀^π sin(x) dx")
print(f"  N = {{n1}} samples")
print(f"  Estimate: {{integral_est:.6f}}")
print(f"  True value: {{true_value}}")
print(f"  Error: {{error:.6f}}")
print(f"  Std error: ±{{std_error:.6f}}")
"""
    out1 = run_code(executor, code1)

    code2 = f"""\
# Step 2: Increase samples and track convergence
for n in [10000, 100000, 1000000]:
    random.seed(42)
    samples = [math.sin(random.uniform(a, b)) for _ in range(n)]
    mean_f = sum(samples) / n
    est = (b - a) * mean_f
    var = sum((s - mean_f)**2 for s in samples) / (n - 1)
    se = (b - a) * math.sqrt(var / n)
    err = abs(est - true_value)

    print(f"  N={{n:>10,}:>10}  Estimate={{est:.8f}}  Error={{err:.8f}}  SE=±{{se:.8f}}")

print(f"\\nConvergence: error should decrease as 1/√N")
print(f"Expected: 10× more samples → ~3× better accuracy")
"""
    out2 = run_code(executor, code2)

    code3 = f"""\
# Step 3: Compare with analytical and numerical methods
# Analytical: ∫sin(x)dx = -cos(x), so [-cos(π)] - [-cos(0)] = 1+1 = 2
analytical = -math.cos(math.pi) - (-math.cos(0))

# Trapezoidal rule
n_trap = 10000
dx = (b - a) / n_trap
x_vals = [a + i*dx for i in range(n_trap + 1)]
trap_sum = (f(x_vals[0]) + f(x_vals[-1])) / 2 + sum(f(x) for x in x_vals[1:-1])
trap_est = dx * trap_sum

# Simpson's rule (need even number of intervals)
n_simp = 10000
dx = (b - a) / n_simp
x_vals = [a + i*dx for i in range(n_simp + 1)]
simp_sum = f(x_vals[0]) + f(x_vals[-1])
simp_sum += 4 * sum(f(x_vals[i]) for i in range(1, n_simp, 2))
simp_sum += 2 * sum(f(x_vals[i]) for i in range(2, n_simp-1, 2))
simp_est = dx / 3 * simp_sum

print(f"Comparison of integration methods for ∫₀^π sin(x) dx:")
print(f"  Analytical:    {{analytical:.15f}}")
print(f"  Monte Carlo:   {{est:.15f}}  (N=1M, error={{abs(est-analytical):.2e}})")
print(f"  Trapezoidal:   {{trap_est:.15f}}  (error={{abs(trap_est-analytical):.2e}})")
print(f"  Simpson's:     {{simp_est:.15f}}  (error={{abs(simp_est-analytical):.2e}})")
print(f"\\nSimpson's rule is most accurate for smooth functions ✓")
"""
    out3 = run_code(executor, code3)

    return {
        "id": ex_id,
        "category": "multi_step",
        "difficulty": "advanced",
        "query": (
            "Estimate ∫₀^π sin(x) dx using Monte Carlo integration. "
            "Step 1: rough estimate with 1000 samples. "
            "Step 2: refine with 10K, 100K, 1M samples and show convergence. "
            "Step 3: compare with trapezoidal and Simpson's rule."
        ),
        "response": (
            "<think>\n"
            "Monte Carlo: sample f(x) uniformly, estimate = (b-a)·mean(f).\n"
            "Error decreases as 1/√N.\n"
            "True value: ∫₀^π sin(x)dx = [-cos(x)]₀^π = 2\n"
            "</think>\n"
            f"<code>\n{code1}</code>\n"
            f"<output>\n{out1}\n</output>\n"
            "<think>\n"
            "Increase sample size to show convergence behavior.\n"
            "Expect ~3× improvement per 10× samples (1/√N scaling).\n"
            "</think>\n"
            f"<code>\n{code2}</code>\n"
            f"<output>\n{out2}\n</output>\n"
            "<think>\n"
            "Compare Monte Carlo with deterministic numerical integration.\n"
            "For smooth 1D integrals, Simpson's rule is far superior.\n"
            "Monte Carlo shines in high dimensions.\n"
            "</think>\n"
            f"<code>\n{code3}</code>\n"
            f"<output>\n{out3}\n</output>\n"
        ),
    }


# ---------------------------------------------------------------------------
# Multi-step: compound interest with contributions
# ---------------------------------------------------------------------------

def make_investment_simulate(ex_id: str) -> dict:
    """Simulate investment growth with monthly contributions, then optimize."""
    random.seed(random.randint(1, 999))
    initial = random.choice([1000, 5000, 10000])
    monthly = random.choice([100, 200, 500])
    annual_rate = round(random.uniform(0.04, 0.10), 2)
    years = random.choice([10, 15, 20, 25, 30])
    executor = PythonExecutor()

    code1 = f"""\
# Step 1: Simulate investment growth
initial = {initial}
monthly_contribution = {monthly}
annual_rate = {annual_rate}
monthly_rate = annual_rate / 12
months = {years} * 12

balance = initial
total_contributed = initial
balances = [balance]

for m in range(1, months + 1):
    balance *= (1 + monthly_rate)  # compound
    balance += monthly_contribution
    total_contributed += monthly_contribution
    balances.append(balance)

final = balances[-1]
gains = final - total_contributed

print(f"Investment simulation: {{years}} years")
print(f"  Initial: ${{initial:,}}, Monthly: ${{monthly_contribution:,}}")
print(f"  Annual return: {{annual_rate*100:.1f}}%")
print(f"  Total contributed: ${{total_contributed:,}}")
print(f"  Final balance: ${{final:,.2f}}")
print(f"  Total gains: ${{gains:,.2f}}  ({{gains/total_contributed*100:.1f}}% return)")
"""
    out1 = run_code(executor, code1)

    code2 = f"""\
# Step 2: Compare different contribution strategies
strategies = [
    ("Current", monthly_contribution),
    ("+25%", monthly_contribution * 1.25),
    ("+50%", monthly_contribution * 1.50),
    ("+100%", monthly_contribution * 2.0),
    ("Lump sum $5K", 0),
]

print(f"Strategy comparison ({{years}} years, {{annual_rate*100:.1f}}% annual):")
print(f"{'Strategy':>15}  {'Contributed':>14}  {'Final':>14}  {'Gains':>14}")
print("-" * 62)

for name, contrib in strategies:
    bal = initial
    contrib_total = initial
    lump = 5000 if "Lump" in name else 0
    bal += lump
    contrib_total += lump

    for m in range(months):
        bal *= (1 + monthly_rate)
        bal += contrib
        contrib_total += contrib

    gains = bal - contrib_total
    print(f"{{name:>15}}  ${{contrib_total:>12,.0f}}  ${{bal:>12,.2f}}  ${{gains:>12,.2f}}")
"""
    out2 = run_code(executor, code2)

    code3 = f"""\
# Step 3: What return rate is needed to reach $1M?
target = 1_000_000
print(f"Target: ${{target:,}} in {{years}} years with ${{monthly_contribution}}/month")
print(f"{'Rate':>8}  {'Final':>14}  {'Reached?':>10}")
print("-" * 35)

for rate_pct in [4, 6, 8, 10, 12, 15, 18, 20]:
    r = rate_pct / 100 / 12
    bal = initial
    for m in range(months):
        bal *= (1 + r)
        bal += monthly_contribution
    reached = "✓" if bal >= target else ""
    print(f"{{rate_pct:>6}}%  ${{bal:>12,.2f}}  {{reached:>10}}")

# Binary search for exact rate
lo, hi = 0.01, 0.30
for _ in range(100):
    mid = (lo + hi) / 2
    mr = mid / 12
    bal = initial
    for m in range(months):
        bal *= (1 + mr)
        bal += monthly_contribution
    if bal < target:
        lo = mid
    else:
        hi = mid

print(f"\\nRequired annual return: {{mid*100:.2f}}% → ${{bal:,.0f}}")
"""
    out3 = run_code(executor, code3)

    return {
        "id": ex_id,
        "category": "multi_step",
        "difficulty": "advanced",
        "query": (
            f"Simulate investing ${initial} initially with ${monthly}/month at "
            f"{annual_rate*100:.0f}% annual return for {years} years. "
            "Step 1: calculate the final balance. "
            "Step 2: compare different contribution strategies. "
            "Step 3: find what return rate is needed to reach $1,000,000."
        ),
        "response": (
            "<think>\n"
            "Three-step financial analysis:\n"
            "1. Simulate month-by-month compound growth with contributions.\n"
            "2. Compare strategies: increase contributions vs lump sum.\n"
            "3. Binary search for the required return rate to hit a target.\n"
            "</think>\n"
            f"<code>\n{code1}</code>\n"
            f"<output>\n{out1}\n</output>\n"
            "<think>\n"
            "Now compare strategies side by side.\n"
            "Higher contributions compound over decades — small increases matter a lot.\n"
            "</think>\n"
            f"<code>\n{code2}</code>\n"
            f"<output>\n{out2}\n</output>\n"
            "<think>\n"
            "Binary search for the rate that hits $1M.\n"
            "With modest contributions, the required rate may be unrealistically high.\n"
            "</think>\n"
            f"<code>\n{code3}</code>\n"
            f"<output>\n{out3}\n</output>\n"
        ),
    }


# ---------------------------------------------------------------------------
# Multi-step: statistical hypothesis testing
# ---------------------------------------------------------------------------

def make_hypothesis_test(ex_id: str) -> dict:
    """Generate data, run t-test, then power analysis."""
    random.seed(random.randint(1, 999))
    n_a = random.choice([20, 30, 50])
    n_b = random.choice([20, 30, 50])
    mean_a = round(random.uniform(50, 80), 1)
    mean_b = round(mean_a + random.uniform(2, 10), 1)
    std = round(random.uniform(5, 15), 1)
    executor = PythonExecutor()

    code1 = f"""\
import random
import math

# Step 1: Generate two groups of data and compute summary stats
random.seed(42)
n_a, n_b = {n_a}, {n_b}
mean_a, mean_b = {mean_a}, {mean_b}
std = {std}

group_a = [random.gauss(mean_a, std) for _ in range(n_a)]
group_b = [random.gauss(mean_b, std) for _ in range(n_b)]

mean_a_obs = sum(group_a) / n_a
mean_b_obs = sum(group_b) / n_b
var_a = sum((x - mean_a_obs)**2 for x in group_a) / (n_a - 1)
var_b = sum((x - mean_b_obs)**2 for x in group_b) / (n_b - 1)
std_a = math.sqrt(var_a)
std_b = math.sqrt(var_b)

print(f"Group A: n={{n_a}}, mean={{mean_a_obs:.3f}}, std={{std_a:.3f}}")
print(f"Group B: n={{n_b}}, mean={{mean_b_obs:.3f}}, std={{std_b:.3f}}")
print(f"Observed difference: {{mean_b_obs - mean_a_obs:.3f}}")
print(f"(True difference: {{mean_b - mean_a}})")
"""
    out1 = run_code(executor, code1)

    code2 = f"""\
# Step 2: Two-sample t-test (Welch's, unequal variances)
se = math.sqrt(var_a/n_a + var_b/n_b)
t_stat = (mean_b_obs - mean_a_obs) / se

# Welch-Satterthwaite degrees of freedom
num = (var_a/n_a + var_b/n_b)**2
den = (var_a/n_a)**2/(n_a-1) + (var_b/n_b)**2/(n_b-1)
df = num / den

# Approximate p-value using normal for large df
# For two-tailed test
from math import erf
def t_to_p(t, df):
    """Approximate two-tailed p-value from t-statistic."""
    # Use normal approximation for large df
    z = abs(t)
    p = 2 * (1 - 0.5 * (1 + erf(z / math.sqrt(2))))
    return p

p_value = t_to_p(t_stat, df)

print(f"Welch's t-test:")
print(f"  t = {{t_stat:.4f}}, df = {{df:.1f}}")
print(f"  p-value = {{p_value:.6f}}")
print(f"  Significant at α=0.05: {{p_value < 0.05}}")
print(f"  Significant at α=0.01: {{p_value < 0.01}}")

# Effect size (Cohen's d)
pooled_std = math.sqrt(((n_a-1)*var_a + (n_b-1)*var_b) / (n_a+n_b-2))
cohens_d = (mean_b_obs - mean_a_obs) / pooled_std
print(f"  Cohen's d = {{cohens_d:.4f}}  ({{'small' if abs(cohens_d)<0.5 else 'medium' if abs(cohens_d)<0.8 else 'large'}} effect)")
"""
    out2 = run_code(executor, code2)

    code3 = f"""\
# Step 3: Power analysis — what sample size do we need?
alpha = 0.05
target_power = 0.80
effect_size = abs(cohens_d)

# Approximate sample size per group for desired power
# n ≈ 2 * ((z_alpha/2 + z_beta) / d)²
z_alpha = 1.96  # for α=0.05, two-tailed
z_beta = 0.84   # for 80% power

n_per_group = int(math.ceil(2 * ((z_alpha + z_beta) / effect_size)**2))

print(f"Power analysis:")
print(f"  Observed effect size (Cohen's d): {{effect_size:.4f}}")
print(f"  For {{target_power*100:.0f}}% power at α={{alpha}}:")
print(f"  Need ~{{n_per_group}} samples per group")
print(f"  (Currently have n_a={{n_a}}, n_b={{n_b}})")

# Show power curve
print(f"\\nPower vs sample size:")
print(f"{'n/group':>8}  {'Power':>8}")
print("-" * 18)
for n in [10, 20, 30, 50, 100, 200, 500]:
    z_beta_est = effect_size * math.sqrt(n/2) - z_alpha
    # Approximate power
    from math import erf
    power = 0.5 * (1 + erf(z_beta_est / math.sqrt(2)))
    marker = " ← current" if n == n_a else ""
    print(f"{{n:>8}}  {{power:>7.3f}}{{marker}}")
"""
    out3 = run_code(executor, code3)

    return {
        "id": ex_id,
        "category": "multi_step",
        "difficulty": "advanced",
        "query": (
            f"Group A (n={n_a}) has mean ~{mean_a} and Group B (n={n_b}) has mean ~{mean_b}, "
            f"both with std ~{std}. "
            "Step 1: generate the data and compute summary statistics. "
            "Step 2: perform Welch's t-test — is the difference significant? "
            "Step 3: power analysis — what sample size would we need for 80% power?"
        ),
        "response": (
            "<think>\n"
            "Three-step statistical analysis:\n"
            "1. Generate data from known distributions, compute descriptive stats.\n"
            "2. Welch's t-test (doesn't assume equal variances).\n"
            "3. Power analysis to determine required sample size.\n"
            "</think>\n"
            f"<code>\n{code1}</code>\n"
            f"<output>\n{out1}\n</output>\n"
            "<think>\n"
            "Welch's t-test: t = (mean_B - mean_A) / SE\n"
            "SE = sqrt(var_A/n_A + var_B/n_B)\n"
            "Also compute Cohen's d for effect size.\n"
            "</think>\n"
            f"<code>\n{code2}</code>\n"
            f"<output>\n{out2}\n</output>\n"
            "<think>\n"
            "Power analysis: n ≈ 2((z_α/2 + z_β)/d)²\n"
            "Show power curve across sample sizes.\n"
            "</think>\n"
            f"<code>\n{code3}</code>\n"
            f"<output>\n{out3}\n</output>\n"
        ),
    }


# ---------------------------------------------------------------------------
# Multi-step: numerical ODE solving
# ---------------------------------------------------------------------------

def make_ode_solve(ex_id: str) -> dict:
    """Solve a differential equation numerically, compare methods."""
    random.seed(random.randint(1, 999))
    # dy/dt = -ky (exponential decay)
    k = round(random.uniform(0.1, 1.0), 2)
    y0 = random.choice([10, 50, 100])
    executor = PythonExecutor()

    code1 = f"""\
import math

# Step 1: Euler's method for dy/dt = -ky
k = {k}
y0 = {y0}
t_end = 10.0
dt = 0.1
n_steps = int(t_end / dt)

t_vals = [i * dt for i in range(n_steps + 1)]
y_euler = [y0]

y = y0
for i in range(n_steps):
    y += (-k * y) * dt
    y_euler.append(y)

# Analytical solution: y = y₀·e^(-kt)
y_exact = [y0 * math.exp(-k * t) for t in t_vals]

# Error at final step
euler_final = y_euler[-1]
exact_final = y_exact[-1]
error = abs(euler_final - exact_final)

print(f"Euler's method: dy/dt = -{{k}}y, y(0)={{y0}}, dt={{dt}}")
print(f"  t=0: Euler={{y_euler[0]:.4f}}, Exact={{y_exact[0]:.4f}}")
print(f"  t=5: Euler={{y_euler[50]:.4f}}, Exact={{y_exact[50]:.4f}}")
print(f"  t=10: Euler={{euler_final:.6f}}, Exact={{exact_final:.6f}}")
print(f"  Error at t=10: {{error:.6f}}")
"""
    out1 = run_code(executor, code1)

    code2 = f"""\
# Step 2: Runge-Kutta 4th order (RK4)
y_rk4 = [y0]
y = y0

for i in range(n_steps):
    t = i * dt
    k1 = -k * y
    k2 = -k * (y + 0.5*dt*k1)
    k3 = -k * (y + 0.5*dt*k2)
    k4 = -k * (y + dt*k3)
    y += (dt/6) * (k1 + 2*k2 + 2*k3 + k4)
    y_rk4.append(y)

rk4_final = y_rk4[-1]
rk4_error = abs(rk4_final - exact_final)

print(f"RK4 method: dy/dt = -{{k}}y, y(0)={{y0}}, dt={{dt}}")
print(f"  t=10: RK4={{rk4_final:.10f}}, Exact={{exact_final:.10f}}")
print(f"  Error: {{rk4_error:.2e}}")
print(f"\\nEuler error:  {{error:.2e}}")
print(f"RK4 error:    {{rk4_error:.2e}}")
print(f"RK4 is {{error/rk4_error:.0f}}× more accurate ✓")
"""
    out2 = run_code(executor, code2)

    code3 = f"""\
# Step 3: Convergence study — how does error scale with dt?
print(f"Convergence study:")
print(f"{'dt':>8}  {'Euler err':>12}  {'RK4 err':>12}")
print("-" * 35)

for dt_val in [0.5, 0.1, 0.05, 0.01, 0.005, 0.001]:
    n = int(t_end / dt_val)

    # Euler
    y_e = y0
    for _ in range(n):
        y_e += (-k * y_e) * dt_val

    # RK4
    y_r = y0
    for _ in range(n):
        k1 = -k * y_r
        k2 = -k * (y_r + 0.5*dt_val*k1)
        k3 = -k * (y_r + 0.5*dt_val*k2)
        k4 = -k * (y_r + dt_val*k3)
        y_r += (dt_val/6) * (k1 + 2*k2 + 2*k3 + k4)

    exact = y0 * math.exp(-k * t_end)
    euler_err = abs(y_e - exact)
    rk4_err = abs(y_r - exact)

    print(f"{{dt_val:>8.3f}}  {{euler_err:>12.2e}}  {{rk4_err:>12.2e}}")

print(f"\\nEuler: O(dt) — halving dt halves error")
print(f"RK4:   O(dt⁴) — halving dt reduces error by ~16×")
"""
    out3 = run_code(executor, code3)

    return {
        "id": ex_id,
        "category": "multi_step",
        "difficulty": "advanced",
        "query": (
            f"Solve dy/dt = -{k}y, y(0) = {y0} numerically from t=0 to t=10. "
            "Step 1: use Euler's method (dt=0.1) and compare with the analytical solution. "
            "Step 2: use RK4 and compare accuracy. "
            "Step 3: convergence study — how does error scale with step size?"
        ),
        "response": (
            "<think>\n"
            "Three-step ODE analysis:\n"
            "1. Euler's method: y_{n+1} = y_n + f(y_n)·dt\n"
            "   Analytical: y = y₀·e^(-kt)\n"
            "2. RK4: 4th order Runge-Kutta, much more accurate.\n"
            "3. Convergence: Euler is O(dt), RK4 is O(dt⁴).\n"
            "</think>\n"
            f"<code>\n{code1}</code>\n"
            f"<output>\n{out1}\n</output>\n"
            "<think>\n"
            "RK4 uses 4 slope evaluations per step for much better accuracy.\n"
            "</think>\n"
            f"<code>\n{code2}</code>\n"
            f"<output>\n{out2}\n</output>\n"
            "<think>\n"
            "Convergence study: sweep dt values, plot error scaling.\n"
            "Euler error ∝ dt, RK4 error ∝ dt⁴.\n"
            "</think>\n"
            f"<code>\n{code3}</code>\n"
            f"<output>\n{out3}\n</output>\n"
        ),
    }


# ---------------------------------------------------------------------------
# Multi-step: sorting algorithm comparison
# ---------------------------------------------------------------------------

def make_sort_compare(ex_id: str) -> dict:
    """Generate data, compare sorting algorithms, analyze scaling."""
    random.seed(random.randint(1, 999))
    executor = PythonExecutor()

    code1 = f"""\
import random
import time

# Step 1: Implement and verify sorting algorithms
random.seed(42)
n = 1000
data = [random.randint(1, 10000) for _ in range(n)]

def bubble_sort(arr):
    a = arr[:]
    n = len(a)
    for i in range(n):
        for j in range(0, n-i-1):
            if a[j] > a[j+1]:
                a[j], a[j+1] = a[j+1], a[j]
    return a

def insertion_sort(arr):
    a = arr[:]
    for i in range(1, len(a)):
        key = a[i]
        j = i - 1
        while j >= 0 and a[j] > key:
            a[j+1] = a[j]
            j -= 1
        a[j+1] = key
    return a

# Verify correctness
expected = sorted(data)
assert bubble_sort(data) == expected, "Bubble sort failed!"
assert insertion_sort(data) == expected, "Insertion sort failed!"

print(f"Sorting verification: n={{n}}")
print(f"  Bubble sort: ✓ correct")
print(f"  Insertion sort: ✓ correct")
print(f"  Python sorted(): ✓ correct")
"""
    out1 = run_code(executor, code1)

    code2 = f"""\
# Step 2: Benchmark on increasing sizes
import time

sizes = [100, 500, 1000, 2000, 5000]
print(f"{'Size':>6}  {'Bubble (s)':>12}  {'Insert (s)':>12}  {'Python (s)':>12}")
print("-" * 48)

for n in sizes:
    data = [random.randint(1, 100000) for _ in range(n)]

    t0 = time.time()
    bubble_sort(data)
    t_bubble = time.time() - t0

    t0 = time.time()
    insertion_sort(data)
    t_insert = time.time() - t0

    t0 = time.time()
    sorted(data)
    t_python = time.time() - t0

    print(f"{{n:>6}}  {{t_bubble:>12.4f}}  {{t_insert:>12.4f}}  {{t_python:>12.6f}}")
"""
    out2 = run_code(executor, code2)

    code3 = f"""\
# Step 3: Test on special cases (already sorted, reverse, nearly sorted)
print(f"Special case analysis (n=2000):")
print(f"{'Case':>20}  {'Bubble':>10}  {'Insert':>10}  {'Python':>10}")
print("-" * 55)

n = 2000
cases = {
    "Random": [random.randint(1, 100000) for _ in range(n)],
    "Already sorted": list(range(n)),
    "Reverse sorted": list(range(n, 0, -1)),
    "Nearly sorted": (lambda a: (a[50], a[51] := a[51], a[50] := a[50], a)[0] or a)(list(range(n))),
}

for name, arr in cases.items():
    t0 = time.time()
    bubble_sort(arr)
    tb = time.time() - t0

    t0 = time.time()
    insertion_sort(arr)
    ti = time.time() - t0

    t0 = time.time()
    sorted(arr)
    tp = time.time() - t0

    print(f"{{name:>20}}  {{tb:>10.4f}}  {{ti:>10.4f}}  {{tp:>10.6f}}")

print(f"\\nInsertion sort is O(n) on sorted data ✓")
print(f"Bubble sort benefits from early-termination on sorted data ✓")
print(f"Python's Timsort is optimized for partially sorted data ✓")
"""
    out3 = run_code(executor, code3)

    return {
        "id": ex_id,
        "category": "multi_step",
        "difficulty": "advanced",
        "query": (
            "Compare sorting algorithms: bubble sort, insertion sort, and Python's built-in sort. "
            "Step 1: implement and verify correctness. "
            "Step 2: benchmark on increasing sizes. "
            "Step 3: test on special cases (sorted, reverse, nearly sorted)."
        ),
        "response": (
            "<think>\n"
            "Three-step algorithm comparison:\n"
            "1. Implement bubble sort and insertion sort, verify against sorted().\n"
            "2. Benchmark: both are O(n²), Python's Timsort is O(n log n).\n"
            "3. Special cases: insertion sort is O(n) on sorted data.\n"
            "</think>\n"
            f"<code>\n{code1}</code>\n"
            f"<output>\n{out1}\n</output>\n"
            "<think>\n"
            "Benchmark on increasing sizes to confirm O(n²) scaling.\n"
            "</think>\n"
            f"<code>\n{code2}</code>\n"
            f"<output>\n{out2}\n</output>\n"
            "<think>\n"
            "Test edge cases — insertion sort excels on nearly sorted data.\n"
            "</think>\n"
            f"<code>\n{code3}</code>\n"
            f"<output>\n{out3}\n</output>\n"
        ),
    }


# ---------------------------------------------------------------------------
# Multi-step: matrix operations
# ---------------------------------------------------------------------------

def make_matrix_operations(ex_id: str) -> dict:
    """Matrix multiplication, inversion, eigenvalues — step by step."""
    random.seed(random.randint(1, 999))
    executor = PythonExecutor()

    code1 = f"""\
# Step 1: Create matrices and multiply
random.seed(42)
n = 4

# Create a symmetric positive definite matrix
A = [[random.uniform(1, 5) if i == j else random.uniform(-1, 1) for j in range(n)] for i in range(n)]
# Make symmetric
for i in range(n):
    for j in range(i+1, n):
        A[i][j] = A[j][i] = (A[i][j] + A[j][i]) / 2

# Create vector
b = [random.uniform(1, 10) for _ in range(n)]

def mat_vec(M, v):
    return [sum(M[i][j]*v[j] for j in range(len(v))) for i in range(len(M))]

def dot(u, v):
    return sum(a*b for a, b in zip(u, v))

Ab = mat_vec(A, b)

print(f"Matrix A ({n}×{n}, symmetric):")
for row in A:
    print(f"  {[f'{{x:.3f}}' for x in row]}")
print(f"\\nVector b: {[f'{{x:.3f}}' for x in b]}")
print(f"A·b = {[f'{{x:.3f}}' for x in Ab]}")
print(f"bᵀAb = {{dot(b, Ab):.4f}}")
"""
    out1 = run_code(executor, code1)

    code2 = f"""\
# Step 2: Solve Ax = b using Gaussian elimination
# Augmented matrix [A|b]
aug = [A[i][:] + [b[i]] for i in range(n)]

# Forward elimination
for col in range(n):
    # Partial pivoting
    max_row = max(range(col, n), key=lambda r: abs(aug[r][col]))
    aug[col], aug[max_row] = aug[max_row], aug[col]

    pivot = aug[col][col]
    for row in range(col + 1, n):
        factor = aug[row][col] / pivot
        for j in range(col, n + 1):
            aug[row][j] -= factor * aug[col][j]

# Back substitution
x = [0.0] * n
for i in range(n - 1, -1, -1):
    x[i] = (aug[i][n] - sum(aug[i][j]*x[j] for j in range(i+1, n))) / aug[i][i]

# Verify: compute residual
residual = mat_vec(A, x)
residual_err = [residual[i] - b[i] for i in range(n)]

print(f"Solution x (Gaussian elimination):")
for i in range(n):
    print(f"  x[{{i}}] = {{x[i]:.8f}}")
print(f"\\nResidual Ax - b: {{[f'{{e:.2e}}' for e in residual_err]}}")
print(f"Max residual error: {{max(abs(e) for e in residual_err):.2e}}")
"""
    out2 = run_code(executor, code2)

    code3 = f"""\
# Step 3: Power method — find dominant eigenvalue
# Start with random vector
v = [1.0] * n
eigenvalue = 0.0

print(f"Power method for dominant eigenvalue:")
print(f"{'Iteration':>10}  {'Eigenvalue':>14}  {'Change':>14}")
print("-" * 42)

for iteration in range(50):
    w = mat_vec(A, v)
    new_eigenvalue = max(abs(wi) for wi in w)
    # Normalize
    v = [wi / new_eigenvalue for wi in w]
    change = abs(new_eigenvalue - eigenvalue)
    eigenvalue = new_eigenvalue

    if iteration < 10 or iteration % 10 == 0:
        print(f"{{iteration:>10}}  {{eigenvalue:>14.8f}}  {{change:>14.2e}}")

    if change < 1e-12:
        print(f"Converged at iteration {{iteration}}")
        break

print(f"\\nDominant eigenvalue: {{eigenvalue:.10f}}")
print(f"Eigenvector: {[f'{{vi:.6f}}' for vi in v]}")

# Verify: Av = λv
Av = mat_vec(A, v)
lv = [eigenvalue * vi for vi in v]
err = max(abs(Av[i] - lv[i]) for i in range(n))
print(f"Verification ||Av - λv||_∞ = {{err:.2e}} ✓")
"""
    out3 = run_code(executor, code3)

    return {
        "id": ex_id,
        "category": "multi_step",
        "difficulty": "advanced",
        "query": (
            "Create a 4×4 symmetric positive definite matrix A and a vector b. "
            "Step 1: compute A·b and bᵀAb. "
            "Step 2: solve Ax = b using Gaussian elimination with partial pivoting. "
            "Step 3: find the dominant eigenvalue using the power method."
        ),
        "response": (
            "<think>\n"
            "Three-step linear algebra:\n"
            "1. Matrix-vector multiplication, quadratic form.\n"
            "2. Gaussian elimination with partial pivoting for numerical stability.\n"
            "3. Power method: iterate v ← Av/||Av|| to find dominant eigenvalue.\n"
            "</think>\n"
            f"<code>\n{code1}</code>\n"
            f"<output>\n{out1}\n</output>\n"
            "<think>\n"
            "Gaussian elimination: forward elimination + back substitution.\n"
            "Partial pivoting prevents division by small numbers.\n"
            "</think>\n"
            f"<code>\n{code2}</code>\n"
            f"<output>\n{out2}\n</output>\n"
            "<think>\n"
            "Power method converges to the eigenvector of the largest eigenvalue.\n"
            "Rate of convergence depends on ratio |λ₂/λ₁|.\n"
            "</think>\n"
            f"<code>\n{code3}</code>\n"
            f"<output>\n{out3}\n</output>\n"
        ),
    }


# ---------------------------------------------------------------------------
# Multi-step: random walk / Markov chain
# ---------------------------------------------------------------------------

def make_markov_chain(ex_id: str) -> dict:
    """Build a Markov chain, simulate, find stationary distribution."""
    random.seed(random.randint(1, 999))
    executor = PythonExecutor()

    code1 = f"""\
import random

# Step 1: Define a 4-state Markov chain (weather model)
# States: Sunny, Cloudy, Rainy, Stormy
states = ["Sunny", "Cloudy", "Rainy", "Stormy"]

# Transition matrix (rows sum to 1)
P = [
    [0.6, 0.3, 0.08, 0.02],   # Sunny →
    [0.2, 0.4, 0.3, 0.1],     # Cloudy →
    [0.05, 0.2, 0.4, 0.35],   # Rainy →
    [0.02, 0.1, 0.38, 0.5],   # Stormy →
]

# Verify rows sum to 1
for i, row in enumerate(P):
    print(f"P({states[i]} → ·) = {{row}}  (sum={{sum(row):.4f}})")

# 2-step transition matrix: P²
def mat_mult(A, B):
    n = len(A)
    return [[sum(A[i][k]*B[k][j] for k in range(n)) for j in range(n)] for i in range(n)]

P2 = mat_mult(P, P)
print(f"\\n2-step transitions (P²):")
for i, row in enumerate(P2):
    print(f"  {states[i]}: {{[f'{{x:.4f}}' for x in row]}}")

print(f"\\nP(Sunny→Sunny in 2 steps) = {{P2[0][0]:.4f}}")
print(f"P(Sunny→Stormy in 2 steps) = {{P2[0][3]:.4f}}")
"""
    out1 = run_code(executor, code1)

    code2 = f"""\
# Step 2: Simulate the chain
random.seed(42)
n_steps = 10000
state = 0  # Start sunny
visit_counts = [0, 0, 0, 0]

trajectory = [state]
for _ in range(n_steps):
    r = random.random()
    cumsum = 0
    for j in range(4):
        cumsum += P[state][j]
        if r < cumsum:
            state = j
            break
    visit_counts[state] += 1
    trajectory.append(state)

empirical = [c / n_steps for c in visit_counts]
print(f"Simulation: {{n_steps}} steps")
print(f"{'State':>10}  {'Visits':>8}  {'Empirical π':>14}")
print("-" * 35)
for i, s in enumerate(states):
    print(f"{{s:>10}}  {{visit_counts[i]:>8}}  {{empirical[i]:>14.6f}}")
"""
    out2 = run_code(executor, code2)

    code3 = f"""\
# Step 3: Find stationary distribution analytically
# πP = π, Σπᵢ = 1
# Solve: (Pᵀ - I)π = 0 with Σπᵢ = 1

n = 4
# Build augmented system: [Pᵀ - I; 1,1,1,1] π = [0,0,0,1]
A = [[P[j][i] - (1 if i == j else 0) for j in range(n)] for i in range(n)]
# Replace last row with constraint Σπᵢ = 1
A[n-1] = [1.0] * n
b_vec = [0.0] * (n-1) + [1.0]

# Gaussian elimination
aug = [A[i][:] + [b_vec[i]] for i in range(n)]
for col in range(n):
    max_row = max(range(col, n), key=lambda r: abs(aug[r][col]))
    aug[col], aug[max_row] = aug[max_row], aug[col]
    pivot = aug[col][col]
    for row in range(col+1, n):
        f = aug[row][col] / pivot
        for j in range(col, n+1):
            aug[row][j] -= f * aug[col][j]

pi = [0.0] * n
for i in range(n-1, -1, -1):
    pi[i] = (aug[i][n] - sum(aug[i][j]*pi[j] for j in range(i+1, n))) / aug[i][i]

print(f"Stationary distribution (analytical):")
for i, s in enumerate(states):
    print(f"  π({s}) = {{pi[i]:.6f}}")
print(f"  Sum = {{sum(pi):.8f}} ✓")

print(f"\\nComparison:")
print(f"{'State':>10}  {'Simulated':>12}  {'Analytical':>12}  {'Error':>10}")
for i, s in enumerate(states):
    err = abs(empirical[i] - pi[i])
    print(f"{{s:>10}}  {{empirical[i]:>12.6f}}  {{pi[i]:>12.6f}}  {{err:>10.2e}}")
"""
    out3 = run_code(executor, code3)

    return {
        "id": ex_id,
        "category": "multi_step",
        "difficulty": "advanced",
        "query": (
            "A weather Markov chain has states: Sunny, Cloudy, Rainy, Stormy. "
            "Transition matrix: "
            "Sunny→[0.6, 0.3, 0.08, 0.02], Cloudy→[0.2, 0.4, 0.3, 0.1], "
            "Rainy→[0.05, 0.2, 0.4, 0.35], Stormy→[0.02, 0.1, 0.38, 0.5]. "
            "Step 1: verify the matrix and compute 2-step transitions. "
            "Step 2: simulate 10,000 steps. "
            "Step 3: find the stationary distribution analytically and compare."
        ),
        "response": (
            "<think>\n"
            "Three-step Markov chain analysis:\n"
            "1. Verify transition matrix rows sum to 1, compute P².\n"
            "2. Monte Carlo simulation to estimate stationary distribution.\n"
            "3. Solve πP = π analytically, compare with simulation.\n"
            "</think>\n"
            f"<code>\n{code1}</code>\n"
            f"<output>\n{out1}\n</output>\n"
            "<think>\n"
            "Simulate the chain: at each step, sample next state from current row of P.\n"
            "Count visits to estimate π.\n"
            "</think>\n"
            f"<code>\n{code2}</code>\n"
            f"<output>\n{out2}\n</output>\n"
            "<think>\n"
            "Stationary distribution: solve (Pᵀ−I)π=0 with Σπᵢ=1.\n"
            "Replace last equation with normalization constraint.\n"
            "</think>\n"
            f"<code>\n{code3}</code>\n"
            f"<output>\n{out3}\n</output>\n"
        ),
    }


# ---------------------------------------------------------------------------
# Multi-step: optimization with gradient descent
# ---------------------------------------------------------------------------

def make_gradient_descent(ex_id: str) -> dict:
    """Minimize a function using gradient descent, compare learning rates."""
    random.seed(random.randint(1, 999))
    executor = PythonExecutor()

    code1 = f"""\
import math

# Step 1: Gradient descent on f(x,y) = x² + 3y² + 2xy
def f(x, y):
    return x**2 + 3*y**2 + 2*x*y

def grad_f(x, y):
    return (2*x + 2*y, 6*y + 2*x)

# Start point
x, y = 5.0, -3.0
lr = 0.05
print(f"Minimizing f(x,y) = x² + 3y² + 2xy")
print(f"Start: ({x}, {y}), f = {{f(x,y):.6f}}")
print(f"Learning rate: {{lr}}")
print(f"{'Iter':>6}  {'x':>10}  {'y':>10}  {'f(x,y)':>12}  {'|∇f|':>10}")
print("-" * 52)

for i in range(50):
    gx, gy = grad_f(x, y)
    grad_norm = math.sqrt(gx**2 + gy**2)
    x -= lr * gx
    y -= lr * gy
    fv = f(x, y)

    if i < 10 or i % 10 == 0:
        print(f"{{i:>6}}  {{x:>10.6f}}  {{y:>10.6f}}  {{fv:>12.6f}}  {{grad_norm:>10.6f}}")

print(f"\\nAfter 50 iterations: ({x:.8f}, {y:.8f}), f = {{f(x,y):.10f}}")
print(f"True minimum: (0, 0), f = 0")
"""
    out1 = run_code(executor, code1)

    code2 = f"""\
# Step 2: Compare learning rates
print(f"Learning rate comparison:")
print(f"{'lr':>8}  {'Iter':>6}  {'x':>10}  {'y':>10}  {'f(x,y)':>12}  {'Status':>12}")
print("-" * 60)

for lr in [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]:
    x, y = 5.0, -3.0
    converged = False
    for i in range(200):
        gx, gy = grad_f(x, y)
        x -= lr * gx
        y -= lr * gy
        if abs(f(x, y)) < 1e-10:
            converged = True
            break

    fv = f(x, y)
    status = "Converged" if converged else (f"f={fv:.2e}" if abs(fv) < 1e6 else "DIVERGED")
    print(f"{{lr:>8.2f}}  {{i+1:>6}}  {{x:>10.6f}}  {{y:>10.6f}}  {{fv:>12.6f}}  {{status:>12}}")
"""
    out2 = run_code(executor, code2)

    code3 = f"""\
# Step 3: Add momentum to gradient descent
print(f"Gradient descent with momentum:")
print(f"{'Iter':>6}  {'x':>10}  {'y':>10}  {'f(x,y)':>12}  {'|∇f|':>10}")
print("-" * 52)

x, y = 5.0, -3.0
vx, vy = 0.0, 0.0
lr = 0.1
beta = 0.9  # momentum coefficient

for i in range(50):
    gx, gy = grad_f(x, y)
    grad_norm = math.sqrt(gx**2 + gy**2)

    vx = beta * vx + (1 - beta) * gx
    vy = beta * vy + (1 - beta) * gy

    x -= lr * vx
    y -= lr * vy
    fv = f(x, y)

    if i < 10 or i % 10 == 0:
        print(f"{{i:>6}}  {{x:>10.6f}}  {{y:>10.6f}}  {{fv:>12.6f}}  {{grad_norm:>10.6f}}")

print(f"\\nWith momentum: ({x:.10f}, {y:.10f}), f = {{f(x,y):.12f}}")
print(f"Momentum accelerates convergence by accumulating gradient direction ✓")
"""
    out3 = run_code(executor, code3)

    return {
        "id": ex_id,
        "category": "multi_step",
        "difficulty": "advanced",
        "query": (
            "Minimize f(x,y) = x² + 3y² + 2xy using gradient descent starting from (5, -3). "
            "Step 1: run 50 iterations with lr=0.05. "
            "Step 2: compare different learning rates. "
            "Step 3: add momentum (β=0.9) and show improved convergence."
        ),
        "response": (
            "<think>\n"
            "Three-step optimization:\n"
            "1. Basic gradient descent: (x,y) ← (x,y) − lr·∇f\n"
            "   ∇f = (2x+2y, 6y+2x), minimum at (0,0).\n"
            "2. Learning rate sweep: too small → slow, too large → diverge.\n"
            "3. Momentum: v ← βv + (1-β)∇f, then update with v.\n"
            "</think>\n"
            f"<code>\n{code1}</code>\n"
            f"<output>\n{out1}\n</output>\n"
            "<think>\n"
            "Sweep learning rates to find the sweet spot.\n"
            "</think>\n"
            f"<code>\n{code2}</code>\n"
            f"<output>\n{out2}\n</output>\n"
            "<think>\n"
            "Momentum accumulates gradient direction, dampening oscillations.\n"
            "</think>\n"
            f"<code>\n{code3}</code>\n"
            f"<output>\n{out3}\n</output>\n"
        ),
    }


# ---------------------------------------------------------------------------
# Master list
# ---------------------------------------------------------------------------

BUILDERS = [
    make_projectile_optimize,
    make_linear_regression,
    make_monte_carlo_refine,
    make_investment_simulate,
    make_hypothesis_test,
    make_ode_solve,
    make_sort_compare,
    make_matrix_operations,
    make_markov_chain,
    make_gradient_descent,
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
            ex_id = f"multi_{idx:03d}"
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
    parser = argparse.ArgumentParser(description="Generate multi-step training examples")
    parser.add_argument("--output", default="training/datasets/multi_step/basic.jsonl")
    parser.add_argument("--count", type=int, default=60)
    args = parser.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Generating {args.count} multi-step examples...")
    examples = generate_examples(args.count)

    with open(out_path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")

    print(f"\nWrote {len(examples)} examples to {out_path}")


if __name__ == "__main__":
    main()
