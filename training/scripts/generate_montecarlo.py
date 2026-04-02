#!/usr/bin/env python3
"""
Generate Monte Carlo simulation training examples.

Covers:
- Pi estimation via random sampling
- Numerical integration
- Random walk / Brownian motion
- Option pricing (Black-Scholes vs MC)
- Risk simulation (project cost overrun)
- Multi-step: combine sampling + convergence analysis

Usage:
    python training/scripts/generate_montecarlo.py
    python training/scripts/generate_montecarlo.py --output training/datasets/montecarlo/basic.jsonl
    python training/scripts/generate_montecarlo.py --count 80
"""

import sys
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
# Example builders
# ---------------------------------------------------------------------------

def make_pi_estimation(ex_id: str) -> dict:
    n_samples = random.choice([10_000, 50_000, 100_000])
    seed = random.randint(1, 999)
    executor = PythonExecutor(timeout=30)
    code = f"""\
import random, math
random.seed({seed})
n = {n_samples}
inside = 0
for _ in range(n):
    x, y = random.uniform(-1, 1), random.uniform(-1, 1)
    if x*x + y*y <= 1:
        inside += 1
pi_est = 4 * inside / n
error = abs(pi_est - math.pi)
print(f"Samples: {{n}}")
print(f"Points inside unit circle: {{inside}}")
print(f"Pi estimate: {{pi_est:.6f}}")
print(f"True pi: {{math.pi:.6f}}")
print(f"Absolute error: {{error:.6f}}")
"""
    output = run_code(executor, code)
    return {
        "id": ex_id,
        "category": "montecarlo",
        "difficulty": "basic",
        "query": (
            f"Use Monte Carlo sampling with {n_samples:,} random points to estimate π. "
            "Throw points uniformly in the square [-1,1]×[-1,1] and check how many land "
            "inside the unit circle. Report the estimate and error."
        ),
        "response": (
            "<think>\n"
            "Monte Carlo pi estimation: ratio of points inside unit circle to total = π/4.\n"
            "Use random (x,y) in [-1,1]×[-1,1]; point is inside if x²+y² ≤ 1.\n"
            "</think>\n"
            f"<code>\n{code}</code>\n"
            f"<output>\n{output}\n</output>\n"
            "The Monte Carlo estimate of π converges to the true value as sample count "
            "increases. With more samples the error decreases roughly as 1/√n."
        ),
    }


def make_integration(ex_id: str) -> dict:
    # Integrate sin(x) from 0 to pi => 2, or x^2 from 0 to 1 => 1/3
    choice = random.randint(0, 1)
    seed = random.randint(1, 999)
    executor = PythonExecutor(timeout=30)
    if choice == 0:
        func_str = "math.sin(x)"
        a, b = 0, "math.pi"
        a_disp, b_disp = "0", "π"
        true_val = "2.0"
        name = "sin(x)"
        n = random.choice([50_000, 100_000])
        code = f"""\
import random, math
random.seed({seed})
n = {n}
a, b = 0, math.pi
total = sum(math.sin(random.uniform(a, b)) for _ in range(n))
mc_integral = (b - a) * total / n
print(f"Integrating sin(x) from 0 to π using {{n:,}} samples")
print(f"MC estimate:  {{mc_integral:.6f}}")
print(f"True value:   2.000000")
print(f"Error:        {{abs(mc_integral - 2):.6f}}")
"""
    else:
        func_str = "x**2"
        a, b = 0, 1
        a_disp, b_disp = "0", "1"
        true_val = "0.333333..."
        name = "x²"
        n = random.choice([50_000, 100_000])
        code = f"""\
import random
random.seed({seed})
n = {n}
total = sum(random.uniform(0, 1)**2 for _ in range(n))
mc_integral = total / n
print(f"Integrating x² from 0 to 1 using {{n:,}} samples")
print(f"MC estimate:  {{mc_integral:.6f}}")
print(f"True value:   0.333333")
print(f"Error:        {{abs(mc_integral - 1/3):.6f}}")
"""
    output = run_code(executor, code)
    return {
        "id": ex_id,
        "category": "montecarlo",
        "difficulty": "basic",
        "query": (
            f"Use Monte Carlo integration with {n:,} samples to estimate ∫{name} dx "
            f"from {a_disp} to {b_disp}. Compare with the exact answer."
        ),
        "response": (
            "<think>\n"
            "Monte Carlo integration: draw uniform samples x in [a,b], average f(x), "
            "multiply by (b-a). The estimate converges to the true integral.\n"
            "</think>\n"
            f"<code>\n{code}</code>\n"
            f"<output>\n{output}\n</output>\n"
            f"Monte Carlo integration gives a good approximation. "
            f"The true value is {true_val} and the error shrinks as we add more samples."
        ),
    }


def make_random_walk(ex_id: str) -> dict:
    steps = random.choice([1000, 5000, 10000])
    n_walks = random.choice([500, 1000])
    seed = random.randint(1, 999)
    executor = PythonExecutor(timeout=30)
    code = f"""\
import random, math
random.seed({seed})
steps = {steps}
n_walks = {n_walks}

final_positions = []
for _ in range(n_walks):
    pos = 0
    for _ in range(steps):
        pos += 1 if random.random() < 0.5 else -1
    final_positions.append(pos)

mean_pos = sum(final_positions) / n_walks
mean_sq = sum(x*x for x in final_positions) / n_walks
std_dev = math.sqrt(mean_sq)
theoretical_std = math.sqrt(steps)

print(f"Random walk: {{steps}} steps, {{n_walks}} simulations")
print(f"Mean final position:   {{mean_pos:.3f}}  (expected: 0)")
print(f"Std dev of positions:  {{std_dev:.3f}}")
print(f"Theoretical std (√n):  {{theoretical_std:.3f}}")
print(f"Ratio (actual/theory): {{std_dev/theoretical_std:.4f}}")
"""
    output = run_code(executor, code)
    return {
        "id": ex_id,
        "category": "montecarlo",
        "difficulty": "intermediate",
        "query": (
            f"Simulate {n_walks:,} 1D random walks of {steps:,} steps each (±1 with equal probability). "
            "Report the mean final position and standard deviation, and compare with the "
            "theoretical prediction (std = √n)."
        ),
        "response": (
            "<think>\n"
            "For an unbiased 1D random walk of n steps, the expected final position is 0 "
            "and the standard deviation is √n. Simulate many walks to verify.\n"
            "</think>\n"
            f"<code>\n{code}</code>\n"
            f"<output>\n{output}\n</output>\n"
            "The simulated standard deviation closely matches the theoretical √n prediction, "
            "confirming the diffusive nature of random walks."
        ),
    }


def make_option_pricing(ex_id: str) -> dict:
    seed = random.randint(1, 999)
    S0 = random.choice([100, 110, 95])
    K = random.choice([100, 105, 110])
    T = random.choice([0.5, 1.0])
    r = 0.05
    sigma = random.choice([0.2, 0.25, 0.3])
    n_sim = random.choice([50_000, 100_000])
    executor = PythonExecutor(timeout=30)
    code = f"""\
import random, math
random.seed({seed})

S0 = {S0}    # current stock price
K  = {K}     # strike price
T  = {T}     # time to expiry (years)
r  = {r}     # risk-free rate
sigma = {sigma}  # volatility

n_sim = {n_sim}
payoffs = []
for _ in range(n_sim):
    # Geometric Brownian Motion terminal price
    Z = random.gauss(0, 1)
    ST = S0 * math.exp((r - 0.5*sigma**2)*T + sigma*math.sqrt(T)*Z)
    payoffs.append(max(ST - K, 0))  # European call payoff

mc_price = math.exp(-r*T) * sum(payoffs) / n_sim

# Black-Scholes analytical price
d1 = (math.log(S0/K) + (r + 0.5*sigma**2)*T) / (sigma*math.sqrt(T))
d2 = d1 - sigma*math.sqrt(T)
def N(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))
bs_price = S0*N(d1) - K*math.exp(-r*T)*N(d2)

print(f"European Call Option Pricing")
print(f"  S0={{S0}}, K={{K}}, T={{T}}yr, r={{r}}, σ={{sigma}}")
print(f"Monte Carlo price ({n_sim:,} sims): ${{mc_price:.4f}}")
print(f"Black-Scholes price:              ${{bs_price:.4f}}")
print(f"Difference: ${{abs(mc_price - bs_price):.4f}}")
"""
    output = run_code(executor, code)
    return {
        "id": ex_id,
        "category": "montecarlo",
        "difficulty": "intermediate",
        "query": (
            f"Price a European call option using Monte Carlo simulation ({n_sim:,} paths). "
            f"Stock price S₀={S0}, strike K={K}, T={T} year(s), risk-free rate r={r}, "
            f"volatility σ={sigma}. Compare with the Black-Scholes analytical price."
        ),
        "response": (
            "<think>\n"
            "Monte Carlo option pricing: simulate GBM paths, compute payoff max(S_T - K, 0), "
            "discount back at risk-free rate. Compare to analytical Black-Scholes.\n"
            f"GBM: S_T = S0 * exp((r - σ²/2)T + σ√T * Z) where Z ~ N(0,1)\n"
            "</think>\n"
            f"<code>\n{code}</code>\n"
            f"<output>\n{output}\n</output>\n"
            "The Monte Carlo price converges to the Black-Scholes price as paths increase. "
            "MC is more flexible — it generalizes to exotic options where no closed form exists."
        ),
    }


def make_risk_simulation(ex_id: str) -> dict:
    seed = random.randint(1, 999)
    n_components = random.choice([5, 6, 7])
    budget = random.choice([500_000, 750_000, 1_000_000])
    n_sim = random.choice([20_000, 50_000])
    executor = PythonExecutor(timeout=30)
    code = f"""\
import random, math
random.seed({seed})

# Project cost components: (base_cost, uncertainty_fraction)
components = [
    ("Design",       50000,  0.10),
    ("Development",  200000, 0.20),
    ("Testing",      80000,  0.15),
    ("Deployment",   40000,  0.10),
    ("Contingency",  30000,  0.30),
][:{ n_components }]

budget = {budget}
n_sim = {n_sim}

overruns = 0
total_costs = []
for _ in range(n_sim):
    cost = sum(base * (1 + random.uniform(-unc, unc))
               for _, base, unc in components)
    total_costs.append(cost)
    if cost > budget:
        overruns += 1

total_costs.sort()
p50 = total_costs[int(0.50 * n_sim)]
p90 = total_costs[int(0.90 * n_sim)]
p95 = total_costs[int(0.95 * n_sim)]
prob_overrun = overruns / n_sim * 100

print(f"Project Cost Risk Simulation ({{n_sim:,}} trials)")
print(f"Budget: ${{budget:,}}")
print(f"P50 cost: ${{p50:,.0f}}")
print(f"P90 cost: ${{p90:,.0f}}")
print(f"P95 cost: ${{p95:,.0f}}")
print(f"Probability of budget overrun: {{prob_overrun:.1f}}%")
"""
    output = run_code(executor, code)
    return {
        "id": ex_id,
        "category": "montecarlo",
        "difficulty": "intermediate",
        "query": (
            f"A project has {n_components} cost components, each with a base cost and "
            f"uncertainty range. Budget is ${budget:,}. Run a Monte Carlo simulation "
            f"({n_sim:,} trials) to find P50, P90, P95 cost estimates and the "
            "probability of exceeding budget."
        ),
        "response": (
            "<think>\n"
            "Monte Carlo risk simulation: for each trial, draw each cost component from "
            "its uncertainty range, sum to get total project cost. Run many trials to "
            "build a distribution and read off percentiles.\n"
            "</think>\n"
            f"<code>\n{code}</code>\n"
            f"<output>\n{output}\n</output>\n"
            "The percentile outputs give a risk profile: P50 is the median outcome, "
            "P90 means only 10% of scenarios exceed this cost. Budget overrun probability "
            "directly informs risk management decisions."
        ),
    }


def make_convergence_analysis(ex_id: str) -> dict:
    """Multi-step: estimate pi at increasing sample sizes to show convergence."""
    seed = random.randint(1, 999)
    executor = PythonExecutor(timeout=30)
    code1 = f"""\
import random, math
random.seed({seed})

# Step 1: estimate pi at several sample sizes
sample_sizes = [100, 1000, 10000, 100000]
results = []
for n in sample_sizes:
    random.seed({seed})  # reset for fair comparison
    inside = sum(1 for _ in range(n)
                 if random.uniform(0,1)**2 + random.uniform(0,1)**2 <= 1)
    pi_est = 4 * inside / n
    results.append((n, pi_est, abs(pi_est - math.pi)))
    print(f"n={{n:>7,}}  pi≈{{pi_est:.6f}}  error={{abs(pi_est - math.pi):.6f}}")
"""
    output1 = run_code(executor, code1)

    code2 = """\
import math
# Step 2: analyse convergence rate
# Theory: error ~ 1/sqrt(n), so log(error) ~ -0.5*log(n)
import math

# Use our results
ns   = [100, 1000, 10000, 100000]
errs = [abs(r[1] - math.pi) for r in results]

# Fit slope in log-log space
log_n    = [math.log10(n)   for n in ns]
log_err  = [math.log10(max(e, 1e-9)) for e in errs]
n_pts = len(ns)
mean_x = sum(log_n) / n_pts
mean_y = sum(log_err) / n_pts
slope = sum((x-mean_x)*(y-mean_y) for x,y in zip(log_n,log_err)) / \
        sum((x-mean_x)**2 for x in log_n)

print(f"\\nConvergence analysis (log-log slope):")
print(f"  Fitted slope: {slope:.3f}  (theoretical: -0.500)")
print(f"  Interpretation: 10x more samples → ~{10**(-slope):.2f}x smaller error")
"""
    output2 = run_code(executor, code2)

    return {
        "id": ex_id,
        "category": "montecarlo",
        "difficulty": "advanced",
        "query": (
            "Demonstrate Monte Carlo convergence for π estimation. "
            "First, estimate π at sample sizes 100, 1000, 10000, 100000 and tabulate the error. "
            "Then analyse the convergence rate by fitting a log-log slope and compare "
            "with the theoretical 1/√n rate."
        ),
        "response": (
            "<think>\n"
            "Monte Carlo error scales as 1/√n. I'll estimate pi at four sample sizes, "
            "then fit the log-log relationship between n and error to measure the exponent.\n"
            "</think>\n"
            f"<code>\n{code1}</code>\n"
            f"<output>\n{output1}\n</output>\n"
            "<think>\n"
            "Now compute the log-log slope. The list `results` is still in scope.\n"
            "</think>\n"
            f"<code>\n{code2}</code>\n"
            f"<output>\n{output2}\n</output>\n"
            "The log-log slope is close to -0.5, confirming the theoretical 1/√n convergence. "
            "This means quadrupling the sample count halves the error — Monte Carlo is slow "
            "to converge but unaffected by problem dimensionality."
        ),
    }


def make_bootstrap_ci(ex_id: str) -> dict:
    seed = random.randint(1, 999)
    n_sample = random.choice([30, 50, 100])
    n_boot = random.choice([5000, 10000])
    executor = PythonExecutor(timeout=30)
    code = f"""\
import random, math
random.seed({seed})

# Original sample (e.g., measured response times in ms)
n = {n_sample}
data = [random.gauss(200, 40) for _ in range(n)]
sample_mean = sum(data) / n

# Bootstrap
n_boot = {n_boot}
boot_means = []
for _ in range(n_boot):
    resample = [random.choice(data) for _ in range(n)]
    boot_means.append(sum(resample) / n)

boot_means.sort()
ci_low  = boot_means[int(0.025 * n_boot)]
ci_high = boot_means[int(0.975 * n_boot)]

# Analytical 95% CI for comparison
import math
sem = math.sqrt(sum((x - sample_mean)**2 for x in data) / (n-1)) / math.sqrt(n)
z = 1.96
analytical_low  = sample_mean - z * sem
analytical_high = sample_mean + z * sem

print(f"Sample: n={{n}}, mean={{sample_mean:.2f}} ms")
print(f"Bootstrap 95% CI:  [{{ci_low:.2f}}, {{ci_high:.2f}}]")
print(f"Analytical 95% CI: [{{analytical_low:.2f}}, {{analytical_high:.2f}}]")
print(f"Bootstrap width: {{ci_high - ci_low:.2f}} ms")
"""
    output = run_code(executor, code)
    return {
        "id": ex_id,
        "category": "montecarlo",
        "difficulty": "intermediate",
        "query": (
            f"You have a sample of {n_sample} response-time measurements (drawn from a "
            "normal distribution). Compute a 95% confidence interval for the mean using "
            f"bootstrap resampling ({n_boot:,} iterations) and compare it with the "
            "analytical t-based CI."
        ),
        "response": (
            "<think>\n"
            "Bootstrap CI: repeatedly resample with replacement from the original data, "
            "compute the statistic each time, then read off the 2.5th and 97.5th percentiles.\n"
            "No distributional assumptions needed.\n"
            "</think>\n"
            f"<code>\n{code}</code>\n"
            f"<output>\n{output}\n</output>\n"
            "Bootstrap and analytical CIs agree closely for this sample size. "
            "The bootstrap approach is more general — it works for any statistic "
            "(median, variance, correlation) without assuming normality."
        ),
    }


# ---------------------------------------------------------------------------
# Master list
# ---------------------------------------------------------------------------

BUILDERS = [
    make_pi_estimation,
    make_integration,
    make_random_walk,
    make_option_pricing,
    make_risk_simulation,
    make_convergence_analysis,
    make_bootstrap_ci,
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
            ex_id = f"mc_{idx:03d}"
            try:
                ex = builder(ex_id)
                examples.append(ex)
                idx += 1
                print(f"  {ex_id}: {ex['query'][:70]}...")
            except Exception as e:
                print(f"  SKIP {ex_id} ({builder.__name__}): {e}")
    return examples


def main():
    parser = argparse.ArgumentParser(description="Generate Monte Carlo training examples")
    parser.add_argument("--output", default="training/datasets/montecarlo/basic.jsonl")
    parser.add_argument("--count", type=int, default=70)
    args = parser.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Generating {args.count} Monte Carlo examples...")
    examples = generate_examples(args.count)

    with open(out_path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")

    print(f"\nWrote {len(examples)} examples to {out_path}")


if __name__ == "__main__":
    main()
