#!/usr/bin/env python3
"""
Generate finance training examples.

Covers:
- Time value of money: PV, FV, NPV
- Loan amortization: monthly payment, schedule, total interest
- Portfolio returns: weighted average, CAGR, Sharpe ratio
- Bond pricing and yield
- Options: basic payoff diagrams, put-call parity
- Multi-step: project NPV vs IRR decision

Usage:
    python training/scripts/generate_finance.py
    python training/scripts/generate_finance.py --output training/datasets/finance/basic.jsonl
    python training/scripts/generate_finance.py --count 100
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
# Time value of money
# ---------------------------------------------------------------------------

def make_pv_fv(ex_id: str) -> dict:
    random.seed(random.randint(1, 999))
    PV = random.choice([1000, 5000, 10000, 25000])
    r = round(random.uniform(0.03, 0.10), 3)
    n = random.choice([5, 10, 15, 20])
    executor = PythonExecutor()
    code = f"""\
PV = {PV}    # present value ($)
r  = {r}     # annual interest rate
n  = {n}     # years

# Future value
FV = PV * (1 + r)**n

# Reverse: what PV gives FV in n years?
PV_check = FV / (1 + r)**n

print(f"Present Value:  ${PV:,}")
print(f"Rate:           {r*100:.1f}% per year")
print(f"Period:         {n} years")
print(f"Future Value:   ${{FV:,.2f}}")
print(f"Total growth:   ${{FV - PV:,.2f}} (+{{(FV/PV - 1)*100:.1f}}%)")
print(f"PV check:       ${{PV_check:,.2f}} ✓")
"""
    output = run_code(executor, code)
    FV = PV * (1 + r)**n
    return {
        "id": ex_id,
        "category": "finance",
        "difficulty": "basic",
        "query": (
            f"${PV:,} is invested at {r*100:.1f}% per year for {n} years. "
            "What is the future value?"
        ),
        "response": (
            "<think>\n"
            "Future value with compound interest: FV = PV × (1+r)^n\n"
            "</think>\n"
            f"<code>\n{code}</code>\n"
            f"<output>\n{output}\n</output>\n"
            f"After {n} years the investment grows to **${FV:,.2f}**."
        ),
    }


def make_npv(ex_id: str) -> dict:
    random.seed(random.randint(1, 999))
    initial = random.choice([50000, 100000, 200000])
    r = round(random.uniform(0.06, 0.12), 2)
    n_years = random.choice([4, 5, 6])
    # Cash flows that may or may not justify investment
    annual_cf = random.choice([15000, 20000, 25000, 30000, 40000])
    executor = PythonExecutor()
    code = f"""\
initial   = {initial}     # upfront investment (negative cash flow)
r         = {r}           # discount rate
cash_flows = [{annual_cf}] * {n_years}  # annual cash flows years 1-{n_years}

# NPV = -initial + sum(CF_t / (1+r)^t)
npv = -initial + sum(cf / (1+r)**t for t, cf in enumerate(cash_flows, 1))

print(f"Project NPV Analysis")
print(f"Initial investment: -${{initial:,}}")
print(f"Annual cash flow:   ${{cash_flows[0]:,}}/year for {{len(cash_flows)}} years")
print(f"Discount rate:      {{r*100:.0f}}%")
print()
print(f"Year | Cash Flow  | Discount Factor | PV of CF")
print(f"-----|------------|-----------------|----------")
total_pv = 0
for t, cf in enumerate(cash_flows, 1):
    df  = 1 / (1+r)**t
    pv  = cf * df
    total_pv += pv
    print(f"  {{t:1d}}  | ${{cf:>9,}} | {{df:.6f}}        | ${{pv:>9,.2f}}")
print(f"-----|------------|-----------------|----------")
print(f"     | Total PV of inflows: ${{total_pv:,.2f}}")
print()
print(f"NPV = ${{total_pv:,.2f}} - ${{initial:,}} = ${{npv:,.2f}}")
print(f"Decision: {{'ACCEPT' if npv > 0 else 'REJECT'}} ({'positive' if npv > 0 else 'negative'} NPV)")
"""
    output = run_code(executor, code)
    return {
        "id": ex_id,
        "category": "finance",
        "difficulty": "intermediate",
        "query": (
            f"A project requires a ${initial:,} upfront investment and generates "
            f"${annual_cf:,}/year for {n_years} years. With a discount rate of "
            f"{r*100:.0f}%, compute the NPV and decide whether to invest."
        ),
        "response": (
            "<think>\n"
            "NPV = sum of discounted future cash flows minus initial investment.\n"
            "NPV = -C0 + Σ(CF_t / (1+r)^t)\n"
            "Positive NPV → invest; negative NPV → reject.\n"
            "</think>\n"
            f"<code>\n{code}</code>\n"
            f"<output>\n{output}\n</output>\n"
        ),
    }


def make_irr(ex_id: str) -> dict:
    """Compute IRR by Newton's method and compare to hurdle rate."""
    random.seed(random.randint(1, 999))
    initial = random.choice([80000, 100000, 150000])
    hurdle = round(random.uniform(0.08, 0.15), 2)
    cfs = [random.randint(20000, 40000) for _ in range(5)]
    executor = PythonExecutor()
    cfs_str = ", ".join(str(c) for c in cfs)
    code = f"""\
initial    = {initial}
hurdle     = {hurdle}
cash_flows = [{cfs_str}]   # years 1-{len(cfs)}

# Compute IRR: rate at which NPV = 0
# Use Newton's method (bisection fallback)
def npv_at(r):
    return -initial + sum(cf/(1+r)**t for t,cf in enumerate(cash_flows,1))

# Bisect between 0% and 200%
lo, hi = 0.0001, 2.0
for _ in range(100):
    mid = (lo + hi) / 2
    if npv_at(mid) > 0:
        lo = mid
    else:
        hi = mid
irr = (lo + hi) / 2

npv_hurdle = npv_at(hurdle)

print(f"IRR Analysis")
print(f"Initial investment:  -${{initial:,}}")
print(f"Cash flows (yr 1-{len(cfs)}): {cfs_str}")
print()
print(f"IRR:          {{irr*100:.2f}}%")
print(f"Hurdle rate:  {{hurdle*100:.1f}}%")
print(f"NPV at hurdle rate: ${{npv_hurdle:,.2f}}")
print()
if irr > hurdle:
    print(f"Decision: ACCEPT (IRR {{irr*100:.1f}}% > hurdle {{hurdle*100:.1f}}%)")
else:
    print(f"Decision: REJECT (IRR {{irr*100:.1f}}% < hurdle {{hurdle*100:.1f}}%)")
"""
    output = run_code(executor, code)
    return {
        "id": ex_id,
        "category": "finance",
        "difficulty": "intermediate",
        "query": (
            f"A ${initial:,} project generates cash flows of ${', $'.join(str(c) for c in cfs)} "
            f"over 5 years. Compute the IRR and compare it to a {hurdle*100:.0f}% hurdle rate."
        ),
        "response": (
            "<think>\n"
            "IRR is the discount rate that makes NPV = 0.\n"
            "No closed form for 5+ periods — solve numerically (bisection between 0% and 200%).\n"
            "If IRR > hurdle rate → accept; if IRR < hurdle rate → reject.\n"
            "</think>\n"
            f"<code>\n{code}</code>\n"
            f"<output>\n{output}\n</output>\n"
        ),
    }


# ---------------------------------------------------------------------------
# Loan amortization
# ---------------------------------------------------------------------------

def make_mortgage(ex_id: str) -> dict:
    random.seed(random.randint(1, 999))
    principal = random.choice([150000, 200000, 300000, 500000])
    annual_r = round(random.uniform(0.04, 0.08), 3)
    years = random.choice([15, 20, 25, 30])
    executor = PythonExecutor()
    code = f"""\
principal = {principal}
annual_r  = {annual_r}
years     = {years}
n_months  = years * 12
r_monthly = annual_r / 12

# Monthly payment formula: M = P * r*(1+r)^n / ((1+r)^n - 1)
M = principal * r_monthly * (1+r_monthly)**n_months / ((1+r_monthly)**n_months - 1)

total_paid    = M * n_months
total_interest = total_paid - principal

print(f"Mortgage: ${{{principal}:,}} at {{annual_r*100:.2f}}% for {{years}} years")
print(f"Monthly payment:   ${{M:,.2f}}")
print(f"Total paid:        ${{total_paid:,.2f}}")
print(f"Total interest:    ${{total_interest:,.2f}} ({{total_interest/principal*100:.1f}}% of principal)")
print()

# First few rows of amortization schedule
print(f"{'Month':>6} {'Payment':>10} {'Interest':>10} {'Principal':>10} {'Balance':>12}")
print("-" * 52)
balance = principal
for month in range(1, min(7, n_months+1)):
    interest_part  = balance * r_monthly
    principal_part = M - interest_part
    balance       -= principal_part
    print(f"{{month:>6}} ${{M:>9,.2f}} ${{interest_part:>9,.2f}} ${{principal_part:>9,.2f}} ${{max(balance,0):>11,.2f}}")
print("  ...")
"""
    output = run_code(executor, code)
    return {
        "id": ex_id,
        "category": "finance",
        "difficulty": "intermediate",
        "query": (
            f"Calculate the monthly payment on a ${principal:,} mortgage at "
            f"{annual_r*100:.2f}% annual interest for {years} years. "
            "Show total interest paid and the first 6 rows of the amortization schedule."
        ),
        "response": (
            "<think>\n"
            "Monthly payment formula: M = P·r·(1+r)^n / ((1+r)^n − 1)\n"
            "where r = monthly rate = annual/12, n = total months.\n"
            "Each month: interest = balance × r; principal = M − interest.\n"
            "</think>\n"
            f"<code>\n{code}</code>\n"
            f"<output>\n{output}\n</output>\n"
        ),
    }


# ---------------------------------------------------------------------------
# Portfolio returns
# ---------------------------------------------------------------------------

def make_portfolio_return(ex_id: str) -> dict:
    random.seed(random.randint(1, 999))
    n_assets = random.choice([3, 4])
    weights = [round(random.uniform(0.1, 0.5), 2) for _ in range(n_assets)]
    total = sum(weights)
    weights = [round(w/total, 3) for w in weights]
    # normalise to exactly 1
    weights[-1] = round(1 - sum(weights[:-1]), 3)
    returns = [round(random.uniform(-0.05, 0.20), 3) for _ in range(n_assets)]
    names = ["AAPL", "MSFT", "GOOG", "AMZN", "TSLA"][:n_assets]
    executor = PythonExecutor()
    assets_code = "\n".join(
        f"    ('{names[i]}', {weights[i]}, {returns[i]}),"
        for i in range(n_assets)
    )
    code = f"""\
import math

# (ticker, weight, annual_return)
portfolio = [
{assets_code}
]

names   = [a[0] for a in portfolio]
weights = [a[1] for a in portfolio]
rets    = [a[2] for a in portfolio]

# Weighted portfolio return
port_return = sum(w*r for w,r in zip(weights, rets))

print(f"Portfolio Analysis")
print(f"{{'-'*45}}")
print(f"{{' Asset':<8}} {{' Weight':>8}} {{' Return':>8}} {{' Contrib':>10}}")
print(f"{{'-'*45}}")
for name, w, r in portfolio:
    contrib = w * r
    print(f"  {{name:<6}} {{w*100:>7.1f}}%  {{r*100:>7.1f}}%  {{contrib*100:>8.2f}}%")
print(f"{{'-'*45}}")
print(f"  Portfolio return: {{port_return*100:.2f}}%")

# Annualised return from monthly returns (hypothetical)
monthly_r = (1 + port_return)**(1/12) - 1
print(f"  Implied monthly:  {{monthly_r*100:.3f}}%")

# Variance (assume zero correlation for simplicity)
vols = [0.15, 0.18, 0.20, 0.25, 0.30][:len(portfolio)]
port_var = sum((w*v)**2 for w,v in zip(weights,vols))
port_std = math.sqrt(port_var)
rf = 0.04  # risk-free rate
sharpe = (port_return - rf) / port_std
print(f"  Port. volatility: {{port_std*100:.2f}}% (approx, no correlation)")
print(f"  Sharpe ratio:     {{sharpe:.3f}}")
"""
    output = run_code(executor, code)
    return {
        "id": ex_id,
        "category": "finance",
        "difficulty": "intermediate",
        "query": (
            f"A portfolio holds {', '.join(names)} with weights "
            f"{', '.join(str(int(w*100))+'%' for w in weights)} and annual returns "
            f"{', '.join(str(int(r*100))+'%' for r in returns)}. "
            "Compute the weighted portfolio return, per-asset contribution, and Sharpe ratio."
        ),
        "response": (
            "<think>\n"
            "Portfolio return = Σ weight_i × return_i\n"
            "Sharpe ratio = (portfolio return − risk-free) / portfolio std dev\n"
            "For std dev with assumed zero correlation: σ_p = √Σ(w_i × σ_i)²\n"
            "</think>\n"
            f"<code>\n{code}</code>\n"
            f"<output>\n{output}\n</output>\n"
        ),
    }


def make_cagr(ex_id: str) -> dict:
    random.seed(random.randint(1, 999))
    start = random.choice([10000, 50000, 100000])
    end = random.choice([15000, 30000, 80000, 200000])
    years = random.choice([3, 5, 7, 10])
    executor = PythonExecutor()
    code = f"""\
import math
start = {start}
end   = {end}
years = {years}

# CAGR: Compound Annual Growth Rate
cagr = (end / start)**(1/years) - 1

# Compare: simple annual growth rate (not compounded)
simple_rate = (end - start) / start / years

# Doubling time (rule of 72)
doubling_years = math.log(2) / math.log(1 + cagr) if cagr > 0 else float('inf')
rule_of_72 = 72 / (cagr * 100) if cagr > 0 else float('inf')

print(f"Growth Analysis")
print(f"Start: ${{start:,}}  →  End: ${{end:,}}  over {{years}} years")
print(f"Total return: {{(end/start - 1)*100:.1f}}%")
print(f"CAGR: {{cagr*100:.3f}}% per year")
print(f"Simple rate: {{simple_rate*100:.3f}}% per year (non-compounded)")
print(f"At this CAGR, investment doubles in {{doubling_years:.1f}} years")
print(f"Rule of 72 estimate: {{rule_of_72:.1f}} years")
"""
    output = run_code(executor, code)
    return {
        "id": ex_id,
        "category": "finance",
        "difficulty": "basic",
        "query": (
            f"An investment grew from ${start:,} to ${end:,} over {years} years. "
            "Calculate the CAGR and how many years it would take to double at that rate."
        ),
        "response": (
            "<think>\n"
            "CAGR = (End/Start)^(1/years) − 1\n"
            "Doubling time: solve 2 = (1+CAGR)^t → t = ln(2)/ln(1+CAGR)\n"
            "Rule of 72: approximately 72/CAGR%\n"
            "</think>\n"
            f"<code>\n{code}</code>\n"
            f"<output>\n{output}\n</output>\n"
        ),
    }


# ---------------------------------------------------------------------------
# Bond pricing
# ---------------------------------------------------------------------------

def make_bond_price(ex_id: str) -> dict:
    random.seed(random.randint(1, 999))
    face = 1000
    coupon_r = round(random.uniform(0.03, 0.08), 2)
    ytm = round(random.uniform(0.03, 0.10), 2)
    years = random.choice([5, 10, 20])
    executor = PythonExecutor()
    code = f"""\
face    = {face}
coupon_r = {coupon_r}   # coupon rate
ytm     = {ytm}         # yield to maturity (market rate)
years   = {years}

coupon = face * coupon_r

# Bond price = PV of coupons + PV of face value
pv_coupons = coupon * (1 - (1+ytm)**(-years)) / ytm
pv_face    = face / (1+ytm)**years
price      = pv_coupons + pv_face

current_yield = coupon / price

print(f"Bond Pricing")
print(f"Face value: ${face:,}, Coupon: {coupon_r*100:.1f}% (${coupon:.0f}/yr), YTM: {ytm*100:.1f}%, Maturity: {years}yr")
print(f"PV of coupons:    ${{pv_coupons:,.2f}}")
print(f"PV of face value: ${{pv_face:,.2f}}")
print(f"Bond price:       ${{price:,.2f}}")
print(f"Current yield:    {{current_yield*100:.3f}}%")
if ytm > coupon_r:
    print(f"Trading at DISCOUNT (YTM > coupon rate, price < face)")
elif ytm < coupon_r:
    print(f"Trading at PREMIUM (YTM < coupon rate, price > face)")
else:
    print(f"Trading at PAR (YTM = coupon rate)")
"""
    output = run_code(executor, code)
    return {
        "id": ex_id,
        "category": "finance",
        "difficulty": "intermediate",
        "query": (
            f"Price a {years}-year bond with face value ${face:,}, "
            f"coupon rate {coupon_r*100:.1f}%, and YTM of {ytm*100:.1f}%."
        ),
        "response": (
            "<think>\n"
            "Bond price = PV of all cash flows discounted at YTM.\n"
            "PV(coupons) = C × [1 − (1+ytm)^(−n)] / ytm\n"
            "PV(face) = F / (1+ytm)^n\n"
            "If YTM > coupon rate → discount; YTM < coupon rate → premium.\n"
            "</think>\n"
            f"<code>\n{code}</code>\n"
            f"<output>\n{output}\n</output>\n"
        ),
    }


# ---------------------------------------------------------------------------
# Multi-step: NPV + sensitivity analysis
# ---------------------------------------------------------------------------

def make_npv_sensitivity(ex_id: str) -> dict:
    """Show NPV across a range of discount rates."""
    random.seed(random.randint(1, 999))
    initial = random.choice([100000, 200000])
    cfs = [random.randint(25000, 60000) for _ in range(5)]
    executor = PythonExecutor()
    cfs_str = ", ".join(str(c) for c in cfs)
    code1 = f"""\
initial    = {initial}
cash_flows = [{cfs_str}]

def compute_npv(r):
    return -initial + sum(cf/(1+r)**t for t,cf in enumerate(cash_flows,1))

# Base case NPV at 10%
base_r   = 0.10
base_npv = compute_npv(base_r)
print(f"Base case NPV at {{base_r*100:.0f}}%: ${{base_npv:,.2f}}")
print(f"Cash flows: {{cash_flows}}")
print()

# Sensitivity: NPV vs discount rate
print(f"{{' Rate':>6}} | {{' NPV':>12}}")
print("-" * 22)
for pct in range(0, 26, 5):
    r   = pct / 100
    npv = compute_npv(r)
    bar = "+" * max(0, int(npv/5000)) if npv > 0 else "-" * max(0, int(-npv/5000))
    print(f"{{pct:>5}}% | ${{npv:>11,.0f}}  {{bar[:20]}}")
"""
    output1 = run_code(executor, code1)

    code2 = """\
# Find IRR (where NPV = 0)
lo, hi = 0.0001, 3.0
for _ in range(120):
    mid = (lo + hi) / 2
    if compute_npv(mid) > 0:
        lo = mid
    else:
        hi = mid
irr = (lo + hi) / 2
print(f"IRR = {irr*100:.2f}%")
print(f"Project earns positive NPV for any discount rate below {irr*100:.2f}%")
"""
    output2 = run_code(executor, code2)

    return {
        "id": ex_id,
        "category": "finance",
        "difficulty": "advanced",
        "query": (
            f"A project costs ${initial:,} upfront and generates ${', $'.join(str(c) for c in cfs)} "
            "over 5 years. Step 1: compute NPV at rates from 0% to 25% (sensitivity table). "
            "Step 2: find the IRR."
        ),
        "response": (
            "<think>\n"
            "Build a reusable npv(r) function, evaluate it over a range, then solve for IRR "
            "using bisection. The IRR is the breakeven rate — above it the project destroys value.\n"
            "</think>\n"
            f"<code>\n{code1}</code>\n"
            f"<output>\n{output1}\n</output>\n"
            "<think>\n"
            "The sensitivity table shows NPV sign change. Now pinpoint IRR with bisection.\n"
            "compute_npv is still in scope from the previous block.\n"
            "</think>\n"
            f"<code>\n{code2}</code>\n"
            f"<output>\n{output2}\n</output>\n"
        ),
    }


# ---------------------------------------------------------------------------
# Master list
# ---------------------------------------------------------------------------

BUILDERS = [
    make_pv_fv,
    make_npv,
    make_irr,
    make_mortgage,
    make_portfolio_return,
    make_cagr,
    make_bond_price,
    make_npv_sensitivity,
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
            ex_id = f"fin_{idx:03d}"
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
    parser = argparse.ArgumentParser(description="Generate finance training examples")
    parser.add_argument("--output", default="training/datasets/finance/basic.jsonl")
    parser.add_argument("--count", type=int, default=100)
    args = parser.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Generating {args.count} finance examples...")
    examples = generate_examples(args.count)

    with open(out_path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")

    print(f"\nWrote {len(examples)} examples to {out_path}")


if __name__ == "__main__":
    main()
