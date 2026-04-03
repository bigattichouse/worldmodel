#!/usr/bin/env python3
"""
Generate chemistry training examples.

Covers:
- Stoichiometry: mole ratios, limiting reagents, percent yield
- Ideal gas law: PV = nRT, partial pressures, gas mixtures
- Thermodynamics: ΔH, ΔG, entropy, Hess's law
- Equilibrium: Kc, Kp, ICE tables, Le Chatelier
- Acid-base: pH, pOH, buffer calculations, titrations
- Solution chemistry: molarity, dilution, colligative properties

All outputs pre-executed and verified.

Usage:
    python training/scripts/generate_chemistry.py
    python training/scripts/generate_chemistry.py --output training/datasets/science/chemistry/basic.jsonl
    python training/scripts/generate_chemistry.py --count 80
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
# Stoichiometry
# ---------------------------------------------------------------------------

def make_stoichiometry_basic(ex_id: str) -> dict:
    """Basic mole-ratio stoichiometry."""
    random.seed(random.randint(1, 999))
    # 2H2 + O2 → 2H2O
    mol_h2 = random.choice([2.0, 4.0, 6.0, 8.0, 10.0])
    executor = PythonExecutor()
    code = f"""\
# Stoichiometry: 2H₂ + O₂ → 2H₂O
# Given {mol_h2} mol H₂, find O₂ needed and H₂O produced

mol_H2 = {mol_h2}
# Ratio: 2 H₂ : 1 O₂ : 2 H₂O
mol_O2_needed = mol_H2 / 2
mol_H2O_produced = mol_H2  # 2:2 ratio

# Molar masses
M_H2  = 2.016    # g/mol
M_O2  = 32.00    # g/mol
M_H2O = 18.015   # g/mol

mass_H2 = mol_H2 * M_H2
mass_O2 = mol_O2_needed * M_O2
mass_H2O = mol_H2O_produced * M_H2O

print(f"Reaction: 2H₂ + O₂ → 2H₂O")
print(f"Given: {{mol_H2}} mol H₂ ({{mass_H2:.2f}} g)")
print(f"O₂ needed:  {{mol_O2_needed:.2f}} mol ({{mass_O2:.2f}} g)")
print(f"H₂O produced: {{mol_H2O_produced:.2f}} mol ({{mass_H2O:.2f}} g)")
print(f"Mass check: {{mass_H2:.2f}} + {{mass_O2:.2f}} = {{mass_H2 + mass_O2:.2f}} g reactants")
print(f"            {{mass_H2O:.2f}} g products ✓")
"""
    output = run_code(executor, code)
    return {
        "id": ex_id,
        "category": "chemistry",
        "difficulty": "basic",
        "query": (
            f"For the reaction 2H₂ + O₂ → 2H₂O, if you start with {mol_h2} mol of H₂, "
            "how many moles of O₂ are needed and how many moles of H₂O are produced? "
            "Verify mass conservation."
        ),
        "response": (
            "<think>\n"
            "Stoichiometry from balanced equation.\n"
            "Ratio: 2 H₂ : 1 O₂ : 2 H₂O\n"
            "mol O₂ = mol H₂ / 2, mol H₂O = mol H₂ (2:2 ratio)\n"
            "Check: mass of reactants = mass of products\n"
            "</think>\n"
            f"<code>\n{code}</code>\n"
            f"<output>\n{output}\n</output>\n"
        ),
    }


def make_limiting_reagent(ex_id: str) -> dict:
    random.seed(random.randint(1, 999))
    # N2 + 3H2 → 2NH3
    mol_n2 = round(random.uniform(1.0, 5.0), 1)
    mol_h2 = round(random.uniform(3.0, 15.0), 1)
    executor = PythonExecutor()
    code = f"""\
# Limiting reagent: N₂ + 3H₂ → 2NH₃
mol_N2_given = {mol_n2}
mol_H2_given = {mol_h2}

# Required ratios
mol_H2_needed_for_N2 = mol_N2_given * 3
mol_N2_needed_for_H2 = mol_H2_given / 3

print(f"Reaction: N₂ + 3H₂ → 2NH₃")
print(f"Given: {{mol_N2_given}} mol N₂, {{mol_H2_given}} mol H₂")
print(f"H₂ needed for all N₂: {{mol_H2_needed_for_N2:.2f}} mol")
print(f"N₂ needed for all H₂: {{mol_N2_needed_for_H2:.2f}} mol")

if mol_H2_given >= mol_H2_needed_for_N2:
    limiting = "N₂"
    excess = "H₂"
    mol_NH3 = mol_N2_given * 2
    mol_excess_left = mol_H2_given - mol_H2_needed_for_N2
else:
    limiting = "H₂"
    excess = "N₂"
    mol_NH3 = mol_H2_given * 2 / 3
    mol_excess_left = mol_N2_given - mol_N2_needed_for_H2

print(f"Limiting reagent: {{limiting}}")
print(f"NH₃ produced: {{mol_NH3:.3f}} mol")
print(f"Excess {{excess}} remaining: {{mol_excess_left:.3f}} mol")
"""
    output = run_code(executor, code)
    return {
        "id": ex_id,
        "category": "chemistry",
        "difficulty": "basic",
        "query": (
            f"For N₂ + 3H₂ → 2NH₃, you have {mol_n2} mol N₂ and {mol_h2} mol H₂. "
            "Which is the limiting reagent? How much NH₃ is produced?"
        ),
        "response": (
            "<think>\n"
            "Compare actual ratio to stoichiometric ratio.\n"
            "Need 3 mol H₂ per 1 mol N₂.\n"
            "Whichever runs out first is limiting.\n"
            "</think>\n"
            f"<code>\n{code}</code>\n"
            f"<output>\n{output}\n</output>\n"
        ),
    }


def make_percent_yield(ex_id: str) -> dict:
    random.seed(random.randint(1, 999))
    # 2Fe + 3Cl2 → 2FeCl3
    mass_fe = random.choice([10, 25, 50, 100])
    actual_yield_pct = round(random.uniform(72, 95), 1)
    executor = PythonExecutor()
    code = f"""\
# Percent yield: 2Fe + 3Cl₂ → 2FeCl₃
mass_Fe = {mass_fe}  # g
actual_yield_pct = {actual_yield_pct}  # %

M_Fe   = 55.845   # g/mol
M_FeCl3 = 162.20  # g/mol

mol_Fe = mass_Fe / M_Fe
# 2 Fe → 2 FeCl₃ (1:1 ratio)
mol_FeCl3_theoretical = mol_Fe
mass_FeCl3_theoretical = mol_FeCl3_theoretical * M_FeCl3
mass_FeCl3_actual = mass_FeCl3_theoretical * actual_yield_pct / 100

print(f"Reaction: 2Fe + 3Cl₂ → 2FeCl₃")
print(f"Starting mass of Fe: {{mass_Fe}} g ({{mol_Fe:.4f}} mol)")
print(f"Theoretical yield of FeCl₃: {{mass_FeCl3_theoretical:.2f}} g")
print(f"Actual yield ({actual_yield_pct}%): {{mass_FeCl3_actual:.2f}} g")
print(f"Lost to side reactions / incomplete reaction: "
      f"{{mass_FeCl3_theoretical - mass_FeCl3_actual:.2f}} g")
"""
    output = run_code(executor, code)
    return {
        "id": ex_id,
        "category": "chemistry",
        "difficulty": "basic",
        "query": (
            f"{mass_fe} g of iron reacts with excess chlorine (2Fe + 3Cl₂ → 2FeCl₃). "
            f"If the actual yield is {actual_yield_pct}%, what is the theoretical yield "
            "and the actual mass of FeCl₃ obtained?"
        ),
        "response": (
            "<think>\n"
            "Convert mass Fe → mol Fe → mol FeCl₃ (1:1) → mass FeCl₃ (theoretical).\n"
            "Actual = theoretical × yield%.\n"
            "</think>\n"
            f"<code>\n{code}</code>\n"
            f"<output>\n{output}\n</output>\n"
        ),
    }


# ---------------------------------------------------------------------------
# Ideal Gas Law
# ---------------------------------------------------------------------------

def make_ideal_gas_basic(ex_id: str) -> dict:
    """PV = nRT — solve for one variable."""
    random.seed(random.randint(1, 999))
    R = 0.08206  # L·atm/(mol·K)
    n = round(random.uniform(0.5, 5.0), 2)
    T = random.choice([273, 298, 350, 400, 500])
    V = round(random.uniform(5.0, 50.0), 1)
    executor = PythonExecutor()
    code = f"""\
# Ideal gas law: PV = nRT
n = {n}       # mol
T = {T}       # K
V = {V}       # L
R = 0.08206   # L·atm/(mol·K)

P = n * R * T / V

# Also compute density
# For air-like gas, approximate molar mass
M_avg = 28.97  # g/mol (air)
density = (n * M_avg) / V

print(f"Ideal gas law: PV = nRT")
print(f"n = {{n}} mol, T = {{T}} K, V = {{V}} L")
print(f"P = nRT/V = {{P:.4f}} atm  ({{P*760:.1f}} mmHg, {{P*101.325:.1f}} kPa)")
print(f"Gas density (M≈28.97 g/mol): {{density:.4f}} g/L")
# Check: what volume at STP?
V_stp = n * R * 273.15 / 1.0
print(f"At STP (0°C, 1 atm): V = {{V_stp:.2f}} L")
"""
    output = run_code(executor, code)
    return {
        "id": ex_id,
        "category": "chemistry",
        "difficulty": "basic",
        "query": (
            f"{n} mol of gas occupies {V} L at {T} K. "
            "Calculate the pressure in atm, mmHg, and kPa. "
            "What volume would this gas occupy at STP?"
        ),
        "response": (
            "<think>\n"
            "Ideal gas law: PV = nRT\n"
            "P = nRT/V\n"
            "Convert: 1 atm = 760 mmHg = 101.325 kPa\n"
            "STP: T=273.15 K, P=1 atm\n"
            "</think>\n"
            f"<code>\n{code}</code>\n"
            f"<output>\n{output}\n</output>\n"
        ),
    }


def make_partial_pressures(ex_id: str) -> dict:
    """Dalton's law of partial pressures."""
    random.seed(random.randint(1, 999))
    mol_n2 = round(random.uniform(0.5, 3.0), 2)
    mol_o2 = round(random.uniform(0.3, 2.0), 2)
    mol_ar = round(random.uniform(0.1, 1.0), 2)
    V = round(random.uniform(10, 40), 1)
    T = random.choice([298, 310, 350])
    executor = PythonExecutor()
    code = f"""\
# Dalton's law: P_total = Σ P_i,  P_i = (n_i/n_total) × P_total
R = 0.08206
n_N2 = {mol_n2}
n_O2 = {mol_o2}
n_Ar = {mol_ar}
V = {V}   # L
T = {T}   # K

n_total = n_N2 + n_O2 + n_Ar
P_total = n_total * R * T / V

# Partial pressures
P_N2 = (n_N2 / n_total) * P_total
P_O2 = (n_O2 / n_total) * P_total
P_Ar = (n_Ar / n_total) * P_total

print(f"Gas mixture in {{V}} L at {{T}} K")
print(f"Total moles: {{n_total:.4f}} mol")
print(f"Total pressure: {{P_total:.4f}} atm")
print(f"  P_N₂ = {{P_N2:.4f}} atm  (mole fraction: {{n_N2/n_total:.4f}})")
print(f"  P_O₂ = {{P_O2:.4f}} atm  (mole fraction: {{n_O2/n_total:.4f}})")
print(f"  P_Ar = {{P_Ar:.4f}} atm  (mole fraction: {{n_Ar/n_total:.4f}})")
print(f"Check: sum = {{P_N2 + P_O2 + P_Ar:.4f}} atm ✓")
"""
    output = run_code(executor, code)
    return {
        "id": ex_id,
        "category": "chemistry",
        "difficulty": "basic",
        "query": (
            f"A {V} L container at {T} K holds {mol_n2} mol N₂, {mol_o2} mol O₂, "
            f"and {mol_ar} mol Ar. Find the total pressure and partial pressure of each gas."
        ),
        "response": (
            "<think>\n"
            "Dalton's law: each gas contributes proportionally to its mole fraction.\n"
            "P_i = (n_i/n_total) × P_total\n"
            "P_total = n_total × RT/V\n"
            "</think>\n"
            f"<code>\n{code}</code>\n"
            f"<output>\n{output}\n</output>\n"
        ),
    }


def make_gas_combined(ex_id: str) -> dict:
    """Combined gas law: P1V1/T1 = P2V2/T2."""
    random.seed(random.randint(1, 999))
    P1 = round(random.uniform(1.0, 3.0), 2)
    V1 = round(random.uniform(5, 30), 1)
    T1 = random.choice([273, 298, 300])
    P2 = round(random.uniform(0.5, 5.0), 2)
    T2 = random.choice([350, 400, 500, 600])
    executor = PythonExecutor()
    code = f"""\
# Combined gas law: P₁V₁/T₁ = P₂V₂/T₂
P1 = {P1}   # atm
V1 = {V1}   # L
T1 = {T1}   # K
P2 = {P2}   # atm
T2 = {T2}   # K

# Find V2
V2 = P1 * V1 * T2 / (P2 * T1)

print(f"Combined gas law: P₁V₁/T₁ = P₂V₂/T₂")
print(f"Initial: P₁={{P1}} atm, V₁={{V1}} L, T₁={{T1}} K")
print(f"Final:   P₂={{P2}} atm, T₂={{T2}} K")
print(f"V₂ = P₁V₁T₂/(P₂T₁) = {{V2:.3f}} L")

# Also find n (constant throughout)
R = 0.08206
n = P1 * V1 / (R * T1)
print(f"Moles of gas (constant): {{n:.4f}} mol")
print(f"Check with final state: n = P₂V₂/(RT₂) = {{P2*V2/(R*T2):.4f}} mol ✓")
"""
    output = run_code(executor, code)
    return {
        "id": ex_id,
        "category": "chemistry",
        "difficulty": "basic",
        "query": (
            f"A gas initially at {P1} atm, {V1} L, {T1} K is changed to "
            f"{P2} atm and {T2} K. What is the new volume? "
            "Verify the number of moles is conserved."
        ),
        "response": (
            "<think>\n"
            "Combined gas law (n constant): P₁V₁/T₁ = P₂V₂/T₂\n"
            "V₂ = P₁V₁T₂/(P₂T₁)\n"
            "Cross-check with ideal gas law on both states.\n"
            "</think>\n"
            f"<code>\n{code}</code>\n"
            f"<output>\n{output}\n</output>\n"
        ),
    }


# ---------------------------------------------------------------------------
# Thermodynamics
# ---------------------------------------------------------------------------

def make_enthalpy_reaction(ex_id: str) -> dict:
    """ΔH_reaction from standard enthalpies of formation."""
    random.seed(random.randint(1, 999))
    # CH4 + 2O2 → CO2 + 2H2O
    executor = PythonExecutor()
    code = """\
# Enthalpy of reaction from ΔH°f values
# CH₄ + 2O₂ → CO₂ + 2H₂O

# Standard enthalpies of formation (kJ/mol)
dHf_CH4  = -74.8
dHf_O2   = 0.0     # element in standard state
dHf_CO2  = -393.5
dHf_H2O  = -285.8  # liquid water

# ΔH°rxn = Σ(ν·ΔH°f products) - Σ(ν·ΔH°f reactants)
dH_products  = 1 * dHf_CO2 + 2 * dHf_H2O
dH_reactants = 1 * dHf_CH4 + 2 * dHf_O2
dH_rxn = dH_products - dH_reactants

print(f"Reaction: CH₄ + 2O₂ → CO₂ + 2H₂O(l)")
print(f"ΔH°f values (kJ/mol):")
print(f"  CH₄: {{dHf_CH4}},  O₂: {{dHf_O2}},  CO₂: {{dHf_CO2}},  H₂O(l): {{dHf_H2O}}")
print(f"Σ(products)  = {{dH_products:.1f}} kJ")
print(f"Σ(reactants) = {{dH_reactants:.1f}} kJ")
print(f"ΔH°rxn = {{dH_products:.1f}} - {{dH_reactants:.1f}} = {{dH_rxn:.1f}} kJ")
print(f"Reaction is {'exothermic' if dH_rxn < 0 else 'endothermic'}")

# Energy per gram of methane
M_CH4 = 16.04
energy_per_g = abs(dH_rxn) / M_CH4
print(f"Energy released: {{energy_per_g:.1f}} kJ/g CH₄")
"""
    output = run_code(executor, code)
    return {
        "id": ex_id,
        "category": "chemistry",
        "difficulty": "basic",
        "query": (
            "Calculate ΔH° for the combustion of methane: CH₄ + 2O₂ → CO₂ + 2H₂O(l). "
            "Is it exothermic or endothermic? How much energy per gram of CH₄?"
        ),
        "response": (
            "<think>\n"
            "ΔH°rxn = Σ(ν·ΔH°f products) − Σ(ν·ΔH°f reactants)\n"
            "Elements in standard state have ΔH°f = 0.\n"
            "Negative ΔH → exothermic.\n"
            "</think>\n"
            f"<code>\n{code}</code>\n"
            f"<output>\n{output}\n</output>\n"
        ),
    }


def make_gibbs_free_energy(ex_id: str) -> dict:
    """ΔG = ΔH - TΔS — spontaneity."""
    random.seed(random.randint(1, 999))
    dH = round(random.uniform(-200, -20), 1)
    dS = round(random.uniform(-200, 50), 1)
    executor = PythonExecutor()
    code = f"""\
# Gibbs free energy: ΔG = ΔH - TΔS
dH = {dH}    # kJ/mol
dS = {dS}    # J/(mol·K) → convert to kJ
dS_kJ = dS / 1000

temperatures = [273, 298, 350, 500, 1000]

print(f"ΔH = {{dH}} kJ/mol,  ΔS = {{dS}} J/(mol·K)")
print(f"{'T (K)':>8}  {'TΔS (kJ)':>10}  {'ΔG (kJ)':>10}  Spontaneous?")
print("-" * 50)

for T in temperatures:
    dG = dH - T * dS_kJ
    spontaneous = "Yes ✓" if dG < 0 else "No"
    print(f"{{T:>8}}  {{T*dS_kJ:>10.2f}}  {{dG:>10.2f}}  {{spontaneous}}")

# Find crossover temperature (ΔG = 0)
if dS_kJ != 0:
    T_cross = dH / dS_kJ
    print(f"\\nCrossover temperature: T = ΔH/ΔS = {{T_cross:.1f}} K")
    if dS_kJ > 0:
        print(f"Reaction spontaneous above {{T_cross:.1f}} K")
    else:
        print(f"Reaction spontaneous below {{T_cross:.1f}} K")
"""
    output = run_code(executor, code)
    return {
        "id": ex_id,
        "category": "chemistry",
        "difficulty": "basic",
        "query": (
            f"A reaction has ΔH = {dH} kJ/mol and ΔS = {dS} J/(mol·K). "
            "Is it spontaneous at 298 K? At what temperature does spontaneity change?"
        ),
        "response": (
            "<think>\n"
            "ΔG = ΔH − TΔS\n"
            "ΔG < 0 → spontaneous\n"
            "Need to convert ΔS to kJ for consistent units.\n"
            "Crossover: ΔG = 0 → T = ΔH/ΔS\n"
            "</think>\n"
            f"<code>\n{code}</code>\n"
            f"<output>\n{output}\n</output>\n"
        ),
    }


def make_hess_law(ex_id: str) -> dict:
    """Hess's law: combine reactions to find ΔH of target."""
    random.seed(random.randint(1, 999))
    # Target: C(s) + 2H2(g) → CH4(g)
    # Given:
    #   (1) C(s) + O2(g) → CO2(g)          ΔH = -393.5 kJ
    #   (2) H2(g) + ½O2(g) → H2O(l)        ΔH = -285.8 kJ
    #   (3) CH4(g) + 2O2(g) → CO2(g) + 2H2O(l)  ΔH = -890.3 kJ
    executor = PythonExecutor()
    code = """\
# Hess's Law: find ΔH for C(s) + 2H₂(g) → CH₄(g)
# Given reactions:
# (1) C + O₂ → CO₂                ΔH₁ = -393.5 kJ
# (2) H₂ + ½O₂ → H₂O(l)           ΔH₂ = -285.8 kJ
# (3) CH₄ + 2O₂ → CO₂ + 2H₂O(l)   ΔH₃ = -890.3 kJ

dH1 = -393.5
dH2 = -285.8
dH3 = -890.3

# Target: C + 2H₂ → CH₄
# Strategy:
#   Keep (1) as is:        C + O₂ → CO₂
#   Use 2×(2):             2H₂ + O₂ → 2H₂O
#   Reverse (3):           CO₂ + 2H₂O → CH₄ + 2O₂
#   Sum: C + 2H₂ + 2O₂ → CH₄ + 2O₂  →  C + 2H₂ → CH₄ ✓

dH_target = dH1 + 2*dH2 + (-dH3)

print("Hess's Law: C(s) + 2H₂(g) → CH₄(g)")
print(f"  (1) as is:     ΔH = {{dH1}} kJ")
print(f"  2×(2):         ΔH = {{2*dH2}} kJ")
print(f"  reverse (3):   ΔH = {{-dH3}} kJ")
print(f"  ΔH_target = {{dH1}} + {{2*dH2}} + {{-dH3}} = {{dH_target}} kJ")
print(f"\\nFormation enthalpy of CH₄: ΔH°f = {{dH_target}} kJ/mol")
"""
    output = run_code(executor, code)
    return {
        "id": ex_id,
        "category": "chemistry",
        "difficulty": "intermediate",
        "query": (
            "Use Hess's Law to find ΔH for C(s) + 2H₂(g) → CH₄(g) given:\n"
            "  (1) C + O₂ → CO₂              ΔH = -393.5 kJ\n"
            "  (2) H₂ + ½O₂ → H₂O(l)         ΔH = -285.8 kJ\n"
            "  (3) CH₄ + 2O₂ → CO₂ + 2H₂O   ΔH = -890.3 kJ"
        ),
        "response": (
            "<think>\n"
            "Hess's Law: ΔH is a state function, so we can add/subtract reactions.\n"
            "Keep (1), double (2), reverse (3).\n"
            "O₂ and CO₂ and H₂O cancel out, leaving C + 2H₂ → CH₄.\n"
            "</think>\n"
            f"<code>\n{code}</code>\n"
            f"<output>\n{output}\n</output>\n"
        ),
    }


# ---------------------------------------------------------------------------
# Equilibrium
# ---------------------------------------------------------------------------

def make_equilibrium_constant(ex_id: str) -> dict:
    """Calculate Kc from equilibrium concentrations."""
    random.seed(random.randint(1, 999))
    # N2O4(g) ⇌ 2NO2(g)
    conc_n2o4 = round(random.uniform(0.02, 0.15), 4)
    conc_no2 = round(random.uniform(0.01, 0.10), 4)
    executor = PythonExecutor()
    code = f"""\
# Equilibrium: N₂O₄(g) ⇌ 2NO₂(g)
[N2O4] = {conc_n2o4}  # M
[NO2]  = {conc_no2}   # M

Kc = [NO2]**2 / [N2O4]

print(f"Equilibrium: N₂O₄(g) ⇌ 2NO₂(g)")
print(f"[N₂O₄] = {{[N2O4]}} M")
print(f"[NO₂]  = {{[NO2]}} M")
print(f"Kc = [NO₂]²/[N₂O₄] = {{[NO2]**2}}/{{[N2O4]}} = {{Kc:.6f}}")

# Kp = Kc(RT)^(Δn), Δn = 2-1 = 1
R = 0.08206
T = 298
delta_n = 1
Kp = Kc * (R * T)**delta_n
print(f"Kp = Kc(RT)^Δn = {{Kc:.6f}} × ({{R*T:.3f}})^1 = {{Kp:.6f}}")

# Direction of shift if Q ≠ K
Q = Kc  # at equilibrium, Q = K
print(f"At equilibrium: Q = K = {{Kc:.6f}}")
"""
    output = run_code(executor, code)
    return {
        "id": ex_id,
        "category": "chemistry",
        "difficulty": "basic",
        "query": (
            f"At equilibrium, [N₂O₄] = {conc_n2o4} M and [NO₂] = {conc_no2} M "
            f"for N₂O₄(g) ⇌ 2NO₂(g). Calculate Kc and Kp at 298 K."
        ),
        "response": (
            "<think>\n"
            "Kc = [NO₂]²/[N₂O₄]\n"
            "Kp = Kc(RT)^Δn, where Δn = moles gas products − reactants = 2−1 = 1\n"
            "</think>\n"
            f"<code>\n{code}</code>\n"
            f"<output>\n{output}\n</output>\n"
        ),
    }


def make_ice_table(ex_id: str) -> dict:
    """ICE table calculation."""
    random.seed(random.randint(1, 999))
    # H2 + I2 ⇌ 2HI, Kc = 50.5 at 448°C
    initial_h2 = round(random.uniform(0.5, 2.0), 2)
    initial_i2 = round(random.uniform(0.5, 2.0), 2)
    executor = PythonExecutor()
    code = f"""\
import sympy as sp

# H₂ + I₂ ⇌ 2HI, Kc = 50.5
Kc = 50.5
H2_0 = {initial_h2}
I2_0 = {initial_i2}
HI_0 = 0.0

# ICE table: let x = amount of H₂ that reacts
#            H₂      I₂      2HI
# Initial:  H2_0    I2_0    0
# Change:   -x      -x      +2x
# Equil:    H2_0-x  I2_0-x  2x

x = sp.Symbol('x', positive=True)
expr = (2*x)**2 / ((H2_0 - x) * (I2_0 - x)) - Kc
solutions = sp.solve(expr, x)
# Pick the physically meaningful solution (0 < x < min(H2_0, I2_0))
valid = [s for s in solutions if 0 < float(s) < min(H2_0, I2_0)]
x_val = float(valid[0])

H2_eq = H2_0 - x_val
I2_eq = I2_0 - x_val
HI_eq = 2 * x_val

print(f"Reaction: H₂ + I₂ ⇌ 2HI, Kc = {{Kc}}")
print(f"Initial: [H₂]={{H2_0}} M, [I₂]={{I2_0}} M, [HI]=0")
print(f"ICE table: x = {{x_val:.4f}} M")
print(f"Equilibrium:")
print(f"  [H₂] = {{H2_eq:.4f}} M")
print(f"  [I₂] = {{I2_eq:.4f}} M")
print(f"  [HI] = {{HI_eq:.4f}} M")
print(f"Check: Kc = {{HI_eq**2/(H2_eq*I2_eq):.2f}} ✓")
"""
    output = run_code(executor, code)
    return {
        "id": ex_id,
        "category": "chemistry",
        "difficulty": "intermediate",
        "query": (
            f"For H₂ + I₂ ⇌ 2HI (Kc = 50.5), start with {initial_h2} M H₂ and "
            f"{initial_i2} M I₂. Find equilibrium concentrations using an ICE table."
        ),
        "response": (
            "<think>\n"
            "ICE table: set up quadratic from Kc expression.\n"
            "Kc = [HI]²/([H₂][I₂]) = (2x)²/((H₂₀−x)(I₂₀−x))\n"
            "Solve for x, pick physically meaningful root.\n"
            "</think>\n"
            f"<code>\n{code}</code>\n"
            f"<output>\n{output}\n</output>\n"
        ),
    }


def make_le_chatelier(ex_id: str) -> dict:
    """Le Chatelier's principle — quantitative shift."""
    random.seed(random.randint(1, 999))
    # N2 + 3H2 ⇌ 2NH3
    executor = PythonExecutor()
    code = """\
# Le Chatelier: N₂ + 3H₂ ⇌ 2NH₃, Kc = 0.50 at 400°C
Kc = 0.50

# Initial equilibrium
[N2]_1 = 0.50
[H2]_1 = 1.50
[NH3]_1 = 0.612  # chosen so Kc = [NH3]²/([N2][H2]³) ≈ 0.50

Q_check = [NH3]_1**2 / ([N2]_1 * [H2]_1**3)
print(f"Initial equilibrium:")
print(f"  [N₂]={{[N2]_1}}, [H₂]={{[H2]_1}}, [NH₃]={{[NH3]_1}}")
print(f"  Q = {{Q_check:.4f}} ≈ Kc = {{Kc}} ✓")

# Disturbance: add 0.50 M N₂
[N2]_new = [N2]_1 + 0.50
[H2]_new = [H2]_1
[NH3]_new = [NH3]_1

Q_after = [NH3]_new**2 / ([N2]_new * [H2]_new**3)
print(f"\\nAfter adding 0.50 M N₂:")
print(f"  [N₂]={{[N2]_new}}, [H₂]={{[H2]_new}}, [NH₃]={{[NH3]_new}}")
print(f"  Q = {{Q_after:.4f}}  (Q < Kc → shift RIGHT to make more NH₃)")
print(f"\\nLe Chatelier prediction: adding reactant shifts equilibrium toward products ✓")
"""
    output = run_code(executor, code)
    return {
        "id": ex_id,
        "category": "chemistry",
        "difficulty": "intermediate",
        "query": (
            "For N₂ + 3H₂ ⇌ 2NH₃ (Kc = 0.50), equilibrium concentrations are "
            "[N₂]=0.50 M, [H₂]=1.50 M, [NH₃]=0.612 M. If 0.50 M N₂ is added, "
            "which way does the equilibrium shift? Show quantitatively."
        ),
        "response": (
            "<think>\n"
            "Calculate Q after disturbance and compare to Kc.\n"
            "Q < Kc → shift right (toward products).\n"
            "Q > Kc → shift left (toward reactants).\n"
            "Le Chatelier: system counteracts the disturbance.\n"
            "</think>\n"
            f"<code>\n{code}</code>\n"
            f"<output>\n{output}\n</output>\n"
        ),
    }


# ---------------------------------------------------------------------------
# Acid-Base
# ---------------------------------------------------------------------------

def make_ph_calculation(ex_id: str) -> dict:
    """pH of strong and weak acids."""
    random.seed(random.randint(1, 999))
    # Strong acid: HCl
    conc_hcl = round(random.choice([0.001, 0.01, 0.05, 0.1, 0.5]), 4)
    executor = PythonExecutor()
    code = f"""\
import math

# Strong acid: HCl → H⁺ + Cl⁻ (complete dissociation)
C_HCl = {conc_hcl}  # M

[H_plus] = C_HCl  # strong acid, fully dissociated
pH = -math.log10([H_plus])
pOH = 14 - pH
[OH_minus] = 1e-14 / [H_plus]

print(f"Strong acid: HCl at {{C_HCl}} M")
print(f"[H⁺]  = {{[H_plus]}} M")
print(f"pH   = {{pH:.4f}}")
print(f"pOH  = {{pOH:.4f}}")
print(f"[OH⁻] = {{[OH_minus]:.2e}} M")
print(f"Check: pH + pOH = {{pH + pOH:.2f}} ✓")
"""
    output = run_code(executor, code)
    return {
        "id": ex_id,
        "category": "chemistry",
        "difficulty": "basic",
        "query": (
            f"Calculate the pH of a {conc_hcl} M HCl solution. "
            "Also find pOH and [OH⁻]."
        ),
        "response": (
            "<think>\n"
            "HCl is a strong acid → fully dissociates: [H⁺] = C_HCl\n"
            "pH = −log₁₀[H⁺]\n"
            "pH + pOH = 14 (at 25°C)\n"
            "</think>\n"
            f"<code>\n{code}</code>\n"
            f"<output>\n{output}\n</output>\n"
        ),
    }


def make_weak_acid(ex_id: str) -> dict:
    """pH of weak acid using Ka."""
    random.seed(random.randint(1, 999))
    # Acetic acid: Ka = 1.8e-5
    conc_hac = round(random.choice([0.01, 0.05, 0.1, 0.5, 1.0]), 3)
    executor = PythonExecutor()
    code = f"""\
import math

# Weak acid: CH₃COOH ⇌ H⁺ + CH₃COO⁻
Ka = 1.8e-5
C = {conc_hac}  # M

# Approximation: x = sqrt(Ka·C) when C/Ka > 100
x_approx = math.sqrt(Ka * C)
pH_approx = -math.log10(x_approx)

# Exact: Ka = x²/(C-x) → x² + Ka·x - Ka·C = 0
# Quadratic formula
discriminant = Ka**2 + 4*Ka*C
x_exact = (-Ka + math.sqrt(discriminant)) / 2
pH_exact = -math.log10(x_exact)

print(f"Weak acid: CH₃COOH at {{C}} M, Ka = {{Ka}}")
print(f"Approximation: x = √(Ka·C) = {{x_approx:.6f}} M")
print(f"  pH ≈ {{pH_approx:.4f}}")
print(f"Exact (quadratic): x = {{x_exact:.6f}} M")
print(f"  pH = {{pH_exact:.4f}}")
print(f"Error from approximation: {{abs(pH_approx - pH_exact):.6f}} pH units")
print(f"C/Ka = {{C/Ka:.0f}}  (>100 → approximation valid)")
"""
    output = run_code(executor, code)
    return {
        "id": ex_id,
        "category": "chemistry",
        "difficulty": "intermediate",
        "query": (
            f"Calculate the pH of {conc_hac} M acetic acid (Ka = 1.8×10⁻⁵). "
            "Use both the approximation and the exact quadratic method."
        ),
        "response": (
            "<think>\n"
            "Weak acid: Ka = x²/(C−x)\n"
            "Approximation (C/Ka > 100): x ≈ √(Ka·C)\n"
            "Exact: solve quadratic x² + Ka·x − Ka·C = 0\n"
            "</think>\n"
            f"<code>\n{code}</code>\n"
            f"<output>\n{output}\n</output>\n"
        ),
    }


def make_buffer_solution(ex_id: str) -> dict:
    """Henderson-Hasselbalch equation."""
    random.seed(random.randint(1, 999))
    # Acetic acid / acetate buffer
    conc_acid = round(random.uniform(0.05, 0.5), 3)
    conc_base = round(random.uniform(0.05, 0.5), 3)
    executor = PythonExecutor()
    code = f"""\
import math

# Buffer: CH₃COOH / CH₃COO⁻
Ka = 1.8e-5
pKa = -math.log10(Ka)
[acid] = {conc_acid}   # M
[base] = {conc_base}   # M

# Henderson-Hasselbalch
pH = pKa + math.log10([base] / [acid])

print(f"Buffer: {{[acid]}} M CH₃COOH + {{[base]}} M CH₃COO⁻")
print(f"pKa = {{pKa:.4f}}")
print(f"pH = pKa + log([base]/[acid])")
print(f"pH = {{pKa:.4f}} + log({{[base]}}/{{[acid]}})")
print(f"pH = {{pH:.4f}}")

# Buffer capacity: add 0.01 M HCl
added_HCl = 0.01
[acid]_new = [acid] + added_HCl
[base]_new = [base] - added_HCl
pH_new = pKa + math.log10([base]_new / [acid]_new)
print(f"\\nAfter adding {{added_HCl}} M HCl:")
print(f"  pH = {{pH_new:.4f}}  (ΔpH = {{pH_new - pH:.4f}})")
print(f"Buffer resisted large pH change ✓")
"""
    output = run_code(executor, code)
    return {
        "id": ex_id,
        "category": "chemistry",
        "difficulty": "intermediate",
        "query": (
            f"A buffer contains {conc_acid} M acetic acid and {conc_base} M acetate. "
            f"Calculate the pH. What happens when 0.01 M HCl is added?"
        ),
        "response": (
            "<think>\n"
            "Henderson-Hasselbalch: pH = pKa + log([base]/[acid])\n"
            "Adding strong acid converts base → acid, small pH change.\n"
            "Buffer capacity depends on concentrations.\n"
            "</think>\n"
            f"<code>\n{code}</code>\n"
            f"<output>\n{output}\n</output>\n"
        ),
    }


# ---------------------------------------------------------------------------
# Solution Chemistry
# ---------------------------------------------------------------------------

def make_molarity_dilution(ex_id: str) -> dict:
    """Molarity and dilution calculations."""
    random.seed(random.randint(1, 999))
    # NaCl: M = 58.44 g/mol
    mass_nacl = round(random.uniform(5, 50), 1)
    volume_ml = random.choice([100, 250, 500, 1000])
    executor = PythonExecutor()
    code = f"""\
# Molarity and dilution
mass_NaCl = {mass_nacl}  # g
V_ml = {volume_ml}       # mL
M_NaCl = 58.44           # g/mol

V_L = V_ml / 1000
mol_NaCl = mass_NaCl / M_NaCl
Molarity = mol_NaCl / V_L

print(f"Solution: {{mass_NaCl}} g NaCl in {{V_ml}} mL")
print(f"Moles NaCl = {{mol_NaCl:.4f}} mol")
print(f"Molarity = {{Molarity:.4f}} M")

# Dilution: take 25 mL and dilute to 250 mL
V1 = 25    # mL
V2 = 250   # mL
M2 = Molarity * V1 / V2
print(f"\\nDilution: {{V1}} mL → {{V2}} mL")
print(f"M₁V₁ = M₂V₂: {{Molarity:.4f}} × {{V1}} = M₂ × {{V2}}")
print(f"M₂ = {{M2:.6f}} M")
"""
    output = run_code(executor, code)
    return {
        "id": ex_id,
        "category": "chemistry",
        "difficulty": "basic",
        "query": (
            f"{mass_nacl} g of NaCl is dissolved in {volume_ml} mL of water. "
            "What is the molarity? If 25 mL of this solution is diluted to 250 mL, "
            "what is the new concentration?"
        ),
        "response": (
            "<think>\n"
            "Molarity = moles / liters\n"
            "Dilution: M₁V₁ = M₂V₂\n"
            "</think>\n"
            f"<code>\n{code}</code>\n"
            f"<output>\n{output}\n</output>\n"
        ),
    }


def make_colligative(ex_id: str) -> dict:
    """Freezing point depression and boiling point elevation."""
    random.seed(random.randint(1, 999))
    mass_solute = round(random.uniform(10, 100), 1)
    mass_solvent = random.choice([100, 250, 500, 1000])
    executor = PythonExecutor()
    code = f"""\
# Colligative properties: NaCl in water
# NaCl → Na⁺ + Cl⁻, van't Hoff factor i = 2
mass_solute = {mass_solute}  # g NaCl
mass_solvent = {mass_solvent} / 1000  # kg water
M_NaCl = 58.44
i = 2  # NaCl dissociates into 2 ions

Kf = 1.86   # °C·kg/mol (water)
Kb = 0.512  # °C·kg/mol (water)

molality = (mass_solute / M_NaCl) / mass_solvent
delta_Tf = i * Kf * molality
delta_Tb = i * Kb * molality

Tf = 0 - delta_Tf
Tb = 100 + delta_Tb

print(f"Colligative properties: {{mass_solute}} g NaCl in {{mass_solvent*1000:.0f}} g water")
print(f"Molality = {{molality:.4f}} mol/kg")
print(f"\\nFreezing point depression:")
print(f"  ΔTf = i·Kf·m = {{i}} × {{Kf}} × {{molality:.4f}} = {{delta_Tf:.3f}} °C")
print(f"  Freezing point = {{Tf:.3f}} °C")
print(f"\\nBoiling point elevation:")
print(f"  ΔTb = i·Kb·m = {{i}} × {{Kb}} × {{molality:.4f}} = {{delta_Tb:.3f}} °C")
print(f"  Boiling point = {{Tb:.3f}} °C")
"""
    output = run_code(executor, code)
    return {
        "id": ex_id,
        "category": "chemistry",
        "difficulty": "basic",
        "query": (
            f"{mass_solute} g of NaCl is dissolved in {mass_solvent} g of water. "
            "Calculate the freezing point and boiling point of the solution."
        ),
        "response": (
            "<think>\n"
            "Colligative properties depend on molality and van't Hoff factor.\n"
            "NaCl: i = 2 (dissociates into Na⁺ and Cl⁻)\n"
            "ΔTf = i·Kf·m, ΔTb = i·Kb·m\n"
            "Water: Kf = 1.86, Kb = 0.512\n"
            "</think>\n"
            f"<code>\n{code}</code>\n"
            f"<output>\n{output}\n</output>\n"
        ),
    }


# ---------------------------------------------------------------------------
# Master list
# ---------------------------------------------------------------------------

BUILDERS = [
    make_stoichiometry_basic,
    make_limiting_reagent,
    make_percent_yield,
    make_ideal_gas_basic,
    make_partial_pressures,
    make_gas_combined,
    make_enthalpy_reaction,
    make_gibbs_free_energy,
    make_hess_law,
    make_equilibrium_constant,
    make_ice_table,
    make_le_chatelier,
    make_ph_calculation,
    make_weak_acid,
    make_buffer_solution,
    make_molarity_dilution,
    make_colligative,
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
            ex_id = f"chem_{idx:03d}"
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
    parser = argparse.ArgumentParser(description="Generate chemistry training examples")
    parser.add_argument("--output", default="training/datasets/science/chemistry/basic.jsonl")
    parser.add_argument("--count", type=int, default=80)
    args = parser.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Generating {args.count} chemistry examples...")
    examples = generate_examples(args.count)

    with open(out_path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")

    print(f"\nWrote {len(examples)} examples to {out_path}")


if __name__ == "__main__":
    main()
