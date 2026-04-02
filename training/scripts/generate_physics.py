#!/usr/bin/env python3
"""
Generate physics training examples.

Covers:
- Kinematics: SUVAT equations, free fall, projectile motion
- Forces: Newton's laws, friction, inclined planes
- Energy: work, kinetic/potential, conservation
- Circular motion and gravity
- Waves and oscillations
- Electric circuits: Ohm's law, series/parallel
- Multi-step: combine kinematics + energy + forces

All outputs pre-executed and verified.

Usage:
    python training/scripts/generate_physics.py
    python training/scripts/generate_physics.py --output training/datasets/science/physics/basic.jsonl
    python training/scripts/generate_physics.py --count 120
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
# Kinematics
# ---------------------------------------------------------------------------

def make_suvat(ex_id: str) -> dict:
    """SUVAT: given 3 of (s, u, v, a, t), find the other 2."""
    seed = random.randint(1, 999)
    random.seed(seed)
    # v = u + at scenario
    u = random.choice([0, 5, 10, 15, 20])
    a = round(random.uniform(1.5, 9.8), 1)
    t = random.choice([2, 3, 4, 5, 6])
    executor = PythonExecutor()
    code = f"""\
# SUVAT kinematics: u={u} m/s, a={a} m/s², t={t} s
u = {u}   # initial velocity (m/s)
a = {a}   # acceleration (m/s²)
t = {t}   # time (s)

v = u + a * t               # final velocity
s = u*t + 0.5*a*t**2       # displacement
v_check = (v**2 - u**2) / (2*a) if a != 0 else 0  # s from v²=u²+2as

print(f"Given: u={{u}} m/s, a={{a}} m/s², t={{t}} s")
print(f"Final velocity:  v = u + at = {{v:.2f}} m/s")
print(f"Displacement:    s = ut + ½at² = {{s:.2f}} m")
print(f"Check (v²=u²+2as): s = {{v_check:.2f}} m ✓")
"""
    output = run_code(executor, code)
    return {
        "id": ex_id,
        "category": "physics",
        "difficulty": "basic",
        "query": (
            f"An object starts at {u} m/s and accelerates at {a} m/s² for {t} seconds. "
            "Find the final velocity and displacement."
        ),
        "response": (
            "<think>\n"
            "SUVAT equations. Known: u, a, t. Find: v and s.\n"
            "v = u + at\n"
            "s = ut + ½at²\n"
            "Cross-check with v² = u² + 2as.\n"
            "</think>\n"
            f"<code>\n{code}</code>\n"
            f"<output>\n{output}\n</output>\n"
            f"After {t} s the object has velocity **{u + a*t:.2f} m/s** and has "
            f"travelled **{u*t + 0.5*a*t**2:.2f} m**."
        ),
    }


def make_free_fall(ex_id: str) -> dict:
    random.seed(random.randint(1, 999))
    h = random.choice([10, 20, 45, 80, 100])
    executor = PythonExecutor()
    code = f"""\
import math
g = 9.81  # m/s²
h = {h}   # drop height (m)

# Time to fall: h = ½gt² → t = sqrt(2h/g)
t = math.sqrt(2 * h / g)

# Velocity on impact: v² = 2gh
v = math.sqrt(2 * g * h)

print(f"Free fall from h = {{h}} m")
print(f"Time to fall:      t = √(2h/g) = {{t:.3f}} s")
print(f"Impact velocity:   v = √(2gh) = {{v:.3f}} m/s  (≈ {{v*3.6:.1f}} km/h)")
print(f"Check: v = g*t = {{g*t:.3f}} m/s ✓")
"""
    output = run_code(executor, code)
    return {
        "id": ex_id,
        "category": "physics",
        "difficulty": "basic",
        "query": (
            f"An object is dropped from rest at height {h} m. "
            "How long does it take to reach the ground, and what is its impact velocity? "
            "(g = 9.81 m/s², ignore air resistance)"
        ),
        "response": (
            "<think>\n"
            "Free fall from rest: u=0, a=g=9.81 m/s².\n"
            "h = ½gt² → t = √(2h/g)\n"
            "v = gt = √(2gh)\n"
            "</think>\n"
            f"<code>\n{code}</code>\n"
            f"<output>\n{output}\n</output>\n"
            f"Dropped from {h} m the object takes **{math.sqrt(2*h/9.81):.3f} s** to land "
            f"and hits at **{math.sqrt(2*9.81*h):.3f} m/s**."
        ),
    }


def make_projectile(ex_id: str) -> dict:
    random.seed(random.randint(1, 999))
    v0 = random.choice([15, 20, 25, 30])
    theta = random.choice([30, 45, 60])
    executor = PythonExecutor()
    code = f"""\
import math
v0    = {v0}     # m/s
theta = {theta}  # degrees
g     = 9.81    # m/s²

rad = math.radians(theta)
vx  = v0 * math.cos(rad)
vy  = v0 * math.sin(rad)

t_flight = 2 * vy / g
range_m  = vx * t_flight
h_max    = vy**2 / (2*g)

print(f"Projectile: v₀={{v0}} m/s at {{theta}}°")
print(f"Components: vx={{vx:.3f}} m/s, vy={{vy:.3f}} m/s")
print(f"Time of flight: {{t_flight:.3f}} s")
print(f"Horizontal range: {{range_m:.3f}} m")
print(f"Maximum height: {{h_max:.3f}} m")
"""
    output = run_code(executor, code)
    rad = math.radians(theta)
    vx = v0 * math.cos(rad); vy = v0 * math.sin(rad)
    return {
        "id": ex_id,
        "category": "physics",
        "difficulty": "basic",
        "query": (
            f"A ball is launched at {v0} m/s at an angle of {theta}° above horizontal. "
            "Find the time of flight, horizontal range, and maximum height. (g = 9.81 m/s²)"
        ),
        "response": (
            "<think>\n"
            "Resolve velocity into x and y components.\n"
            "vx = v₀cos(θ) — constant throughout\n"
            "vy = v₀sin(θ) — decreases at rate g\n"
            "Time of flight: t = 2vy/g (symmetric trajectory)\n"
            "Range = vx × t, max height = vy²/(2g)\n"
            "</think>\n"
            f"<code>\n{code}</code>\n"
            f"<output>\n{output}\n</output>\n"
            f"Range: **{vx*2*vy/9.81:.2f} m**, max height: **{vy**2/(2*9.81):.2f} m**, "
            f"flight time: **{2*vy/9.81:.3f} s**."
        ),
    }


# ---------------------------------------------------------------------------
# Forces
# ---------------------------------------------------------------------------

def make_newtons_second(ex_id: str) -> dict:
    random.seed(random.randint(1, 999))
    mass = random.choice([2, 5, 10, 20, 50])
    force = random.choice([10, 20, 50, 100, 200])
    friction_coeff = round(random.uniform(0.1, 0.4), 2)
    executor = PythonExecutor()
    code = f"""\
g      = 9.81
mass   = {mass}           # kg
F_app  = {force}          # applied force (N)
mu     = {friction_coeff} # kinetic friction coefficient

F_friction = mu * mass * g
F_net      = F_app - F_friction
accel      = F_net / mass

print(f"Mass: {{mass}} kg, Applied force: {{F_app}} N, μ = {{mu}}")
print(f"Friction force:  F_f = μmg = {{F_friction:.2f}} N")
print(f"Net force:       F_net = {{F_app}} - {{F_friction:.2f}} = {{F_net:.2f}} N")
print(f"Acceleration:    a = F/m = {{accel:.3f}} m/s²")
if F_net <= 0:
    print("Object does not accelerate (friction exceeds applied force)")
"""
    output = run_code(executor, code)
    F_f = friction_coeff * mass * 9.81
    return {
        "id": ex_id,
        "category": "physics",
        "difficulty": "basic",
        "query": (
            f"A {mass} kg object on a surface (μ = {friction_coeff}) has a horizontal force "
            f"of {force} N applied to it. What is its acceleration?"
        ),
        "response": (
            "<think>\n"
            "Newton's second law: F_net = ma\n"
            "Net force = Applied − Friction\n"
            "Friction = μmg (kinetic)\n"
            "a = F_net / m\n"
            "</think>\n"
            f"<code>\n{code}</code>\n"
            f"<output>\n{output}\n</output>\n"
            f"Friction is {F_f:.2f} N, net force is {force - F_f:.2f} N, "
            f"acceleration is **{(force - F_f)/mass:.3f} m/s²**."
        ),
    }


def make_inclined_plane(ex_id: str) -> dict:
    random.seed(random.randint(1, 999))
    mass = random.choice([3, 5, 10, 15])
    theta = random.choice([20, 30, 45])
    mu = round(random.uniform(0.1, 0.35), 2)
    executor = PythonExecutor()
    code = f"""\
import math
mass  = {mass}   # kg
theta = {theta}  # incline angle (degrees)
mu    = {mu}     # friction coefficient
g     = 9.81

rad        = math.radians(theta)
F_gravity  = mass * g                          # total weight
F_parallel = F_gravity * math.sin(rad)         # component along slope
F_normal   = F_gravity * math.cos(rad)         # normal force
F_friction = mu * F_normal                     # friction (up the slope)
F_net      = F_parallel - F_friction
accel      = F_net / mass

print(f"Inclined plane: {{mass}} kg, θ={{theta}}°, μ={{mu}}")
print(f"Weight component along slope: {{F_parallel:.3f}} N")
print(f"Normal force:                 {{F_normal:.3f}} N")
print(f"Friction force (up slope):    {{F_friction:.3f}} N")
print(f"Net force (down slope):       {{F_net:.3f}} N")
print(f"Acceleration:                 {{accel:.3f}} m/s²")
if F_net <= 0:
    print("Object stays stationary (friction prevents sliding)")
"""
    output = run_code(executor, code)
    return {
        "id": ex_id,
        "category": "physics",
        "difficulty": "intermediate",
        "query": (
            f"A {mass} kg block is on an inclined plane at {theta}°. "
            f"The coefficient of kinetic friction is {mu}. "
            "What is the block's acceleration down the slope?"
        ),
        "response": (
            "<think>\n"
            "Resolve forces along and perpendicular to slope.\n"
            "F_parallel = mg sin(θ) — pulls block down the slope\n"
            "N = mg cos(θ) — normal force\n"
            "F_friction = μN — acts up the slope (opposing motion)\n"
            "F_net = F_parallel − F_friction, a = F_net/m\n"
            "</think>\n"
            f"<code>\n{code}</code>\n"
            f"<output>\n{output}\n</output>\n"
        ),
    }


# ---------------------------------------------------------------------------
# Energy
# ---------------------------------------------------------------------------

def make_energy_conservation(ex_id: str) -> dict:
    random.seed(random.randint(1, 999))
    mass = random.choice([1, 2, 5, 10])
    h = random.choice([5, 10, 20, 50])
    executor = PythonExecutor()
    code = f"""\
import math
mass = {mass}   # kg
h    = {h}      # initial height (m)
g    = 9.81

# Gravitational PE at top
PE = mass * g * h
# KE at bottom (PE all converted, no friction)
KE = PE
v  = math.sqrt(2 * KE / mass)   # = sqrt(2gh)

print(f"Mass: {{mass}} kg, Height: {{h}} m")
print(f"Potential energy at top: PE = mgh = {{PE:.2f}} J")
print(f"Kinetic energy at bottom: KE = {{KE:.2f}} J (all PE converted)")
print(f"Speed at bottom: v = √(2gh) = {{v:.3f}} m/s")
print(f"Check: ½mv² = {{0.5*mass*v**2:.2f}} J ✓")
"""
    output = run_code(executor, code)
    return {
        "id": ex_id,
        "category": "physics",
        "difficulty": "basic",
        "query": (
            f"A {mass} kg ball rolls from rest down a frictionless slope of height {h} m. "
            "What is its speed at the bottom? (Use conservation of energy.)"
        ),
        "response": (
            "<think>\n"
            "No friction → mechanical energy conserved.\n"
            "PE at top = KE at bottom\n"
            "mgh = ½mv² → v = √(2gh)\n"
            "</think>\n"
            f"<code>\n{code}</code>\n"
            f"<output>\n{output}\n</output>\n"
            f"All potential energy converts to kinetic energy. Speed at bottom: "
            f"**{math.sqrt(2*9.81*h):.3f} m/s**."
        ),
    }


def make_work_energy(ex_id: str) -> dict:
    random.seed(random.randint(1, 999))
    mass = random.choice([2, 5, 10])
    v0 = random.choice([0, 2, 5])
    F = random.choice([20, 40, 60, 100])
    d = random.choice([5, 10, 20])
    executor = PythonExecutor()
    code = f"""\
import math
mass = {mass}  # kg
v0   = {v0}    # initial speed (m/s)
F    = {F}     # net force (N)
d    = {d}     # displacement (m)

# Work-energy theorem: W = ΔKE
W   = F * d
KE0 = 0.5 * mass * v0**2
KE1 = KE0 + W
v1  = math.sqrt(2 * KE1 / mass)

print(f"Work-energy theorem")
print(f"Net force {{F}} N over {{d}} m: W = {{W}} J")
print(f"Initial KE = ½mv₀² = {{KE0:.2f}} J")
print(f"Final KE = {{KE0:.2f}} + {{W}} = {{KE1:.2f}} J")
print(f"Final speed: v = √(2·KE/m) = {{v1:.3f}} m/s")
"""
    output = run_code(executor, code)
    return {
        "id": ex_id,
        "category": "physics",
        "difficulty": "basic",
        "query": (
            f"A {mass} kg object moving at {v0} m/s has a net force of {F} N applied "
            f"over {d} m. Use the work-energy theorem to find its final speed."
        ),
        "response": (
            "<think>\n"
            "Work-energy theorem: W = ΔKE = KE_final - KE_initial\n"
            "W = Fd\n"
            "KE_final = KE_initial + W\n"
            "v = √(2·KE_final/m)\n"
            "</think>\n"
            f"<code>\n{code}</code>\n"
            f"<output>\n{output}\n</output>\n"
        ),
    }


# ---------------------------------------------------------------------------
# Circular motion & gravity
# ---------------------------------------------------------------------------

def make_circular_motion(ex_id: str) -> dict:
    random.seed(random.randint(1, 999))
    mass = random.choice([0.5, 1.0, 2.0])
    r = random.choice([0.5, 1.0, 2.0])
    v = round(random.uniform(2, 8), 1)
    executor = PythonExecutor()
    code = f"""\
import math
mass = {mass}  # kg
r    = {r}     # radius (m)
v    = {v}     # speed (m/s)

# Centripetal acceleration and force
a_c = v**2 / r
F_c = mass * a_c

# Angular velocity and period
omega  = v / r
T      = 2 * math.pi / omega
freq   = 1 / T

print(f"Circular motion: m={{mass}} kg, r={{r}} m, v={{v}} m/s")
print(f"Centripetal acceleration: a = v²/r = {{a_c:.3f}} m/s²")
print(f"Centripetal force:        F = ma  = {{F_c:.3f}} N")
print(f"Angular velocity:         ω = v/r = {{omega:.3f}} rad/s")
print(f"Period:                   T = 2π/ω = {{T:.3f}} s")
print(f"Frequency:                f = 1/T  = {{freq:.3f}} Hz")
"""
    output = run_code(executor, code)
    return {
        "id": ex_id,
        "category": "physics",
        "difficulty": "basic",
        "query": (
            f"A {mass} kg object moves in a circle of radius {r} m at {v} m/s. "
            "Find the centripetal acceleration, centripetal force, period, and frequency."
        ),
        "response": (
            "<think>\n"
            "Centripetal acceleration: a = v²/r\n"
            "Centripetal force: F = ma = mv²/r\n"
            "Angular velocity: ω = v/r\n"
            "Period: T = 2π/ω = 2πr/v\n"
            "</think>\n"
            f"<code>\n{code}</code>\n"
            f"<output>\n{output}\n</output>\n"
        ),
    }


# ---------------------------------------------------------------------------
# Waves and oscillations
# ---------------------------------------------------------------------------

def make_pendulum(ex_id: str) -> dict:
    random.seed(random.randint(1, 999))
    L = round(random.uniform(0.25, 2.5), 2)
    executor = PythonExecutor()
    code = f"""\
import math
L = {L}     # pendulum length (m)
g = 9.81   # m/s²

# Simple pendulum (small angle)
T     = 2 * math.pi * math.sqrt(L / g)
freq  = 1 / T
omega = 2 * math.pi / T

print(f"Simple pendulum: L = {{L}} m")
print(f"Period:   T = 2π√(L/g) = {{T:.4f}} s")
print(f"Frequency: f = 1/T     = {{freq:.4f}} Hz")
print(f"Angular freq: ω = 2π/T = {{omega:.4f}} rad/s")
# For a 1-second clock pendulum, L = g/4π²
L_1s = g / (4 * math.pi**2)
print(f"(1-second pendulum needs L = {{L_1s:.4f}} m)")
"""
    output = run_code(executor, code)
    return {
        "id": ex_id,
        "category": "physics",
        "difficulty": "basic",
        "query": (
            f"A simple pendulum has length {L} m. "
            "Find its period and frequency. What length would give a 1-second period?"
        ),
        "response": (
            "<think>\n"
            "Simple pendulum (small-angle): T = 2π√(L/g)\n"
            "Period depends only on length and g, not mass.\n"
            "For T=1s: L = g/(4π²)\n"
            "</think>\n"
            f"<code>\n{code}</code>\n"
            f"<output>\n{output}\n</output>\n"
        ),
    }


def make_wave_properties(ex_id: str) -> dict:
    random.seed(random.randint(1, 999))
    freq = random.choice([440, 1000, 2000, 5000])
    v = random.choice([340, 343, 1500])  # air or water
    medium = "water" if v > 500 else "air"
    executor = PythonExecutor()
    code = f"""\
freq = {freq}  # Hz
v    = {v}     # wave speed (m/s) — {medium}

wavelength = v / freq
period     = 1 / freq
omega      = 2 * 3.14159265 * freq
k          = 2 * 3.14159265 / wavelength  # wave number (rad/m)

print(f"Wave in {medium}: f={{freq}} Hz, v={{v}} m/s")
print(f"Wavelength:    λ = v/f  = {{wavelength:.6f}} m  ({{wavelength*100:.4f}} cm)")
print(f"Period:        T = 1/f  = {{period:.6f}} s")
print(f"Angular freq:  ω = 2πf  = {{omega:.3f}} rad/s")
print(f"Wave number:   k = 2π/λ = {{k:.3f}} rad/m")
print(f"Check: v = fλ = {{freq * wavelength:.1f}} m/s ✓")
"""
    output = run_code(executor, code)
    return {
        "id": ex_id,
        "category": "physics",
        "difficulty": "basic",
        "query": (
            f"A {freq} Hz wave travels through {medium} at {v} m/s. "
            "Find the wavelength, period, angular frequency, and wave number."
        ),
        "response": (
            "<think>\n"
            "Wave relationships:\n"
            "λ = v/f\n"
            "T = 1/f\n"
            "ω = 2πf\n"
            "k = 2π/λ = ω/v\n"
            "</think>\n"
            f"<code>\n{code}</code>\n"
            f"<output>\n{output}\n</output>\n"
        ),
    }


# ---------------------------------------------------------------------------
# Electric circuits
# ---------------------------------------------------------------------------

def make_ohms_law(ex_id: str) -> dict:
    random.seed(random.randint(1, 999))
    V = random.choice([5, 9, 12, 24])
    R = random.choice([10, 22, 47, 100, 220])
    executor = PythonExecutor()
    code = f"""\
V = {V}   # volts
R = {R}   # ohms

I = V / R                   # current (Amperes)
P = V * I                   # power (Watts)
P_check = I**2 * R          # check with P = I²R
P_check2 = V**2 / R         # check with P = V²/R

print(f"Ohm's law: V={{V}}V, R={{R}}Ω")
print(f"Current:  I = V/R = {{I*1000:.3f}} mA  ({{I:.6f}} A)")
print(f"Power:    P = VI  = {{P*1000:.3f}} mW")
print(f"Check P=I²R: {{P_check*1000:.3f}} mW ✓")
print(f"Check P=V²/R: {{P_check2*1000:.3f}} mW ✓")
"""
    output = run_code(executor, code)
    return {
        "id": ex_id,
        "category": "physics",
        "difficulty": "basic",
        "query": (
            f"A {V}V source drives current through a {R}Ω resistor. "
            "Find the current and power dissipated. Verify with all three power formulas."
        ),
        "response": (
            "<think>\n"
            "Ohm's law: V = IR → I = V/R\n"
            "Power: P = VI = I²R = V²/R\n"
            "All three should agree.\n"
            "</think>\n"
            f"<code>\n{code}</code>\n"
            f"<output>\n{output}\n</output>\n"
        ),
    }


def make_series_parallel_circuit(ex_id: str) -> dict:
    random.seed(random.randint(1, 999))
    V = random.choice([9, 12, 24])
    R1 = random.choice([100, 220, 330])
    R2 = random.choice([100, 220, 470])
    R3 = random.choice([100, 470, 1000])
    executor = PythonExecutor()
    code = f"""\
V  = {V}    # supply voltage
R1 = {R1}   # Ω
R2 = {R2}   # Ω
R3 = {R3}   # Ω

# Series: R1 in series with (R2 parallel R3)
R_parallel = (R2 * R3) / (R2 + R3)
R_total    = R1 + R_parallel

I_total = V / R_total
V_R1    = I_total * R1
V_para  = V - V_R1   # voltage across parallel combination

I2 = V_para / R2
I3 = V_para / R3

P1 = I_total**2 * R1
P2 = I2**2 * R2
P3 = I3**2 * R3
P_total = P1 + P2 + P3

print(f"Circuit: R1={{R1}}Ω in series with R2={{R2}}Ω ∥ R3={{R3}}Ω, V={{V}}V")
print(f"R2 ∥ R3 = {{R_parallel:.2f}} Ω")
print(f"Total resistance = {{R_total:.2f}} Ω")
print(f"Total current    = {{I_total*1000:.3f}} mA")
print(f"Voltage across R1: {{V_R1:.3f}} V")
print(f"Voltage across R2∥R3: {{V_para:.3f}} V")
print(f"Current through R2: {{I2*1000:.3f}} mA")
print(f"Current through R3: {{I3*1000:.3f}} mA")
print(f"Check I2+I3 = {{(I2+I3)*1000:.3f}} mA (= total {{I_total*1000:.3f}} mA) ✓")
print(f"Total power = {{P_total*1000:.3f}} mW = {{V*I_total*1000:.3f}} mW ✓")
"""
    output = run_code(executor, code)
    return {
        "id": ex_id,
        "category": "physics",
        "difficulty": "intermediate",
        "query": (
            f"A {V}V source drives a circuit where R1={R1}Ω is in series with "
            f"R2={R2}Ω and R3={R3}Ω in parallel. Find the total resistance, "
            "current through each resistor, and power dissipated."
        ),
        "response": (
            "<think>\n"
            "Simplify: find R2∥R3 first, then add R1 in series.\n"
            "R_para = R2×R3/(R2+R3)\n"
            "Total I from Ohm's law; then use voltage divider for parallel section.\n"
            "</think>\n"
            f"<code>\n{code}</code>\n"
            f"<output>\n{output}\n</output>\n"
        ),
    }


# ---------------------------------------------------------------------------
# Multi-step
# ---------------------------------------------------------------------------

def make_roller_coaster(ex_id: str) -> dict:
    """Multi-step: energy → speed → centripetal condition."""
    random.seed(random.randint(1, 999))
    h_top = random.choice([5, 8, 10, 15])
    r_loop = random.choice([2, 3, 4])
    mass = random.choice([1, 2, 5])
    executor = PythonExecutor()
    code1 = f"""\
import math
g      = 9.81
mass   = {mass}     # kg
h_top  = {h_top}    # height of loop top above ground (m)
r      = {r_loop}   # loop radius (m)
h_base = 0          # ground level reference

# Step 1: minimum speed at top of loop for contact (centripetal condition)
# mg = mv²/r  → v_min = sqrt(g*r)
v_min_top = math.sqrt(g * r)
print(f"Minimum speed at loop top: v_min = √(gr) = {{v_min_top:.3f}} m/s")
print(f"(At this speed, normal force = 0, gravity provides all centripetal force)")
"""
    output1 = run_code(executor, code1)
    code2 = f"""\
# Step 2: energy conservation to find required launch height
# Use energy conservation: mgh_launch = mgh_top + ½mv_top²
# h_launch = h_top + v_top²/(2g)
h_launch = h_top + v_min_top**2 / (2*g)
print(f"\\nRequired launch height (energy conservation):")
print(f"  h_launch = h_top + v²/(2g)")
print(f"  h_launch = {{h_top}} + {{v_min_top**2:.3f}}/{{2*g:.3f}}")
print(f"  h_launch = {{h_launch:.3f}} m")
print(f"  (Minimum start height above ground to complete loop: {{h_launch:.3f}} m)")

# Step 3: speed at bottom of the loop
v_bottom = math.sqrt(2 * g * h_launch)
print(f"\\nSpeed at bottom of loop (launched from h_launch with v≈0):")
print(f"  v_bottom = √(2g·h_launch) = {{v_bottom:.3f}} m/s")

# Normal force at bottom
N_bottom = mass * (g + v_bottom**2 / r)
print(f"Normal force at bottom: N = m(g + v²/r) = {{N_bottom:.2f}} N")
print(f"  ({{N_bottom/(mass*g):.1f}}× body weight — classic 'roller coaster feeling')")
"""
    output2 = run_code(executor, code2)
    return {
        "id": ex_id,
        "category": "physics",
        "difficulty": "advanced",
        "query": (
            f"A roller coaster car (mass {mass} kg) must complete a vertical loop of radius "
            f"{r_loop} m, whose top is {h_top} m above the ground. "
            "Step 1: find the minimum speed at the top for the car to maintain contact. "
            "Step 2: use energy conservation to find the minimum launch height. "
            "Step 3: find the speed and normal force at the bottom of the loop."
        ),
        "response": (
            "<think>\n"
            "Three linked sub-problems:\n"
            "1. Centripetal condition at top: mg = mv²/r → v_min = √(gr)\n"
            "2. Energy conservation from launch: mgh = mgh_top + ½mv_top²\n"
            "3. Energy at bottom: v_bottom = √(2g·h_launch); N_bottom = m(g + v²/r)\n"
            "</think>\n"
            f"<code>\n{code1}</code>\n"
            f"<output>\n{output1}\n</output>\n"
            "<think>\n"
            "Now back-calculate the required launch height and find bottom-of-loop conditions.\n"
            "</think>\n"
            f"<code>\n{code2}</code>\n"
            f"<output>\n{output2}\n</output>\n"
            "The minimum launch height is more than the loop top height — the car needs "
            "extra height to carry enough KE to maintain contact at the top of the loop."
        ),
    }


# ---------------------------------------------------------------------------
# Master list
# ---------------------------------------------------------------------------

BUILDERS = [
    make_suvat,
    make_free_fall,
    make_projectile,
    make_newtons_second,
    make_inclined_plane,
    make_energy_conservation,
    make_work_energy,
    make_circular_motion,
    make_pendulum,
    make_wave_properties,
    make_ohms_law,
    make_series_parallel_circuit,
    make_roller_coaster,
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
            ex_id = f"phys_{idx:03d}"
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
    parser = argparse.ArgumentParser(description="Generate physics training examples")
    parser.add_argument("--output", default="training/datasets/science/physics/basic.jsonl")
    parser.add_argument("--count", type=int, default=120)
    args = parser.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Generating {args.count} physics examples...")
    examples = generate_examples(args.count)

    with open(out_path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")

    print(f"\nWrote {len(examples)} examples to {out_path}")


if __name__ == "__main__":
    main()
