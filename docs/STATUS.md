# Project Status

**Last updated:** 2026-04-05
**Phase:** 2 — v1 trained and running; v2 training in progress (chat template + prompt masking)

---

## What this project is

Training Qwen3-1.7B (LoRA) to reason through problems by writing and executing Python code within a single generation pass. The model generates `<think>`, `<model>`, `<code>` tokens; the runtime intercepts after `</code>`, runs the code, and injects `<output>` back into context so generation can continue with real results.

**Authoritative design:** `docs/DESIGN.md`

---

## Current state of the codebase

### Done and working
- `src/executor/python_exec.py` — inline exec() with shared namespace, timeout, `use_tool()` + `ToolNotAvailableError`
- `src/executor/vm_exec.py` — scratchpad QEMU bridge (for complex tasks)
- `src/inference/generation_loop.py` — custom generation loop that intercepts `</code>`
- `src/training/dataset.py` — JSONL loader for new format

### Datasets (928 examples total, all validated)

| Dataset | Examples | Topics |
|---------|----------|--------|
| `arithmetic/basic.jsonl` | 200 | percentages, compound interest, area, conversions |
| `algebra/basic.jsonl` | 150 | linear/quadratic, systems, polynomials, word problems |
| `geometry/basic.jsonl` | 120 | 2D/3D shapes, trig, Pythagorean, coordinate |
| `statistics/basic.jsonl` | 120 | descriptive, probability, distributions, correlation |
| `logic/basic.jsonl` | 80 | graph reachability, family relations, set logic, topo sort |
| `fourier/basic.jsonl` | 60 | FFT, Nyquist, Butterworth filter, Welch PSD, pipeline |
| `bayesian/basic.jsonl` | 60 | Bayes theorem, Beta updates, Naive Bayes, A/B testing |
| `montecarlo/basic.jsonl` | 70 | π estimation, integration, random walks, options, risk |
| `taguchi/basic.jsonl` | 60 | L4/L9 arrays, S/N ratios, signal vs noise, parameter sweeps |
| `tool_requests/basic.jsonl` | 8 | use_tool(), ToolNotAvailableError, graceful fallback |

### Tool-request protocol (new)
- `use_tool(name, **kwargs)` and `ToolNotAvailableError` are always in the executor namespace
- `executor.register_tool(name, fn)` lets the runtime provide tools at run time
- Training data teaches: try → catch ToolNotAvailableError → explain need → offer fallback
- See `docs/DESIGN.md` §3.4 for full spec

### Generators (all working)
- `training/scripts/generate_arithmetic.py`
- `training/scripts/generate_algebra.py`
- `training/scripts/generate_geometry.py`
- `training/scripts/generate_statistics.py`
- `training/scripts/generate_logic.py`
- `training/scripts/generate_fourier.py`
- `training/scripts/generate_bayesian.py`
- `training/scripts/generate_montecarlo.py`
- `training/scripts/generate_taguchi.py`
- `training/scripts/generate_tool_requests.py`

### Still empty (needs filling)
- `training/datasets/science/physics/` — kinematics, forces, energy, waves
- `training/datasets/science/chemistry/` — stoichiometry, ideal gas, thermodynamics
- `training/datasets/design/` — convert from `history/blueprint/training/`
- `training/datasets/multi_step/` — complex multi-cycle problems

### Not yet written
- `training/scripts/generate_physics.py`
- `training/scripts/generate_finance.py`
- `training/scripts/generate_programming.py`
- `training/scripts/convert_blueprint_to_design.py` (convert old blueprint data)

### Archived (do not rebuild)
- Everything in `history/` — ByteLogic/WASM and Blueprint-only era

---

## Hardware
- AMD Radeon MI50, 32GB VRAM
- ROCm 7.2 at `~/workspace/rocm/7.2.0/`
- Sandboxed Python via QEMU VM at `~/workspace/scratchpad/`
- venv at `./venv/`

---

## Current model status

### v1 (`output/worldmodel/final`) — trained 2026-04-04
- Execution loop working via guided generation (StoppingCriteria on `</think>`)
- Root issue: model learned proxy tokens instead of `<code>` tags — caused by
  training without prompt masking and without Qwen3 chat template
- Chat works via compatibility path in `generation_loop.py`

### v2 (`output/worldmodel_v2/`) — **training now**
- Same 1443 examples, 10 epochs
- Fixes applied: Qwen3 chat template + prompt-masked labels
- Expected: model generates `<code>` tags natively, no guided-generation fallback needed
- ETA: ~midnight 2026-04-05

## Immediate next steps (after v2 completes)

1. **Test v2** — `python chat.py` auto-detects the latest model
   - Check that `<code>` tags are generated natively (not proxy tokens)
   - If working, the guided-generation fallback path in `generation_loop.py` will be unused

2. **Generate multi-step dataset**
   - File: `training/scripts/generate_multistep.py`
   - Problems requiring 3+ code/output cycles with shared state
   - Target: ~60 examples → `training/datasets/multi_step/basic.jsonl`

3. **Generate chemistry dataset**
   - File: `training/scripts/generate_chemistry.py`
   - Topics: stoichiometry, ideal gas law, thermodynamics (ΔH, ΔG), equilibrium
   - Target: ~80 examples → `training/datasets/science/chemistry/basic.jsonl`

4. **v3 training run** — incorporate multi-step + chemistry datasets

---

## Training data format (quick reference)

```jsonl
{"id": "arith_001", "category": "arithmetic", "difficulty": "basic",
 "query": "What is 15% of 240?",
 "response": "<think>\nPercentage: multiply by 0.15\n</think>\n<code>\nresult = 240 * 0.15\nprint(f'15% of 240 = {result}')\n</code>\n<output>\n15% of 240 = 36.0\n</output>\n15% of 240 is 36."}
```

**Critical:** All `<output>` blocks must be pre-executed. Use `src/executor/python_exec.py` to generate them. Never handwrite outputs.

Multi-step: repeat `<think>`/`<code>`/`<output>` cycles. Python state persists across blocks.

---

## Token system (quick reference)

| Token | Generated by | Purpose |
|-------|-------------|---------|
| `<think>` | Model | Internal reasoning, hidden from user |
| `<model>` | Model | Design/plan — phrase to full Blueprint notation, optional |
| `<code>` | Model | Python to execute; must print() to produce output |
| `<output>` | **Runtime** (injected) | Execution result; model never generates this |

`use_tool(name, **kwargs)` is always available in code blocks. Raises `ToolNotAvailableError` if the tool isn't registered. Model is trained to handle this gracefully.

---

## Key files

| File | Purpose |
|------|---------|
| `docs/DESIGN.md` | Full architecture spec |
| `docs/STATUS.md` | This file — current state |
| `src/executor/python_exec.py` | Run Python code; provides use_tool() |
| `src/inference/generation_loop.py` | Custom generation with code interception |
| `src/training/dataset.py` | Load JSONL training data |
| `training/scripts/validate_datasets.py` | Validate all JSONL files |
