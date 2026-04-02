# Project Status

**Last updated:** 2026-04-02
**Phase:** 2 — Dataset complete (1443 examples), training pipeline validated

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

## Immediate next steps (in order)

### After power cycle (GPU recovery)
The MI50 had a dirty AtomBIOS state — GPU POST was failing with
`atombios stuck executing 4EC8`. This is transient; cold boot resets it.

1. **Verify GPU is back**
   ```bash
   amd-smi list
   # Should show: Instinct MI50 or similar
   ```

2. **Smoke-test training run** (3 epochs, core categories only)
   ```bash
   ./train_rocm.sh --categories arithmetic,algebra,geometry,statistics \
                   --epochs 3 --output ./output/smoke_test
   ```
   Expected: GPU detected, loss decreasing, checkpoint saved in `output/smoke_test/`

3. **Full training run** (all 1443 examples, 10 epochs)
   ```bash
   ./train_rocm.sh --output ./output/worldmodel_v1
   ```
   - ROCm env vars + LD_LIBRARY_PATH auto-set by train_rocm.sh
   - Float32 only (no fp16/bf16/quantization) — required for gfx906 stability
   - Saves best checkpoint; EarlyStopping patience=3

4. **Evaluate trained model vs base**
   ```bash
   python infer.py --model ./output/worldmodel_v1
   # Compare responses to base: python infer.py --model ~/workspace/model/Qwen3-1.7B
   ```

5. **Generate chemistry dataset** (not yet written)
   - File: `training/scripts/generate_chemistry.py`
   - Topics: stoichiometry, ideal gas law, thermodynamics (ΔH, ΔG), equilibrium
   - Target: ~80 examples → `training/datasets/science/chemistry/basic.jsonl`

6. **Generate multi-step dataset** (not yet written)
   - File: `training/scripts/generate_multistep.py`
   - Purpose: problems that genuinely require 3+ code/output cycles, with state
     passed between blocks (e.g. simulate → analyse → optimise)
   - Target: ~60 examples → `training/datasets/multi_step/basic.jsonl`

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
