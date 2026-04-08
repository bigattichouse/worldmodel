# Geometric Vector-Space Optimization for WorldModel LoRA Training

## Overview

Integrate geometric parameter-space exploration (vector jumps + hypersphere search) as a **meta-optimization layer** on top of the existing LoRA fine-tuning pipeline. The current training works and produces valid checkpoints — this adds a second pass that explores beyond gradient-descent local minima.

## Core Concept

Standard LoRA training follows gradient descent step-by-step. This technique:
1. Takes the normal gradient step
2. **Extrapolates further** along the gradient direction (vector jump)
3. **Probes random directions** at the same distance (hypersphere search)
4. Accepts candidates **only if they strictly improve loss**

Result: escapes shallow local minima, finds better basins the gradient alone wouldn't reach.

## Design Decisions

### Why a Separate Script (Not Modify Existing)

| Aspect | Decision | Rationale |
|---|---|---|
| **Existing train.py** | Keep as-is | Works, produces valid checkpoints, just needs more time |
| **New script** | `train_geometric.py` | Different training paradigm, different hyperparameters, different loss tracking |
| **Checkpoint compatibility** | Load existing LoRA adapters as starting point | No need to retrain from scratch — build on what's already learned |
| **Risk** | Zero risk to current pipeline | If geometric training doesn't help, current training is untouched |

### Checkpoint Strategy

Load the latest trained LoRA adapter (`output/worldmodel/final/`) as the **initial parameter state**. This means:
- The model already knows the `<code>`, `</code>`, `<output>`, `</think>`, `<think>` token semantics
- Geometric exploration starts from an already-reasonable region of parameter space
- Much more effective than starting from the raw base model

If geometric training degrades quality, we can always fall back to the original checkpoint.

### Parameter Scope: LoRA Matrices Only

**Do NOT flatten the entire 1.7B parameter model.** Only operate on LoRA adapter matrices:

```
For each target_module in [q_proj, v_proj, k_proj, o_proj, gate_proj, up_proj, down_proj]:
    lora_A: [rank, hidden_dim]   →  typically [8, 2048]  = 16,384 params
    lora_B: [hidden_dim, rank]   →  typically [2048, 8]  = 16,384 params
```

With rank=8 and 7 target modules × 2 matrices each:
- **Total geometric parameters**: ~262K (vs 1.7B full model)
- This is small enough to meaningfully explore with vector operations
- Still captures all the learned behavior for the code-generation task

### Memory Management

Each candidate evaluation requires:
1. Copy current LoRA params → stash
2. Apply candidate params to model
3. Forward pass on mini-batch → get loss
4. Restore original params (or keep candidate if it wins)

With 262K params at FP32: **~1 MB per copy**. Even 8 candidates = negligible memory overhead.

The forward pass itself (1.7B model in FP32) dominates memory at ~6.8 GB, which is the same as normal training.

## Algorithm

### Per Mini-Batch (Pseudocode)

```
for batch in dataloader:
    # ─── Step 1: Standard LoRA gradient step ───────────────────────────
    θ_before = capture_lora_params()
    
    outputs = model(batch)
    loss_0 = loss_fn(outputs, targets)
    loss_0.backward()
    optimizer.step()
    optimizer.zero_grad()
    
    θ_after = capture_lora_params()
    gradient_direction = θ_after - θ_before
    loss_1 = evaluate_loss(model, batch)  # post-gradient loss
    
    # ─── Step 2: Vector space jump (every N steps) ─────────────────────
    if step % vector_jump_freq == 0:
        candidates = []
        
        # Primary candidate: extrapolate along gradient direction
        θ_jump = θ_after + jump_factor * gradient_direction
        
        # Random direction candidates at same distance
        jump_distance = ||gradient_direction|| * jump_factor
        for _ in range(num_random_candidates):
            random_dir = random_unit_vector(θ_after.shape)
            θ_rand = θ_after + jump_distance * random_dir
            candidates.append(θ_rand)
        candidates.append(θ_jump)
        
        best_candidate, best_loss = evaluate_candidates(candidates, batch, loss_1)
        
        if best_loss < loss_1:  # Aggressive acceptance
            apply_lora_params(best_candidate)
            jump_factor *= 1.02  # Slight increase on success
        else:
            jump_factor = max(0.01, jump_factor * 0.99)  # Decay on failure
    
    # ─── Step 3: Hypersphere search (every M steps) ───────────────────
    if step % sphere_freq == 0:
        θ_current = capture_lora_params()
        sphere_loss = evaluate_loss(model, batch)
        
        candidates = []
        for _ in range(num_sphere_candidates):
            random_dir = random_unit_vector(θ_current.shape)
            θ_sphere = θ_current + sphere_radius * random_dir
            candidates.append(θ_sphere)
        
        best_candidate, best_loss = evaluate_candidates(candidates, batch, sphere_loss)
        
        if best_loss < sphere_loss:
            apply_lora_params(best_candidate)
            sphere_radius *= 1.01  # Slight increase on success
        else:
            sphere_radius = max(0.001, sphere_radius * 0.995)  # Decay on failure
```

### Key Differences from ShapeOfThought Original

| Aspect | ShapeOfThought (2.8M net) | WorldModel (1.7B + LoRA) |
|---|---|---|
| Parameter scope | Full network | LoRA matrices only (~262K) |
| Jump factor | 10.0 (full network gradients) | 0.05–0.15 (LoRA deltas) |
| Sphere radius | 1.0 | 0.01–0.05 of param magnitude |
| Random candidates | 4 (jump) + 7 (sphere) | 2–4 each (memory budget) |
| Acceptance | V2 < V1 | Same (aggressive) |
| Frequency | Every step (small net) | Every 50–200 steps (LLM) |

## Hyperparameter Recommendations

### Primary (Start Here)

```
vector_jump_freq     = 50       # Apply vector jump every 50 steps
sphere_freq          = 150      # Apply hypersphere every 150 steps
jump_factor_init     = 0.1      # Initial extrapolation multiplier
sphere_radius_init   = 0.02     # Initial sphere radius (fraction of param norm)
num_random_candidates = 3       # Random directions in vector jump
num_sphere_candidates = 4       # Points on hypersphere
jump_decay_on_fail   = 0.99     # Decay jump factor on rejection
sphere_decay_on_fail = 0.995    # Decay radius on rejection
```

### Conservative (If Aggressive Fails)

```
vector_jump_freq     = 100
sphere_freq          = 300
jump_factor_init     = 0.05
sphere_radius_init   = 0.01
num_random_candidates = 2
num_sphere_candidates = 3
```

### Aggressive (If Conservative Works Well)

```
vector_jump_freq     = 25
sphere_freq          = 75
jump_factor_init     = 0.15
sphere_radius_init   = 0.05
num_random_candidates = 4
num_sphere_candidates = 6
```

## Expected Outcomes

| Metric | Estimate |
|---|---|
| Compute overhead | +10–20% over standard training |
| Memory overhead | +15–30% (candidate storage) |
| Convergence speed | 10–25% faster to target loss |
| Final quality | 2–8% improvement on downstream tasks |
| Failure risk | Low — aggressive acceptance means bad candidates are rejected, params stay at gradient-descent quality |

## Risk Mitigation

| Risk | Mitigation |
|---|---|
| Candidate evaluation corrupts model | Params are copied → candidate applied → evaluated → original restored if rejected |
| Numerical instability | FP32 precision for all geometric operations; clip candidate param norms |
| Training diverges | Aggressive acceptance criterion prevents degradation; always falls back to gradient step |
| VRAM exhaustion | Limit candidate count; use same batch size as standard training |

## File Structure

```
train_geometric.py          # New training script
docs/GEOMETRIC_TRAINING.md  # This design doc
```

The script reuses:
- `src/training/dataset.py` — same data loading, same Qwen3 chat template
- `src/training/gpu_monitor.py` — same thermal monitoring
- `train.py` model loading logic — same LoRA adapter loading, same base model
