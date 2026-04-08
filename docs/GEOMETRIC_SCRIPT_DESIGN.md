# train_geometric.py — Script Design

## Purpose

Load an existing LoRA checkpoint, then run geometric vector-space optimization on the LoRA adapter parameters only. Produces a new checkpoint in a separate output directory.

## Command Line Interface

```bash
# Default: use latest trained model as starting point
python train_geometric.py

# Specify starting checkpoint
python train_geometric.py --model output/worldmodel/final

# Override hyperparameters
python train_geometric.py --jump-factor 0.15 --sphere-radius 0.05 --aggressive

# Conservative mode
python train_geometric.py --conservative

# Resume from a previous geometric run
python train_geometric.py --resume output/worldmodel_geometric/step_500

# Dry run (show config, don't train)
python train_geometric.py --dry-run
```

### Arguments

```
--model PATH            Starting LoRA checkpoint (default: auto-detect latest)
--output-dir PATH       Where to save geometric checkpoints (default: output/worldmodel_geometric)
--epochs N              Number of geometric optimization epochs (default: 10)
--batch-size N          Mini-batch size (default: 4)
--lr FLOAT              Base LoRA learning rate (default: 2e-4, same as original)

--jump-factor FLOAT     Initial vector jump multiplier (default: 0.1)
--sphere-radius FLOAT   Initial hypersphere radius (default: 0.02)
--jump-freq N           Vector jump frequency in steps (default: 50)
--sphere-freq N         Hypersphere frequency in steps (default: 150)
--num-jump-candidates N Random candidates in vector jump (default: 3)
--num-sphere-candidates N Candidates on hypersphere (default: 4)

--aggressive            Use aggressive hyperparameter preset
--conservative          Use conservative hyperparameter preset

--resume PATH           Resume from a geometric checkpoint
--dry-run               Print configuration and exit
--save-every N          Save checkpoint every N steps (default: 100)
```

## Architecture

### Module Structure

```
train_geometric.py
├── GeometricConfig (dataclass)
│   └── All hyperparameters + presets
├── GeometricOptimizer
│   ├── capture_lora_params() → Tensor
│   ├── apply_lora_params(tensor)
│   ├── vector_jump_step(θ_before, θ_after, batch) → (θ_new, accepted)
│   ├── hypersphere_step(θ_current, batch) → (θ_new, accepted)
│   └── evaluate_candidates(candidates, batch, baseline_loss) → (best_θ, best_loss)
├── GeometricTrainer
│   ├── load_model() — reuses train.py's load_model pattern
│   ├── build_dataloader() — reuses dataset.py
│   ├── training_loop() — main epoch/step loop
│   ├── save_checkpoint() — saves LoRA adapter to output dir
│   └── load_checkpoint() — for resume
└── main()
```

### GeometricOptimizer — Core Class

```python
class GeometricOptimizer:
    """
    Applies geometric exploration on top of standard LoRA gradient updates.
    Only operates on LoRA adapter parameters (lora_A, lora_B matrices).
    """

    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.jump_factor = config.jump_factor_init
        self.sphere_radius = config.sphere_radius_init
        self.stats = {"jump_accepted": 0, "jump_rejected": 0,
                       "sphere_accepted": 0, "sphere_rejected": 0}

    def _get_lora_param_names(self) -> List[str]:
        """Return sorted list of LoRA parameter names (lora_A, lora_B only)."""
        names = []
        for name, param in self.model.named_parameters():
            if "lora_" in name:
                names.append(name)
        return sorted(names)

    def capture_lora_params(self) -> torch.Tensor:
        """Flatten all LoRA params into a single vector."""
        params = []
        for name in self._get_lora_param_names():
            param = dict(self.model.named_parameters())[name]
            params.append(param.data.view(-1))
        return torch.cat(params).clone()

    def apply_lora_params(self, vector: torch.Tensor):
        """Unflatten vector back into LoRA params."""
        names = self._get_lora_param_names()
        offset = 0
        for name in names:
            param = dict(self.model.named_parameters())[name]
            numel = param.numel()
            param.data.copy_(vector[offset:offset + numel].view_as(param.data))
            offset += numel

    def _unit_vector(self, dim: int) -> torch.Tensor:
        """Random unit vector in R^dim."""
        v = torch.randn(dim, device=self.device, dtype=torch.float32)
        return v / v.norm()

    def evaluate_loss(self, batch) -> float:
        """Forward pass on batch, return scalar loss (no gradient needed)."""
        with torch.no_grad():
            outputs = self.model(**batch)
            return outputs.loss.item()

    def evaluate_candidates(self, candidates: List[torch.Tensor],
                            batch, baseline_loss: float) -> Tuple[torch.Tensor, float]:
        """
        Evaluate each candidate by loading params and running forward pass.
        Return (best_candidate, best_loss) among candidates and baseline.
        """
        best_loss = baseline_loss
        best_candidate = None
        original_params = self.capture_lora_params()

        for i, candidate in enumerate(candidates):
            self.apply_lora_params(candidate)
            loss = self.evaluate_loss(batch)
            if loss < best_loss:
                best_loss = loss
                best_candidate = candidate.clone()

        # Restore original params (caller decides whether to apply best)
        self.apply_lora_params(original_params)
        return best_candidate, best_loss

    def vector_jump_step(self, θ_before: torch.Tensor, θ_after: torch.Tensor,
                         batch) -> Tuple[bool, float]:
        """
        Extrapolate along gradient direction + random directions.
        Returns (accepted, best_loss).
        """
        gradient_direction = θ_after - θ_before
        jump_distance = gradient_direction.norm() * self.jump_factor

        candidates = []

        # Primary: extrapolate along gradient
        θ_jump = θ_after + self.jump_factor * gradient_direction
        candidates.append(θ_jump)

        # Random direction candidates
        for _ in range(self.config.num_jump_candidates):
            random_dir = self._unit_vector(θ_after.shape[0])
            θ_rand = θ_after + jump_distance * random_dir
            candidates.append(θ_rand)

        baseline_loss = self.evaluate_loss(batch)
        best_candidate, best_loss = self.evaluate_candidates(candidates, batch, baseline_loss)

        if best_candidate is not None and best_loss < baseline_loss:
            self.apply_lora_params(best_candidate)
            self.jump_factor = min(self.jump_factor * 1.02, self.config.jump_factor_init * 2.0)
            self.stats["jump_accepted"] += 1
            return True, best_loss
        else:
            self.jump_factor = max(0.01, self.jump_factor * self.config.jump_decay_on_fail)
            self.stats["jump_rejected"] += 1
            return False, baseline_loss

    def hypersphere_step(self, batch) -> Tuple[bool, float]:
        """
        Sample candidates on hypersphere around current position.
        Returns (accepted, best_loss).
        """
        θ_current = self.capture_lora_params()
        baseline_loss = self.evaluate_loss(batch)
        param_norm = θ_current.norm()
        absolute_radius = self.sphere_radius * param_norm

        candidates = []
        for _ in range(self.config.num_sphere_candidates):
            direction = self._unit_vector(θ_current.shape[0])
            θ_sphere = θ_current + absolute_radius * direction
            candidates.append(θ_sphere)

        best_candidate, best_loss = self.evaluate_candidates(candidates, batch, baseline_loss)

        if best_candidate is not None and best_loss < baseline_loss:
            self.apply_lora_params(best_candidate)
            self.sphere_radius = min(self.sphere_radius * 1.01, self.config.sphere_radius_init * 2.0)
            self.stats["sphere_accepted"] += 1
            return True, best_loss
        else:
            self.sphere_radius = max(0.001, self.sphere_radius * self.config.sphere_decay_on_fail)
            self.stats["sphere_rejected"] += 1
            return False, baseline_loss
```

### GeometricTrainer — Training Loop

```python
class GeometricTrainer:
    def __init__(self, config: GeometricConfig):
        self.config = config
        self.device = "cuda"
        self.start_step = 0
        self.global_step = 0

    def setup(self):
        """Load model, tokenizer, dataloader, optimizer, geometric optimizer."""
        self.model, self.tokenizer = self._load_model()
        self.dataloader = self._build_dataloader()
        self.optimizer = self._build_optimizer()
        self.geometric = GeometricOptimizer(self.model, self.config)

    def _load_model(self):
        """
        Load base model + LoRA adapter from checkpoint.
        Reuses the same pattern as train.py's load_model().
        If --resume is set, load from geometric output dir.
        Otherwise load from --model (existing LoRA checkpoint).
        """
        # ... same as train.py but returns (model, tokenizer)
        pass

    def _build_dataloader(self):
        """Use src.training.dataset for data loading."""
        # ... reuse existing data loading
        pass

    def _build_optimizer(self):
        """AdamW on LoRA parameters only."""
        lora_params = [p for n, p in self.model.named_parameters()
                       if "lora_" in n]
        return torch.optim.AdamW(lora_params, lr=self.config.lr)

    def training_loop(self):
        for epoch in range(self.config.epochs):
            for batch_idx, batch in enumerate(self.dataloader):
                step = self.global_step

                # ─── Standard gradient step ───────────────────────────
                θ_before = self.geometric.capture_lora_params()

                self.model.train()
                outputs = self.model(**batch)
                loss = outputs.loss
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

                θ_after = self.geometric.capture_lora_params()
                loss_post_grad = self.geometric.evaluate_loss(batch)

                # ─── Vector space jump ────────────────────────────────
                if step % self.config.jump_freq == 0 and step > 0:
                    accepted, loss_after_jump = self.geometric.vector_jump_step(
                        θ_before, θ_after, batch
                    )
                    if accepted:
                        loss = torch.tensor(loss_after_jump)

                # ─── Hypersphere search ───────────────────────────────
                if step % self.config.sphere_freq == 0 and step > 0:
                    accepted, loss_after_sphere = self.geometric.hypersphere_step(batch)
                    if accepted:
                        loss = torch.tensor(loss_after_sphere)

                # ─── Logging ──────────────────────────────────────────
                if step % self.config.logging_steps == 0:
                    self._log_step(step, loss.item(), epoch)

                # ─── Checkpoint ───────────────────────────────────────
                if step % self.config.save_every == 0 and step > 0:
                    self._save_checkpoint(step, epoch)

                # ─── Thermal monitoring ───────────────────────────────
                if self.thermal_controller:
                    self._check_thermal()

                self.global_step += 1

            self._save_checkpoint(self.global_step, epoch, suffix=f"epoch_{epoch}")

    def _log_step(self, step, loss, epoch):
        """Log step with geometric stats."""
        g = self.geometric
        logger.info(
            f"Step {step:5d} | Epoch {epoch} | Loss: {loss:.6f} | "
            f"Jump: {g.jump_factor:.4f} | Sphere: {g.sphere_radius:.5f} | "
            f"Jump acc: {g.stats['jump_accepted']}/{g.stats['jump_accepted'] + g.stats['jump_rejected']} | "
            f"Sphere acc: {g.stats['sphere_accepted']}/{g.stats['sphere_accepted'] + g.stats['sphere_rejected']}"
        )

    def _save_checkpoint(self, step, epoch, suffix=None):
        """Save LoRA adapter to output directory."""
        # ... save adapter_config.json + adapter_model.safetensors
        pass
```

## Data Flow

```
┌─────────────────────────────────────┐
│  training/history/*.jsonl            │  (same data as original training)
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  dataset.py                          │
│  - Qwen3 chat template               │
│  - Prompt token masking              │
│  - Same collation                    │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  GeometricTrainer.training_loop()   │
│                                      │
│  1. optimizer.step()                 │  ← standard LoRA gradient
│  2. geometric.vector_jump_step()     │  ← every 50 steps
│  3. geometric.hypersphere_step()     │  ← every 150 steps
│  4. log + checkpoint                 │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  output/worldmodel_geometric/        │
│    step_100/  (LoRA adapter)         │
│    step_200/                         │
│    ...                               │
│    final/                            │
└─────────────────────────────────────┘
```

## Key Implementation Notes

### 1. Parameter Copying Must Be Clean

```python
# CRITICAL: use .data.copy_() not .detach()
# .detach() still shares storage; .data.copy_() is a true copy
param.data.copy_(candidate_segment.view_as(param.data))
```

### 2. No Gradient on Candidate Evaluation

```python
@torch.no_grad()
def evaluate_loss(self, batch):
    # Candidate evaluation is purely for comparison — no backprop needed
    outputs = self.model(**batch)
    return outputs.loss.item()
```

### 3. Preserve Base Model Weights

Only LoRA parameters change. The base model (Qwen3-1.7B) is **frozen** and never modified:

```python
for name, param in self.model.named_parameters():
    if "lora_" not in name:
        param.requires_grad = False
```

### 4. Checkpoint Format

Saves in standard LoRA adapter format so `chat.py` and `infer.py` can load it directly:

```
output/worldmodel_geometric/step_500/
├── adapter_config.json       (same as original)
├── adapter_model.safetensors (updated LoRA weights)
└── training_state.json       (step, epoch, jump_factor, sphere_radius, stats)
```

### 5. Resume Capability

`training_state.json` stores all geometric optimizer state:

```json
{
    "global_step": 500,
    "epoch": 2,
    "jump_factor": 0.098,
    "sphere_radius": 0.0197,
    "stats": {
        "jump_accepted": 4,
        "jump_rejected": 6,
        "sphere_accepted": 1,
        "sphere_rejected": 2
    },
    "optimizer_state": "optimizer_state.pt"
}
```

### 6. Shared Components with train.py

Reuse, don't duplicate:

| Component | Source | Usage |
|---|---|---|
| Model loading | `train.py` pattern | Load base + LoRA |
| Dataset | `src/training/dataset.py` | Same data pipeline |
| GPU monitor | `src/training/gpu_monitor.py` | Same thermal control |
| Tokenizer | `train.py` pattern | Same chat template |
| Progress callback | Simplified inline logging | No TrainerCallback needed |
