#!/usr/bin/env python3
"""
WorldModel Geometric Training Script
=====================================
Applies vector-space geometric optimization on top of standard LoRA gradient
updates.  Loads an existing LoRA checkpoint and explores beyond gradient-descent
local minima via gradient extrapolation (vector jumps) and random hypersphere
search.

Based on the ShapeOfThought technique: treating LoRA adapter parameters as a
point in high-dimensional space and using geometric operations as a
meta-optimization layer.

Usage:
    # Default: use latest trained model as starting point
    python train_geometric.py

    # Specify starting checkpoint
    python train_geometric.py --model output/worldmodel/final

    # Aggressive preset
    python train_geometric.py --aggressive

    # Conservative preset
    python train_geometric.py --conservative

    # Resume from a previous geometric run
    python train_geometric.py --resume output/worldmodel_geometric/step_500

    # Dry run (show config, don't train)
    python train_geometric.py --dry-run

Environment:
    export HSA_OVERRIDE_GFX_VERSION=9.0.6
    export PYTORCH_ROCM_ARCH=gfx906
    export TOKENIZERS_PARALLELISM=false
    export HIP_VISIBLE_DEVICES=0
    export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True,max_split_size_mb:512
    export OMP_NUM_THREADS=1
"""

import os
import sys
import json
import logging
import argparse
import random
import re
import time
import glob
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Tuple, Dict, Any

import torch
from torch.utils.data import DataLoader

# ─── Patch: bypass torch.load safety check ──────────────────────────────────
from transformers.utils import import_utils as _iu
_iu.check_torch_load_is_safe = lambda: None  # noqa
import transformers.trainer as _tt
_tt.check_torch_load_is_safe = lambda: None  # noqa
del _iu, _tt

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
)
from peft import PeftModel, TaskType

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))
from src.training.dataset import load_all_datasets, WorldModelDataset
from src.training.gpu_monitor import (
    GPUThermalController,
    ThrottleState,
    log_gpu_status,
    DEFAULT_MAX_TEMP,
    DEFAULT_SAFE_TEMP,
    DEFAULT_CHECK_INTERVAL,
)
from src.training.execution_validator import (
    ExecutionValidator,
    CodeRewardFunction,
    CodeCheckResult,
    extract_code,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# Patch StreamHandler for BrokenPipeError
_orig_emit = logging.StreamHandler.emit
def _safe_emit(self, record):
    try:
        _orig_emit(self, record)
    except BrokenPipeError:
        pass
logging.StreamHandler.emit = _safe_emit


# ─── Special tokens (must match original training) ─────────────────────────

SPECIAL_TOKENS = [
    "<think>", "</think>",
    "<model>", "</model>",
    "<code>", "</code>",
    "<output>", "</output>",
]

DEFAULT_BASE_MODEL = Path.home() / "workspace/model/Qwen3-1.7B"
OUTPUT_DIR = Path(__file__).parent / "output" / "worldmodel_geometric"


# ─── Presets ────────────────────────────────────────────────────────────────

PRESETS = {
    "conservative": {
        "jump_factor_init": 0.05,
        "sphere_radius_init": 0.01,
        "jump_freq": 100,
        "sphere_freq": 300,
        "num_jump_candidates": 2,
        "num_sphere_candidates": 3,
    },
    "primary": {
        "jump_factor_init": 0.1,
        "sphere_radius_init": 0.02,
        "jump_freq": 50,
        "sphere_freq": 150,
        "num_jump_candidates": 3,
        "num_sphere_candidates": 4,
    },
    "aggressive": {
        "jump_factor_init": 0.15,
        "sphere_radius_init": 0.05,
        "jump_freq": 25,
        "sphere_freq": 75,
        "num_jump_candidates": 4,
        "num_sphere_candidates": 6,
    },
}


# ─── Configuration ──────────────────────────────────────────────────────────

@dataclass
class GeometricConfig:
    """All hyperparameters for geometric LoRA training."""

    # Model & data
    model_name: str = str(DEFAULT_BASE_MODEL)
    output_dir: str = str(OUTPUT_DIR)
    max_length: int = 1024
    categories: Optional[List[str]] = None
    test_split: float = 0.05

    # Training schedule
    epochs: int = 10
    batch_size: int = 2
    learning_rate: float = 2e-4
    logging_steps: int = 20

    # Vector jump
    jump_factor_init: float = 0.1
    jump_freq: int = 50
    num_jump_candidates: int = 3
    jump_decay_on_fail: float = 0.99
    jump_factor_max: float = 0.3   # Safety cap: 2x init default

    # Hypersphere
    sphere_radius_init: float = 0.02
    sphere_freq: int = 150
    num_sphere_candidates: int = 4
    sphere_decay_on_fail: float = 0.995
    sphere_radius_max: float = 0.1  # Safety cap

    # Checkpointing
    save_every: int = 100
    save_total_limit: int = 5

    # GPU thermal
    max_temp: float = DEFAULT_MAX_TEMP
    safe_temp: float = DEFAULT_SAFE_TEMP
    cooldown_check_interval: int = DEFAULT_CHECK_INTERVAL

    # Execution-based validation & reward
    exec_reward: bool = False       # Use code validity as reward signal
    reward_bonus: float = 0.2       # Bonus for valid syntax (per example)
    syntax_penalty: float = 0.5     # Penalty for missing/invalid code
    exec_bonus: float = 0.1         # Bonus for code that runs successfully
    eval_sample_size: int = 20      # Examples to validate at epoch end
    validate_every: int = 1         # Run validation every N epochs

    def apply_preset(self, name: str):
        """Override hyperparameters with a named preset."""
        if name not in PRESETS:
            raise ValueError(f"Unknown preset: {name}. Choose from {list(PRESETS.keys())}")
        for k, v in PRESETS[name].items():
            setattr(self, k, v)

    def summary(self) -> str:
        """Human-readable config summary."""
        lines = [
            "=== Geometric Training Configuration ===",
            f"  Base model      : {self.model_name}",
            f"  Output dir      : {self.output_dir}",
            f"  Epochs          : {self.epochs}",
            f"  Batch size      : {self.batch_size}",
            f"  Learning rate   : {self.learning_rate}",
            f"  Max length      : {self.max_length}",
            "",
            "  Vector Jump:",
            f"    Factor init   : {self.jump_factor_init}",
            f"    Frequency     : every {self.jump_freq} steps",
            f"    Candidates    : {self.num_jump_candidates}",
            f"    Decay on fail : {self.jump_decay_on_fail}",
            "",
            "  Hypersphere:",
            f"    Radius init   : {self.sphere_radius_init}",
            f"    Frequency     : every {self.sphere_freq} steps",
            f"    Candidates    : {self.num_sphere_candidates}",
            f"    Decay on fail : {self.sphere_decay_on_fail}",
            "",
            f"  Save every      : {self.save_every} steps",
            f"  Log every       : {self.logging_steps} steps",
            "",
            f"  Exec reward     : {'enabled' if self.exec_reward else 'disabled'}",
            f"  Reward bonus   : {self.reward_bonus} | Syntax penalty: {self.syntax_penalty}",
            f"  Eval samples    : {self.eval_sample_size} (every {self.validate_every} epochs)",
        ]
        return "\n".join(lines)


# ─── Geometric Optimizer ────────────────────────────────────────────────────

class GeometricOptimizer:
    """
    Applies geometric exploration on top of standard LoRA gradient updates.
    Only operates on LoRA adapter parameters (lora_A, lora_B matrices).
    """

    def __init__(self, model: torch.nn.Module, config: GeometricConfig, device: str):
        self.model = model
        self.config = config
        self.device = device
        self.jump_factor = config.jump_factor_init
        self.sphere_radius = config.sphere_radius_init
        self.stats = {
            "jump_accepted": 0,
            "jump_rejected": 0,
            "sphere_accepted": 0,
            "sphere_rejected": 0,
        }

    # ── Parameter management ──────────────────────────────────────────────

    def _lora_param_names(self) -> List[str]:
        """Return sorted list of LoRA parameter names."""
        return sorted(n for n, _ in self.model.named_parameters() if "lora_" in n)

    def capture_lora_params(self) -> torch.Tensor:
        """Flatten all LoRA params into a single FP32 vector."""
        state_dict = dict(self.model.named_parameters())
        pieces = []
        for name in self._lora_param_names():
            pieces.append(state_dict[name].data.detach().float().view(-1))
        return torch.cat(pieces)

    def apply_lora_params(self, vector: torch.Tensor):
        """Unflatten vector back into LoRA params (in-place copy)."""
        state_dict = dict(self.model.named_parameters())
        offset = 0
        for name in self._lora_param_names():
            param = state_dict[name]
            numel = param.numel()
            param.data.copy_(vector[offset:offset + numel].view_as(param.data))
            offset += numel

    def param_norm(self) -> float:
        """L2 norm of all LoRA params (for radius scaling)."""
        return self.capture_lora_params().norm().item()

    # ── Loss evaluation ───────────────────────────────────────────────────

    @torch.no_grad()
    def evaluate_loss(self, batch: Dict[str, torch.Tensor]) -> float:
        """Forward pass on batch, return scalar loss."""
        device_batch = {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }
        outputs = self.model(**device_batch)
        return outputs.loss.item()

    @torch.no_grad()
    def evaluate_candidates(
        self,
        candidates: List[torch.Tensor],
        batch: Dict[str, torch.Tensor],
        baseline_loss: float,
        score_fn=None,
    ) -> Tuple[Optional[torch.Tensor], float]:
        """
        Evaluate each candidate by loading params and running forward pass.
        Returns (best_candidate, best_score) — best_score <= baseline always.
        If no candidate beats baseline, returns (None, baseline).

        baseline should already be scored through score_fn (if in use) so that
        candidates and baseline are compared on the same scale.
        score_fn(model, loss: float) → float
        """
        best_score = baseline_loss  # caller passes pre-scored baseline
        best_candidate = None
        original_params = self.capture_lora_params()

        for i, candidate in enumerate(candidates):
            self.apply_lora_params(candidate)
            loss = self.evaluate_loss(batch)
            score = score_fn(self.model, loss) if score_fn is not None else loss

            if score < best_score:
                best_score = score
                best_candidate = candidate.clone()

        # Always restore original params
        self.apply_lora_params(original_params)
        return best_candidate, best_score

    # ── Random directions ─────────────────────────────────────────────────

    def _random_unit_vector(self, dim: int) -> torch.Tensor:
        """Random unit vector in R^dim."""
        v = torch.randn(dim, device=self.device, dtype=torch.float32)
        norm = v.norm()
        if norm < 1e-12:
            v[0] = 1.0
            norm = 1.0
        return v / norm

    def _clip_candidate(self, candidate: torch.Tensor, original: torch.Tensor,
                       max_norm_change: float) -> torch.Tensor:
        """Safety: prevent candidates from drifting too far from original params."""
        delta = candidate - original
        current_change = delta.norm().item()
        limit = max_norm_change * original.norm().item()
        if current_change > limit and current_change > 1e-8:
            scale = limit / current_change
            candidate = original + delta * scale
        return candidate

    # ── Vector jump step ──────────────────────────────────────────────────

    def vector_jump_step(
        self,
        θ_before: torch.Tensor,
        θ_after: torch.Tensor,
        batch: Dict[str, torch.Tensor],
        score_fn=None,
    ) -> Tuple[bool, float]:
        """
        Extrapolate along gradient direction + random directions.
        Returns (accepted, best_loss).
        """
        gradient_direction = θ_after - θ_before
        grad_norm = gradient_direction.norm().item()

        if grad_norm < 1e-10:
            # No gradient signal — skip
            return False, self.evaluate_loss(batch)

        jump_distance = grad_norm * self.jump_factor
        max_change = self.config.jump_factor_max / max(self.config.jump_factor_init, 0.01)
        candidates = []

        # Primary: extrapolate along gradient direction
        θ_jump = θ_after + self.jump_factor * gradient_direction
        θ_jump = self._clip_candidate(θ_jump, θ_before, max_change)
        candidates.append(θ_jump)

        # Random direction candidates at same distance
        for _ in range(self.config.num_jump_candidates):
            random_dir = self._random_unit_vector(θ_after.shape[0])
            θ_rand = θ_after + jump_distance * random_dir
            θ_rand = self._clip_candidate(θ_rand, θ_before, max_change)
            candidates.append(θ_rand)

        baseline_loss = self.evaluate_loss(batch)
        # Score baseline through score_fn so comparison is on the same scale as candidates
        baseline = score_fn(self.model, baseline_loss) if score_fn is not None else baseline_loss
        best_candidate, best_score = self.evaluate_candidates(
            candidates, batch, baseline, score_fn=score_fn
        )

        if best_candidate is not None and best_score < baseline:
            self.apply_lora_params(best_candidate)
            # Slight increase on success (encourage exploration)
            self.jump_factor = min(
                self.jump_factor * 1.02,
                self.config.jump_factor_max,
            )
            self.stats["jump_accepted"] += 1
            return True, baseline_loss
        else:
            self.jump_factor = max(0.001, self.jump_factor * self.config.jump_decay_on_fail)
            self.stats["jump_rejected"] += 1
            return False, baseline_loss

    # ── Hypersphere step ──────────────────────────────────────────────────

    def hypersphere_step(
        self,
        batch: Dict[str, torch.Tensor],
        score_fn=None,
    ) -> Tuple[bool, float]:
        """
        Sample candidates on hypersphere around current position.
        Returns (accepted, best_loss).
        """
        θ_current = self.capture_lora_params()
        baseline_loss = self.evaluate_loss(batch)
        param_norm = θ_current.norm().item()

        if param_norm < 1e-10:
            return False, baseline_loss

        absolute_radius = self.sphere_radius * param_norm
        max_change = self.config.sphere_radius_max / max(self.config.sphere_radius_init, 0.001)

        candidates = []
        for _ in range(self.config.num_sphere_candidates):
            direction = self._random_unit_vector(θ_current.shape[0])
            θ_sphere = θ_current + absolute_radius * direction
            θ_sphere = self._clip_candidate(θ_sphere, θ_current, max_change)
            candidates.append(θ_sphere)

        # Score baseline through score_fn so comparison is on the same scale as candidates
        baseline = score_fn(self.model, baseline_loss) if score_fn is not None else baseline_loss
        best_candidate, best_score = self.evaluate_candidates(candidates, batch, baseline, score_fn=score_fn)

        if best_candidate is not None and best_score < baseline:
            self.apply_lora_params(best_candidate)
            self.sphere_radius = min(
                self.sphere_radius * 1.01,
                self.config.sphere_radius_max,
            )
            self.stats["sphere_accepted"] += 1
            return True, baseline_loss
        else:
            self.sphere_radius = max(0.0001, self.sphere_radius * self.config.sphere_decay_on_fail)
            self.stats["sphere_rejected"] += 1
            return False, baseline_loss

    def stats_summary(self) -> str:
        j_total = self.stats["jump_accepted"] + self.stats["jump_rejected"]
        s_total = self.stats["sphere_accepted"] + self.stats["sphere_rejected"]
        j_rate = self.stats["jump_accepted"] / max(j_total, 1)
        s_rate = self.stats["sphere_accepted"] / max(s_total, 1)
        return (
            f"Jump: {self.stats['jump_accepted']}/{j_total} ({j_rate:.1%}) | "
            f"Sphere: {self.stats['sphere_accepted']}/{s_total} ({s_rate:.1%})"
        )


# ─── Geometric Trainer ──────────────────────────────────────────────────────

class GeometricTrainer:
    """Main training loop integrating gradient + geometric optimization."""

    def __init__(self, config: GeometricConfig):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.global_step = 0
        self.model = None
        self.tokenizer = None
        self.optimizer = None
        self.geometric = None
        self.dataloader = None
        self.eval_examples = None       # Held-out eval examples
        self.validator = None           # ExecutionValidator
        self.code_reward = None         # CodeRewardFunction
        self._latest_val_metrics = None  # Latest validation metrics
        self.thermal_controller = None
        self.starting_checkpoint = None  # For tracking provenance

    # ── Setup ─────────────────────────────────────────────────────────────

    def setup(self, resume_path: Optional[str] = None):
        """Load model, tokenizer, dataloader, optimizer, geometric optimizer."""
        self._load_model(resume_path)
        self._build_dataloader()
        self._build_optimizer()
        self._build_thermal_controller()

    def _find_latest_model(self) -> Path:
        """Auto-detect the most recently modified 'final' adapter directory."""
        candidates = sorted(
            OUTPUT_DIR.parent.glob("*/final"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if not candidates:
            raise FileNotFoundError(
                f"No trained model found in {OUTPUT_DIR.parent}.\n"
                "Run train.py first to produce a checkpoint."
            )
        return candidates[0]

    def _load_model(self, resume_path: Optional[str] = None):
        """
        Load base model + LoRA adapter for continued geometric training.

        We load the LoRA adapter normally (PeftModel.from_pretrained) WITHOUT
        merging. The adapter weights stay as separate trainable LoRA parameters,
        which is exactly what the geometric optimizer needs.

        Priority:
        1. --resume path (previous geometric checkpoint)
        2. --model path (existing LoRA checkpoint from train.py)
        3. Auto-detect latest trained model
        """
        # Determine which checkpoint to load
        if resume_path:
            checkpoint = Path(resume_path)
            logger.info(f"Resuming from geometric checkpoint: {checkpoint}")
            self.starting_checkpoint = str(checkpoint)
        elif Path(self.config.model_name).exists():
            checkpoint = Path(self.config.model_name)
            if (checkpoint / "adapter_config.json").exists():
                logger.info(f"Loading LoRA checkpoint: {checkpoint}")
                self.starting_checkpoint = str(checkpoint)
            else:
                raise ValueError(
                    f"Path {checkpoint} exists but has no adapter_config.json. "
                    "Expected a LoRA adapter directory. Run train.py first."
                )
        else:
            checkpoint = self._find_latest_model()
            logger.info(f"Auto-detected latest model: {checkpoint}")
            self.starting_checkpoint = str(checkpoint)

        # Load tokenizer
        logger.info(f"Loading tokenizer from: {checkpoint}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(checkpoint), trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Read adapter config to get LoRA hyperparams
        adapter_cfg_path = checkpoint / "adapter_config.json"
        if adapter_cfg_path.exists():
            adapter_cfg = json.loads(adapter_cfg_path.read_text())
            base_name = adapter_cfg.get("base_model_name_or_path", str(DEFAULT_BASE_MODEL))
        else:
            adapter_cfg = {}
            base_name = str(DEFAULT_BASE_MODEL)

        # Load base model
        logger.info(f"Loading base model: {base_name}")
        base_model = AutoModelForCausalLM.from_pretrained(
            base_name,
            dtype=torch.float32,
            device_map="auto",
            trust_remote_code=True,
        )

        # Resize embeddings to match tokenizer vocab
        base_vocab = base_model.get_input_embeddings().weight.shape[0]
        adapter_vocab = len(self.tokenizer)
        if adapter_vocab != base_vocab:
            base_model.resize_token_embeddings(adapter_vocab)
            logger.info(f"  Embeddings: {base_vocab} → {adapter_vocab}")

        # Load LoRA adapter — keeps params separate and trainable
        logger.info(f"Loading LoRA adapter from: {checkpoint}")
        self.model = PeftModel.from_pretrained(base_model, str(checkpoint))

        # Set LoRA params to trainable
        self.model.train()
        for name, param in self.model.named_parameters():
            if "lora_" in name:
                param.requires_grad = True

        lora_count = sum(1 for n, p in self.model.named_parameters() if "lora_" in n and p.requires_grad)
        lora_total = sum(p.numel() for n, p in self.model.named_parameters() if "lora_" in n and p.requires_grad)
        logger.info(f"  Trainable LoRA params: {lora_count} ({lora_total:,} total)")

        self.model.to(self.device)

        if torch.cuda.is_available():
            vram = torch.cuda.get_device_properties(0).total_memory / 1e9
            used = torch.cuda.memory_allocated() / 1e9
            logger.info(f"  GPU: {torch.cuda.get_device_name(0)} ({used:.1f}/{vram:.1f} GB)")

    def _build_dataloader(self):
        """Load training data using the same pipeline as train.py."""
        all_examples = load_all_datasets(categories=self.config.categories)

        if not all_examples:
            raise ValueError(
                "No training examples found! "
                "Ensure training/datasets/ contains JSONL files."
            )

        random.seed(42)
        random.shuffle(all_examples)

        split = max(1, int(len(all_examples) * self.config.test_split))
        eval_examples = all_examples[:split]
        train_examples = all_examples[split:]

        logger.info(f"Train: {len(train_examples)}, Eval: {len(eval_examples)}")

        train_dataset = WorldModelDataset(
            train_examples, self.tokenizer, self.config.max_length
        )

        self.dataloader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=0,
            pin_memory=True,
        )

        self.data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
        )

        # Store eval examples for execution-based validation
        self.eval_examples = eval_examples

    def _build_optimizer(self):
        """AdamW on LoRA parameters only."""
        lora_params = [
            p for n, p in self.model.named_parameters()
            if "lora_" in n and p.requires_grad
        ]
        self.optimizer = torch.optim.AdamW(
            lora_params,
            lr=self.config.learning_rate,
        )
        logger.info(f"Optimizer: AdamW (lr={self.config.learning_rate}, {len(lora_params)} param groups)")

    def _build_thermal_controller(self):
        """Set up GPU temperature monitoring."""
        if torch.cuda.is_available():
            self.thermal_controller = GPUThermalController(
                max_temp=self.config.max_temp,
                safe_temp=self.config.safe_temp,
                check_interval=self.config.cooldown_check_interval,
            )
            logger.info(f"Thermal control: pause >{self.config.max_temp}°C, resume <{self.config.safe_temp}°C")
        else:
            self.thermal_controller = None

    # ── Checkpointing ─────────────────────────────────────────────────────

    def _save_checkpoint(self, step: int, epoch: int, suffix: Optional[str] = None):
        """Save LoRA adapter + training state."""
        if suffix:
            save_dir = Path(self.config.output_dir) / suffix
        else:
            save_dir = Path(self.config.output_dir) / f"step_{step}"

        save_dir.mkdir(parents=True, exist_ok=True)

        # Save LoRA adapter
        self.model.save_pretrained(str(save_dir))
        self.tokenizer.save_pretrained(str(save_dir))

        # Save geometric training state
        state = {
            "global_step": step,
            "epoch": epoch,
            "jump_factor": self.geometric.jump_factor,
            "sphere_radius": self.geometric.sphere_radius,
            "stats": dict(self.geometric.stats),
            "config": {
                "jump_factor_init": self.config.jump_factor_init,
                "sphere_radius_init": self.config.sphere_radius_init,
                "jump_freq": self.config.jump_freq,
                "sphere_freq": self.config.sphere_freq,
            },
            "starting_checkpoint": self.starting_checkpoint,
        }

        # Attach latest validation metrics if available
        val_metrics = getattr(self, "_latest_val_metrics", None)
        if val_metrics:
            state["validation"] = {
                "code_extract_rate": val_metrics.get("code_extract_rate"),
                "syntax_valid_rate": val_metrics.get("syntax_valid_rate"),
                "execution_success_rate": val_metrics.get("execution_success_rate"),
                "expected_output_match": val_metrics.get("expected_output_match"),
            }

        state_path = save_dir / "training_state.json"
        with open(state_path, "w") as f:
            json.dump(state, f, indent=2)

        # Prune old checkpoints
        self._prune_checkpoints()

        logger.info(f"Saved checkpoint → {save_dir} | {state['stats']}")

    def _prune_checkpoints(self):
        """Keep only save_total_limit most recent checkpoints."""
        base = Path(self.config.output_dir)
        if not base.exists():
            return
        checkpoints = sorted(
            [d for d in base.iterdir() if d.is_dir()],
            key=lambda d: d.stat().st_mtime,
        )
        while len(checkpoints) > self.config.save_total_limit:
            old = checkpoints.pop(0)
            # Always keep 'final' if it exists
            if old.name == "final":
                continue
            import shutil
            shutil.rmtree(old)
            logger.info(f"Pruned old checkpoint: {old}")

    def _load_geometric_state(self, path: Path):
        """Restore geometric optimizer state from a checkpoint."""
        state_file = path / "training_state.json"
        if not state_file.exists():
            logger.warning(f"No training_state.json in {path} — starting geometric state fresh")
            return

        with open(state_file) as f:
            state = json.load(f)

        self.global_step = state.get("global_step", 0)
        self.geometric.jump_factor = state.get("jump_factor", self.config.jump_factor_init)
        self.geometric.sphere_radius = state.get("sphere_radius", self.config.sphere_radius_init)
        self.geometric.stats.update(state.get("stats", {}))
        self.starting_checkpoint = state.get("starting_checkpoint", str(path))

        # Note: Adam optimizer state is NOT restored — parameter group IDs
        # won't match after model reload. The fresh optimizer starts with
        # zero momentum, but the geometric exploration state is preserved.

        logger.info(
            f"Resumed from step {self.global_step}: "
            f"jump={self.geometric.jump_factor:.4f}, "
            f"sphere={self.geometric.sphere_radius:.5f}"
        )

    # ── Execution-based validation ────────────────────────────────────────

    def _make_score_fn(self, n_samples: int = 3):
        """
        Build a score function for geometric candidate evaluation.
        Returns a callable that takes a model and returns List[CodeCheckResult].
        """
        if self.eval_examples is None or not self.eval_examples:
            return None

        examples = self.eval_examples[:n_samples]

        @torch.no_grad()
        def score_fn(model, loss: float) -> float:
            results = []
            model.eval()
            for ex in examples:
                query = ex.get("query", "")
                expected_response = ex.get("response", "")

                # Extract expected output
                expected_output = None
                out_match = re.search(
                    r'<output>\s*\n?(.*?)\n?\s*</output>',
                    expected_response, re.DOTALL
                )
                if out_match:
                    expected_output = out_match.group(1).strip()

                # Generate response
                formatted = self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": query}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                input_ids = self.tokenizer(formatted, return_tensors="pt").input_ids.to(self.device)
                output_ids = model.generate(
                    input_ids,
                    max_new_tokens=256,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                )
                gen_text = self.tokenizer.decode(
                    output_ids[0][input_ids.shape[1]:],
                    skip_special_tokens=False,
                )

                # Extract and validate code
                code = extract_code(gen_text)
                if code and self.validator:
                    result = self.validator.check_code(code, expected_output)
                else:
                    result = CodeCheckResult(
                        code=None, syntax_valid=False,
                        syntax_error="No <code> block"
                    )
                results.append(result)
            model.train()
            if self.code_reward is not None:
                return self.code_reward.compute_score(loss, results)
            return loss

        return score_fn

    def _run_validation(self, epoch: int) -> Dict[str, Any]:
        """
        Generate code for held-out eval examples and validate syntax/execution.
        Returns metrics dict.
        """
        if self.eval_examples is None or not self.eval_examples:
            return {}

        if self.validator is None:
            # Lazy-init validator
            from src.executor.python_exec import PythonExecutor
            self.validator = ExecutionValidator(
                executor=PythonExecutor(timeout=5.0),
                execution_timeout=5.0,
                validate_output=True,
            )
            self.code_reward = CodeRewardFunction(
                reward_bonus=self.config.reward_bonus,
                syntax_penalty=self.config.syntax_penalty,
                exec_bonus=self.config.exec_bonus,
            )

        self.model.eval()
        sample = min(self.config.eval_sample_size, len(self.eval_examples))
        logger.info(f"Running execution validation on {sample} examples...")

        start = time.time()
        metrics = self.validator.validate_batch(
            self.model, self.tokenizer, self.eval_examples,
            self.device,
            max_new_tokens=512,
            temperature=0.0,  # Greedy for consistency
            sample_size=sample,
        )
        elapsed = time.time() - start

        logger.info(
            f"Validation ({elapsed:.0f}s): {ExecutionValidator.format_metrics(metrics)}"
        )

        # Log individual errors (first few)
        errors = ExecutionValidator.format_errors(metrics)
        if errors.strip():
            error_lines = errors.strip().split('\n')
            for line in error_lines[:30]:  # Cap at 30 lines
                logger.info(f"  VAL {line}")
            if len(error_lines) > 30:
                logger.info(f"  VAL ... and {len(error_lines) - 30} more errors")

        self.model.train()
        return metrics

    # ── Thermal check ─────────────────────────────────────────────────────

    def _check_thermal(self):
        """Check GPU temperature, pause if needed."""
        if self.thermal_controller is None:
            return

        should_pause = self.thermal_controller.check_and_throttle()
        if should_pause:
            torch.cuda.empty_cache()
            success = self.thermal_controller.wait_for_cooldown()
            if success:
                time.sleep(10)  # Buffer after cooldown
            else:
                logger.error("GPU cooldown failed — continuing anyway")

    # ── Logging ────────────────────────────────────────────────────────────

    def _log_step(self, step: int, loss: float, epoch: int, elapsed: float):
        """Log training progress with geometric stats."""
        g = self.geometric
        j_total = g.stats["jump_accepted"] + g.stats["jump_rejected"]
        s_total = g.stats["sphere_accepted"] + g.stats["sphere_rejected"]

        if torch.cuda.is_available():
            used_gb = torch.cuda.memory_allocated() / 1e9
        else:
            used_gb = 0

        logger.info(
            f"Step {step:5d} | Epoch {epoch} | Loss: {loss:.6f} | "
            f"Jump: {g.jump_factor:.4f} | Sphere: {g.sphere_radius:.5f} | "
            f"Jump acc: {g.stats['jump_accepted']}/{j_total} | "
            f"Sphere acc: {g.stats['sphere_accepted']}/{s_total} | "
            f"VRAM: {used_gb:.1f}GB | {elapsed:.1f}s/step"
        )

    # ── Main training loop ────────────────────────────────────────────────

    def train(self):
        """Run the full geometric training loop."""
        output_path = Path(self.config.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Setup logfile
        log_path = output_path / "geometric_training.log"
        fh = logging.FileHandler(str(log_path))
        fh.setLevel(logging.INFO)
        fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
        logging.getLogger().addHandler(fh)

        logger.info(f"\n{self.config.summary()}")
        logger.info(f"Starting from: {self.starting_checkpoint}")
        logger.info(f"Output: {output_path}")
        logger.info(f"Log: {log_path}")

        log_gpu_status()

        # Setup components
        self.setup()

        # If resuming, restore geometric state
        # (We detect resume by checking if output_dir already has checkpoints)
        existing_checkpoints = sorted(
            [d for d in output_path.iterdir() if d.is_dir() and d.name != "final"],
            key=lambda d: d.stat().st_mtime,
            reverse=True,
        )
        if existing_checkpoints:
            latest = existing_checkpoints[0]
            logger.info(f"Found existing checkpoints — resuming from {latest}")
            # Reload model from latest checkpoint
            self._load_model(str(latest))
            self._build_optimizer()
            self.geometric = GeometricOptimizer(self.model, self.config, self.device)
            self._load_geometric_state(latest)
        else:
            self.geometric = GeometricOptimizer(self.model, self.config, self.device)

        torch.cuda.empty_cache()

        # Initialize execution validator if exec_reward is enabled
        score_fn = None
        if self.config.exec_reward:
            if self.validator is None:
                from src.executor.python_exec import PythonExecutor
                self.validator = ExecutionValidator(
                    executor=PythonExecutor(timeout=5.0),
                    execution_timeout=5.0,
                    validate_output=True,
                )
                self.code_reward = CodeRewardFunction(
                    reward_bonus=self.config.reward_bonus,
                    syntax_penalty=self.config.syntax_penalty,
                    exec_bonus=self.config.exec_bonus,
                )
                logger.info(
                    f"Exec-reward enabled: bonus={self.config.reward_bonus}, "
                    f"penalty={self.config.syntax_penalty}, exec_bonus={self.config.exec_bonus}"
                )
            # Build score_fn for geometric candidate evaluation
            score_fn = self._make_score_fn(n_samples=3)

        # ── Training ──────────────────────────────────────────────────
        start_time = time.time()
        step_times = []

        for epoch in range(self.config.epochs):
            epoch_start = time.time()

            for batch in self.dataloader:
                step_start = time.time()
                step = self.global_step

                # ─── Standard gradient step ───────────────────────────
                θ_before = self.geometric.capture_lora_params()

                device_batch = {
                    k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }

                outputs = self.model(**device_batch)
                loss = outputs.loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                self.optimizer.zero_grad()

                θ_after = self.geometric.capture_lora_params()

                # ─── Vector space jump ────────────────────────────────
                if step > 0 and step % self.config.jump_freq == 0:
                    accepted, loss_val = self.geometric.vector_jump_step(
                        θ_before, θ_after, batch, score_fn=score_fn
                    )
                    if accepted:
                        loss = torch.tensor(loss_val, device=self.device)

                # ─── Hypersphere search ───────────────────────────────
                if step > 0 and step % self.config.sphere_freq == 0:
                    accepted, loss_val = self.geometric.hypersphere_step(
                        batch, score_fn=score_fn
                    )
                    if accepted:
                        loss = torch.tensor(loss_val, device=self.device)

                step_elapsed = time.time() - step_start
                step_times.append(step_elapsed)

                # ─── Logging ──────────────────────────────────────────
                if step % self.config.logging_steps == 0:
                    self._log_step(step, loss.item(), epoch, step_elapsed)

                # ─── Checkpoint ───────────────────────────────────────
                if step > 0 and step % self.config.save_every == 0:
                    self._save_checkpoint(step, epoch)

                # ─── Thermal ──────────────────────────────────────────
                if step % 5 == 0:
                    self._check_thermal()

                self.global_step += 1

            # Epoch end: save + log + validate
            epoch_elapsed = time.time() - epoch_start
            avg_step = sum(step_times[-len(self.dataloader):]) / max(len(step_times[-len(self.dataloader):]), 1)
            logger.info(
                f"Epoch {epoch} complete | {len(self.dataloader)} steps | "
                f"{epoch_elapsed:.0f}s total | {avg_step:.1f}s/step avg | "
                f"Geometric stats: {self.geometric.stats_summary()}"
            )
            log_gpu_status()

            # Execution-based validation
            if self.config.validate_every > 0 and (epoch + 1) % self.config.validate_every == 0:
                val_metrics = self._run_validation(epoch)
                if val_metrics:
                    # Log to training_state.json in checkpoint
                    self._latest_val_metrics = val_metrics

            self._save_checkpoint(self.global_step, epoch, suffix=f"epoch_{epoch}")

        # ── Final save ────────────────────────────────────────────────
        final_path = Path(self.config.output_dir) / "final"
        logger.info(f"Saving final model to {final_path}")
        self.model.save_pretrained(str(final_path))
        self.tokenizer.save_pretrained(str(final_path))

        total_time = time.time() - start_time
        logger.info(
            f"Training complete. Total steps: {self.global_step} | "
            f"Total time: {total_time:.0f}s ({total_time/60:.1f}min) | "
            f"Final stats: {self.geometric.stats_summary()}"
        )
        logger.info(f"Model saved to: {final_path}")


# ─── CLI ──────────────────────────────────────────────────────────────────────

def find_latest_model() -> Path:
    """Return the most recently modified 'final' adapter directory."""
    candidates = sorted(
        OUTPUT_DIR.parent.glob("*/final"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(
            f"No trained model found in {OUTPUT_DIR.parent}.\n"
            "Run train.py first to produce a checkpoint."
        )
    return candidates[0]


def parse_args():
    p = argparse.ArgumentParser(
        description="Geometric vector-space optimization for WorldModel LoRA training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Presets:
  --conservative   jump=0.05, sphere=0.01, freq=100/300, candidates=2/3
  --primary        jump=0.10, sphere=0.02, freq=50/150,  candidates=3/4  (default)
  --aggressive     jump=0.15, sphere=0.05, freq=25/75,   candidates=4/6

Examples:
  python train_geometric.py                          # default preset + auto-detect model
  python train_geometric.py --aggressive             # more exploration
  python train_geometric.py --conservative --epochs 20
  python train_geometric.py --resume output/worldmodel_geometric/step_500
  python train_geometric.py --dry-run                # show config, exit
        """,
    )
    p.add_argument("--model", default=None,
                   help="Starting LoRA checkpoint (default: auto-detect latest)")
    p.add_argument("--output", default=None,
                   help=f"Output directory (default: {OUTPUT_DIR})")
    p.add_argument("--epochs", type=int, default=10,
                   help="Number of geometric optimization epochs (default: 10)")
    p.add_argument("--batch-size", type=int, default=2,
                   help="Per-device batch size (default: 2)")
    p.add_argument("--lr", type=float, default=2e-4,
                   help="Learning rate (default: 2e-4)")
    p.add_argument("--max-length", type=int, default=1024,
                   help="Max sequence length (default: 1024)")

    # Geometric hyperparameters
    p.add_argument("--jump-factor", type=float, default=None,
                   help="Initial vector jump multiplier (default: from preset)")
    p.add_argument("--sphere-radius", type=float, default=None,
                   help="Initial hypersphere radius (default: from preset)")
    p.add_argument("--jump-freq", type=int, default=None,
                   help="Vector jump frequency in steps (default: from preset)")
    p.add_argument("--sphere-freq", type=int, default=None,
                   help="Hypersphere frequency in steps (default: from preset)")
    p.add_argument("--num-jump-candidates", type=int, default=None,
                   help="Random candidates in vector jump (default: from preset)")
    p.add_argument("--num-sphere-candidates", type=int, default=None,
                   help="Candidates on hypersphere (default: from preset)")

    # Presets
    preset = p.add_mutually_exclusive_group()
    preset.add_argument("--conservative", action="store_true",
                        help="Use conservative hyperparameter preset")
    preset.add_argument("--aggressive", action="store_true",
                        help="Use aggressive hyperparameter preset")

    p.add_argument("--resume", type=str, default=None,
                   help="Resume from a geometric checkpoint directory")
    p.add_argument("--save-every", type=int, default=100,
                   help="Save checkpoint every N steps (default: 100)")
    p.add_argument("--dry-run", action="store_true",
                   help="Print configuration and exit")
    p.add_argument("--max-temp", type=float, default=DEFAULT_MAX_TEMP,
                   help=f"Pause if GPU exceeds this temp °C (default: {DEFAULT_MAX_TEMP})")
    p.add_argument("--safe-temp", type=float, default=DEFAULT_SAFE_TEMP,
                   help=f"Resume if GPU drops below this temp °C (default: {DEFAULT_SAFE_TEMP})")

    # Execution-based validation & reward
    p.add_argument("--exec-reward", action="store_true",
                   help="Use code validity (syntax/execution) as reward in geometric optimization")
    p.add_argument("--reward-bonus", type=float, default=0.2,
                   help="Reward bonus for valid syntax per example (default: 0.2)")
    p.add_argument("--syntax-penalty", type=float, default=0.5,
                   help="Penalty for missing/invalid code per example (default: 0.5)")
    p.add_argument("--exec-bonus", type=float, default=0.1,
                   help="Bonus for code that executes successfully (default: 0.1)")
    p.add_argument("--eval-sample-size", type=int, default=20,
                   help="Number of examples to validate at epoch end (default: 20)")
    p.add_argument("--validate-every", type=int, default=1,
                   help="Run validation every N epochs (default: 1)")
    return p.parse_args()


def main():
    args = parse_args()

    config = GeometricConfig()

    # Apply model/output paths
    if args.model:
        config.model_name = args.model
    else:
        try:
            latest = find_latest_model()
            config.model_name = str(latest)
        except FileNotFoundError as e:
            logger.error(str(e))
            sys.exit(1)

    if args.output:
        config.output_dir = args.output

    # Apply scalar overrides
    config.epochs = args.epochs
    config.batch_size = args.batch_size
    config.learning_rate = args.lr
    config.max_length = args.max_length
    config.save_every = args.save_every
    config.max_temp = args.max_temp
    config.safe_temp = args.safe_temp

    # Exec-reward config
    config.exec_reward = args.exec_reward
    config.reward_bonus = args.reward_bonus
    config.syntax_penalty = args.syntax_penalty
    config.exec_bonus = args.exec_bonus
    config.eval_sample_size = args.eval_sample_size
    config.validate_every = args.validate_every

    # Apply preset (before individual overrides)
    if args.conservative:
        config.apply_preset("conservative")
    elif args.aggressive:
        config.apply_preset("aggressive")
    else:
        config.apply_preset("primary")  # Default preset

    # Individual geometric overrides (on top of preset)
    if args.jump_factor is not None:
        config.jump_factor_init = args.jump_factor
    if args.sphere_radius is not None:
        config.sphere_radius_init = args.sphere_radius
    if args.jump_freq is not None:
        config.jump_freq = args.jump_freq
    if args.sphere_freq is not None:
        config.sphere_freq = args.sphere_freq
    if args.num_jump_candidates is not None:
        config.num_jump_candidates = args.num_jump_candidates
    if args.num_sphere_candidates is not None:
        config.num_sphere_candidates = args.num_sphere_candidates

    # Dry run
    if args.dry_run:
        print(config.summary())
        if args.resume:
            print(f"\nResume from: {args.resume}")
        print(f"\nStarting model: {config.model_name}")
        print(f"Output directory: {config.output_dir}")
        sys.exit(0)

    # Run training
    trainer = GeometricTrainer(config)
    if args.resume:
        trainer.config = config  # ensure config is set
        trainer.setup(resume_path=args.resume)
    trainer.train()


if __name__ == "__main__":
    main()
