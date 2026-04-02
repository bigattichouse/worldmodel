#!/usr/bin/env python3
"""
WorldModel Training Script
===========================
Fine-tunes Qwen3-1.7B with LoRA on the think/model/code/output dataset.

Validated for AMD MI50 (gfx906) with ROCm 7.1.1 + PyTorch 2.4.1+rocm6.0.
See docs/rocm/ROCm_Training_Success_Guide.md for environment setup.

Usage:
    # Set ROCm environment first:
    export HSA_OVERRIDE_GFX_VERSION=9.0.6
    export PYTORCH_ROCM_ARCH=gfx906
    export TOKENIZERS_PARALLELISM=false
    export HIP_VISIBLE_DEVICES=0
    export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True,max_split_size_mb:512
    export OMP_NUM_THREADS=1
    export HSA_DISABLE_CACHE=1

    source venv/bin/activate
    python train.py [options]

Options:
    --model         Base model (default: Qwen/Qwen3-1.7B)
    --output        Output directory (default: ./output/worldmodel)
    --epochs        Training epochs (default: 10)
    --batch-size    Per-device batch size (default: 2)
    --grad-accum    Gradient accumulation steps (default: 8)
    --lr            Learning rate (default: 2e-4)
    --max-length    Max sequence length (default: 1024)
    --lora-rank     LoRA rank (default: 16)
    --lora-alpha    LoRA alpha (default: 32)
    --categories    Comma-separated dataset categories to load (default: all)
    --test-split    Fraction held out for evaluation (default: 0.05)
    --resume        Path to checkpoint to resume from
"""

import os
import sys
import json
import logging
import argparse
import random
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    TrainerCallback,
    TrainerState,
    TrainerControl,
    EarlyStoppingCallback,
)
from peft import LoraConfig, get_peft_model, TaskType

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))
from src.training.dataset import load_all_datasets, WorldModelDataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ─── Configuration ────────────────────────────────────────────────────────────

@dataclass
class TrainConfig:
    # Model
    model_name: str = os.path.expanduser("~/workspace/model/Qwen3-1.7B")
    output_dir: str = "./output/worldmodel"

    # Training
    epochs: int = 10
    batch_size: int = 2           # per-device; MI50 32GB can handle 2 for 1.7B
    grad_accum: int = 8           # effective batch = batch_size * grad_accum = 16
    learning_rate: float = 2e-4
    lr_scheduler: str = "cosine"
    warmup_ratio: float = 0.10
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0

    # Sequence
    max_length: int = 1024

    # LoRA
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: tuple = (
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    )

    # Data
    categories: Optional[List[str]] = None   # None = load all
    test_split: float = 0.05

    # ROCm: no quantization, no fp16/bf16 (not stable on gfx906)
    fp16: bool = False
    bf16: bool = False

    # Checkpointing
    save_steps: int = 200
    eval_steps: int = 200
    logging_steps: int = 20
    save_total_limit: int = 3
    load_best_model_at_end: bool = True
    resume_from_checkpoint: Optional[str] = None


# ─── Special tokens ───────────────────────────────────────────────────────────

SPECIAL_TOKENS = [
    "<think>", "</think>",
    "<model>", "</model>",
    "<code>", "</code>",
    "<output>", "</output>",
]


# ─── Setup ────────────────────────────────────────────────────────────────────

def setup_environment():
    """Verify ROCm environment and log GPU info."""
    print("\n=== WorldModel Training ===")
    print(f"PyTorch: {torch.__version__}")
    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU: {gpu}  VRAM: {vram:.1f}GB")
        torch.cuda.empty_cache()
    else:
        print("WARNING: No CUDA/ROCm device found — training on CPU will be very slow")
    print()


def load_tokenizer_and_model(cfg: TrainConfig):
    """Load tokenizer with special tokens, then load and LoRA-wrap the model."""
    logger.info(f"Loading tokenizer: {cfg.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, trust_remote_code=True)

    # Add our special tokens
    new_tokens = [t for t in SPECIAL_TOKENS if t not in tokenizer.get_vocab()]
    if new_tokens:
        tokenizer.add_special_tokens({"additional_special_tokens": new_tokens})
        logger.info(f"Added {len(new_tokens)} special tokens: {new_tokens}")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("Set pad_token = eos_token")

    logger.info(f"Loading model: {cfg.model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        torch_dtype=torch.float32,   # float32 required for ROCm gfx906 stability
        device_map="auto",
        trust_remote_code=True,
    )

    # Resize embeddings if we added tokens
    if new_tokens:
        model.resize_token_embeddings(len(tokenizer))
        logger.info("Resized token embeddings")

    # Apply LoRA
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=cfg.lora_rank,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=list(cfg.lora_target_modules),
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return tokenizer, model


# ─── Dataset ──────────────────────────────────────────────────────────────────

def build_datasets(cfg: TrainConfig, tokenizer):
    """Load JSONL data, split train/eval, return WorldModelDataset instances."""
    all_examples = load_all_datasets(categories=cfg.categories)

    if not all_examples:
        raise ValueError(
            "No training examples found! "
            "Run training/scripts/generate_*.py first to populate training/datasets/"
        )

    # Shuffle deterministically
    random.seed(42)
    random.shuffle(all_examples)

    split = max(1, int(len(all_examples) * cfg.test_split))
    eval_examples = all_examples[:split]
    train_examples = all_examples[split:]

    logger.info(f"Train: {len(train_examples)}, Eval: {len(eval_examples)}")

    train_dataset = WorldModelDataset(train_examples, tokenizer, cfg.max_length)
    eval_dataset = WorldModelDataset(eval_examples, tokenizer, cfg.max_length)

    return train_dataset, eval_dataset


# ─── Callbacks ────────────────────────────────────────────────────────────────

class ProgressCallback(TrainerCallback):
    """Log epoch progress and memory usage."""

    def on_epoch_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        if torch.cuda.is_available():
            used = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            logger.info(
                f"Epoch {state.epoch:.1f} complete | "
                f"Loss: {state.log_history[-1].get('loss', '?'):.4f} | "
                f"GPU: {used:.1f}GB used / {reserved:.1f}GB reserved"
            )

    def on_log(self, args, state: TrainerState, control: TrainerControl, logs=None, **kwargs):
        if logs and "loss" in logs:
            loss = logs["loss"]
            step = state.global_step
            if step % (args.logging_steps * 5) == 0:
                logger.info(f"Step {step} | loss={loss:.4f}")


# ─── Training ─────────────────────────────────────────────────────────────────

def train(cfg: TrainConfig):
    setup_environment()

    tokenizer, model = load_tokenizer_and_model(cfg)
    train_dataset, eval_dataset = build_datasets(cfg, tokenizer)

    training_args = TrainingArguments(
        output_dir=cfg.output_dir,
        num_train_epochs=cfg.epochs,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.grad_accum,
        learning_rate=cfg.learning_rate,
        lr_scheduler_type=cfg.lr_scheduler,
        warmup_ratio=cfg.warmup_ratio,
        weight_decay=cfg.weight_decay,
        max_grad_norm=cfg.max_grad_norm,
        fp16=cfg.fp16,
        bf16=cfg.bf16,
        logging_steps=cfg.logging_steps,
        eval_strategy="steps",
        eval_steps=cfg.eval_steps,
        save_strategy="steps",
        save_steps=cfg.save_steps,
        save_total_limit=cfg.save_total_limit,
        load_best_model_at_end=cfg.load_best_model_at_end,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        dataloader_num_workers=0,     # 0 avoids multiprocess issues on ROCm
        remove_unused_columns=False,
        report_to="none",             # disable wandb/tensorboard by default
        seed=42,
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        callbacks=[
            ProgressCallback(),
            EarlyStoppingCallback(early_stopping_patience=3),
        ],
    )

    logger.info("Starting training...")
    trainer.train(resume_from_checkpoint=cfg.resume_from_checkpoint)

    # Save final model
    final_path = Path(cfg.output_dir) / "final"
    logger.info(f"Saving final model to {final_path}")
    trainer.save_model(str(final_path))
    tokenizer.save_pretrained(str(final_path))

    logger.info("Training complete.")
    return trainer


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Train WorldModel (think/model/code/output)")
    p.add_argument("--model", default=os.path.expanduser("~/workspace/model/Qwen3-1.7B"))
    p.add_argument("--output", default="./output/worldmodel")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--max-length", type=int, default=1024)
    p.add_argument("--lora-rank", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--categories", type=str, default=None,
                   help="Comma-separated list of dataset categories to include")
    p.add_argument("--test-split", type=float, default=0.05)
    p.add_argument("--resume", type=str, default=None)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    cfg = TrainConfig(
        model_name=args.model,
        output_dir=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        learning_rate=args.lr,
        max_length=args.max_length,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        categories=args.categories.split(",") if args.categories else None,
        test_split=args.test_split,
        resume_from_checkpoint=args.resume,
    )

    train(cfg)
