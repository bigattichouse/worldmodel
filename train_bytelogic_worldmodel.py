#!/usr/bin/env python3
"""
ByteLogic WorldModel Training
============================

Trains the worldmodel LLM using ByteLogic-only training data.
Replaces WAT-based computation with ByteLogic structured reasoning.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, TrainingArguments, 
    Trainer, TrainerCallback, DataCollatorForLanguageModeling
)
import json
import time
from pathlib import Path
from typing import List, Dict, Optional
import logging
import argparse

# ByteLogic components
from src.training.bytelogic_dataset_new import load_bytelogic_dataset

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ROCm Detection
print("=== ByteLogic WorldModel Training ===")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    device = torch.device("cuda")
    gpu_name = torch.cuda.get_device_name()
    total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"✅ GPU: {gpu_name}")
    print(f"   Memory: {total_memory:.1f}GB")
    torch.cuda.empty_cache()
else:
    device = torch.device("cpu")
    print("⚠️ No GPU detected, using CPU")


class ByteLogicTrainingCallback(TrainerCallback):
    """Callback for monitoring ByteLogic training progress."""
    
    def __init__(self, save_frequency: int = 500):
        self.save_frequency = save_frequency
        self.step_times = []
        self.start_time = time.time()
    
    def on_step_end(self, args, state, control, **kwargs):
        """Monitor training progress."""
        current_time = time.time()
        if len(self.step_times) > 0:
            step_time = current_time - self.step_times[-1]
            self.step_times.append(current_time)
            
            # Log progress every 50 steps
            if state.global_step % 50 == 0:
                elapsed = current_time - self.start_time
                steps_per_sec = state.global_step / elapsed if elapsed > 0 else 0
                
                logger.info(f"Step {state.global_step}: "
                          f"{steps_per_sec:.2f} steps/sec, "
                          f"recent step: {step_time:.2f}s")
        else:
            self.step_times.append(current_time)
    
    def on_save(self, args, state, control, **kwargs):
        """Handle model saving."""
        logger.info(f"💾 Saved checkpoint at step {state.global_step}")


def setup_model_and_tokenizer(model_path: str, max_length: int = 1024):
    """Setup model and tokenizer with ByteLogic support."""
    logger.info(f"Loading base model from {model_path}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True
    )
    
    # Resize token embeddings if new tokens were added
    original_vocab_size = len(tokenizer)
    
    # Add ByteLogic tokens through dataset loading (this will modify tokenizer)
    dummy_dataset = load_bytelogic_dataset(
        data_file="training/datasets/bytelogic_train_comprehensive.jsonl",
        tokenizer=tokenizer,
        max_length=max_length
    )
    
    # Resize model embeddings if tokenizer vocabulary expanded
    new_vocab_size = len(tokenizer)
    if new_vocab_size > original_vocab_size:
        model.resize_token_embeddings(new_vocab_size)
        logger.info(f"Resized token embeddings from {original_vocab_size} to {new_vocab_size}")
    
    return model, tokenizer


def create_training_arguments(output_dir: str, epochs: int, learning_rate: float = 2e-5):
    """Create training arguments optimized for ByteLogic training."""
    return TrainingArguments(
        output_dir=output_dir,
        
        # Training parameters
        num_train_epochs=epochs,
        per_device_train_batch_size=2,  # Conservative for large models
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=8,  # Effective batch size = 2*8 = 16
        learning_rate=learning_rate,
        weight_decay=0.01,
        
        # Learning rate scheduling
        lr_scheduler_type="cosine",
        warmup_steps=100,
        
        # Evaluation and saving
        eval_strategy="steps",
        eval_steps=200,
        save_strategy="steps", 
        save_steps=400,  # Multiple of eval_steps (200)
        save_total_limit=5,
        
        # Optimization
        bf16=True if torch.cuda.is_available() else False,  # Use bfloat16 if available
        fp16=False,  # Don't use fp16 with bf16
        dataloader_num_workers=2,  # Reduced to prevent file descriptor leaks
        dataloader_persistent_workers=False,  # Disable to prevent leaks
        
        # Logging
        logging_steps=50,
        report_to=None,  # Disable wandb/tensorboard for now
        
        # Memory optimization
        gradient_checkpointing=True,
        
        # Other
        remove_unused_columns=False,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        
        # Seed for reproducibility
        seed=42
    )


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="ByteLogic WorldModel Training")
    parser.add_argument("--model", default="../model/Qwen3-0.6B", 
                       help="Path to base model")
    parser.add_argument("--dataset", default="training/datasets/comprehensive_bytelogic_dataset.json",
                       help="Path to ByteLogic training dataset")
    parser.add_argument("--output-dir", default="bytelogic_worldmodel_output",
                       help="Output directory for trained model")
    parser.add_argument("--epochs", type=int, default=5,
                       help="Number of training epochs")
    parser.add_argument("--learning-rate", type=float, default=2e-5,
                       help="Learning rate")
    parser.add_argument("--max-length", type=int, default=1024,
                       help="Maximum sequence length")
    parser.add_argument("--curriculum", default="all",
                       choices=["basic", "intermediate", "advanced", "all"],
                       help="Curriculum learning stage")
    
    args = parser.parse_args()
    
    logger.info("🚀 Starting ByteLogic WorldModel Training")
    logger.info(f"   Model: {args.model}")
    logger.info(f"   Dataset: {args.dataset}")
    logger.info(f"   Output: {args.output_dir}")
    logger.info(f"   Epochs: {args.epochs}")
    logger.info(f"   Curriculum: {args.curriculum}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    try:
        # Setup model and tokenizer
        logger.info("📚 Setting up model and tokenizer...")
        model, tokenizer = setup_model_and_tokenizer(args.model, args.max_length)
        
        # Load datasets
        logger.info("📖 Loading training datasets...")
        train_dataset = load_bytelogic_dataset(
            data_file=args.dataset,
            tokenizer=tokenizer,
            max_length=args.max_length,
            curriculum_stage=args.curriculum,
            validation_mode=False
        )
        
        eval_dataset = load_bytelogic_dataset(
            data_file=args.dataset,
            tokenizer=tokenizer,
            max_length=args.max_length,
            curriculum_stage=args.curriculum,
            validation_mode=True
        )
        
        logger.info(f"   Training examples: {len(train_dataset)}")
        logger.info(f"   Validation examples: {len(eval_dataset)}")
        
        # Create data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False  # Causal language modeling
        )
        
        # Create training arguments
        training_args = create_training_arguments(
            output_dir=args.output_dir,
            epochs=args.epochs,
            learning_rate=args.learning_rate
        )
        
        # Create trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            callbacks=[ByteLogicTrainingCallback()]
        )
        
        # Save training configuration
        config = {
            "model_path": args.model,
            "dataset_path": args.dataset,
            "curriculum_stage": args.curriculum,
            "training_args": training_args.to_dict(),
            "train_examples": len(train_dataset),
            "eval_examples": len(eval_dataset),
            "vocab_size": len(tokenizer)
        }
        
        config_file = os.path.join(args.output_dir, "training_config.json")
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"📁 Saved training config to {config_file}")
        
        # Start training
        logger.info("🏋️ Starting training...")
        trainer.train()
        
        # Save final model
        logger.info("💾 Saving final model...")
        final_model_path = os.path.join(args.output_dir, "final_model")
        trainer.save_model(final_model_path)
        tokenizer.save_pretrained(final_model_path)
        
        # Save training summary
        summary = {
            "status": "completed",
            "final_model_path": final_model_path,
            "training_time": time.time() - trainer.state.log_history[0]['train_runtime'] if trainer.state.log_history else "unknown",
            "total_steps": trainer.state.global_step,
            "final_loss": trainer.state.log_history[-1].get('train_loss') if trainer.state.log_history else "unknown"
        }
        
        summary_file = os.path.join(args.output_dir, "training_summary.json")
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"✅ Training completed successfully!")
        logger.info(f"   Final model: {final_model_path}")
        logger.info(f"   Training summary: {summary_file}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)