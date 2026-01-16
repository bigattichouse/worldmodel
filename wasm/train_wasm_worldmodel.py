#!/usr/bin/env python3
"""
WASM WorldModel Training for ROCm MI50
======================================

Trains multimodal text+WASM model using the same ROCm-optimized approach
as the successful WorldModel training, but with WASM modal architecture.

Based on train_worldmodel_rocm_stable.py with WASM extensions.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer, TrainingArguments, 
    Trainer, TrainerCallback
)
import json
import time
from pathlib import Path
from typing import List, Dict, Optional
import logging

# WASM components
from src.models.qwen_wasm_adapter import QwenWASMAdapter
from src.tokenization.wat_tokenizer import WATTokenizer
from src.training.wasm_dataset import WASMCurriculumDataset, WASMModalDataset
from src.training.wasm_data_collator import WASMDataCollator

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ROCm Detection (same as stable training)
print("=== WASM WorldModel ROCm Training ===")
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
    print("❌ Using CPU")

print("=" * 60)

# ROCm-stable attention backends (from successful training)
print("🚀 Configuring attention backends for ROCm stability...")
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)  
print("✅ Attention: Using stable SDPA (ROCm-optimized)")

class WASMPerformanceMonitor:
    """Monitor WASM training performance metrics (adapted from original)."""
    
    def __init__(self):
        self.iteration_times = []
        self.memory_usage = []
        self.start_time = None
        self.wasm_execution_times = []
        
    def start_iteration(self):
        self.start_time = time.time()
        
    def end_iteration(self, batch_size, seq_length, wasm_executions=0):
        if self.start_time:
            iteration_time = time.time() - self.start_time
            self.iteration_times.append(iteration_time)
            
            if torch.cuda.is_available():
                memory_gb = torch.cuda.max_memory_allocated() / 1e9
                self.memory_usage.append(memory_gb)
            
            tokens_per_sec = (batch_size * seq_length) / iteration_time
            
            # Only print every 50 iterations to reduce spam (as requested)
            if len(self.iteration_times) % 50 == 0:
                avg_time = sum(self.iteration_times[-50:]) / min(50, len(self.iteration_times))
                speedup = 6.0 / avg_time if avg_time > 0 else 0
                print(f"⚡ Performance avg: {avg_time:.2f}s/it, "
                      f"{tokens_per_sec:.0f} tok/s, "
                      f"mem: {memory_gb:.1f}GB, "
                      f"speedup: {speedup:.1f}x, "
                      f"WASM exec: {wasm_executions}")

class WASMTrainingCallback(TrainerCallback):
    """Custom callback for WASM training progress (adapted from original)."""
    
    def __init__(self):
        self.monitor = WASMPerformanceMonitor()
        
    def on_step_begin(self, args, state, control, **kwargs):
        self.monitor.start_iteration()
        
    def on_step_end(self, args, state, control, **kwargs):
        # Count WASM executions if available
        wasm_executions = 0
        if "logs" in kwargs and "wasm_executions" in kwargs["logs"]:
            wasm_executions = kwargs["logs"]["wasm_executions"]
            
        self.monitor.end_iteration(
            args.per_device_train_batch_size * args.gradient_accumulation_steps,
            400,  # max_length
            wasm_executions
        )

class WASMTrainer(Trainer):
    """Custom trainer for WASM multimodal training."""
    
    def __init__(self, wasm_adapter, *args, **kwargs):
        self.wasm_adapter = wasm_adapter
        super().__init__(*args, **kwargs)
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """Compute loss for both text and WASM streams."""
        # Extract inputs
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"] 
        labels = inputs["labels"]
        
        wasm_input_ids = inputs.get("wasm_input_ids")
        wasm_attention_mask = inputs.get("wasm_attention_mask")
        execution_targets = inputs.get("execution_targets")
        
        # Forward pass through WASM adapter
        outputs = self.wasm_adapter(
            input_ids=input_ids,
            attention_mask=attention_mask,
            wasm_ids=wasm_input_ids,
            wasm_attention_mask=wasm_attention_mask,
            execute_wasm=True
        )
        
        # Text generation loss (primary)
        text_logits = outputs["logits"]
        loss_fct = nn.CrossEntropyLoss()
        
        # Shift labels for causal LM
        shift_logits = text_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        text_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        
        # WASM execution loss (if available)
        execution_loss = 0.0
        wasm_executions = 0
        
        if execution_targets is not None and len(outputs["execution_results"]) > 0:
            # Simple execution accuracy loss
            for i, result in enumerate(outputs["execution_results"]):
                if result["success"] and i < len(execution_targets):
                    predicted = torch.tensor(result["result"], dtype=torch.float32, device=text_loss.device)
                    target = execution_targets[i].to(text_loss.device)
                    execution_loss += nn.functional.mse_loss(predicted, target)
                    wasm_executions += 1
        
        # Combined loss (text is primary, WASM execution is auxiliary)
        total_loss = text_loss + 0.1 * execution_loss  # Weighted combination
        
        # Track WASM executions for monitoring (store in state)
        if not hasattr(self, "_custom_metrics"):
            self._custom_metrics = {}
        self._custom_metrics["wasm_executions"] = wasm_executions
        
        return (total_loss, outputs) if return_outputs else total_loss
    
    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        """Override evaluate to ensure eval_loss is returned."""
        # Call parent evaluate
        metrics = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)
        
        # Ensure eval_loss is included
        if 'eval_loss' not in metrics:
            # Manually compute eval_loss if missing
            eval_dataloader = self.get_eval_dataloader(eval_dataset)
            total_loss = 0.0
            num_samples = 0
            
            self.model.eval()
            with torch.no_grad():
                for batch in eval_dataloader:
                    # Move batch to device
                    batch = {k: v.to(self.args.device) if isinstance(v, torch.Tensor) else v 
                            for k, v in batch.items()}
                    
                    loss = self.compute_loss(self.model, batch, return_outputs=False)
                    total_loss += loss.item()
                    num_samples += 1
            
            if num_samples > 0:
                metrics['eval_loss'] = total_loss / num_samples
            else:
                metrics['eval_loss'] = 0.0
        
        return metrics

def setup_wasm_model_and_tokenizers(model_path: str, use_sandbox: bool = True, sandbox_config: Dict = None):
    """Setup WASM adapter and tokenizers with sandbox configuration."""
    print(f"🔄 Loading WASM model components...")
    
    # Create WASM tokenizer
    wasm_tokenizer = WATTokenizer(vocab_size=8000)
    
    # Create WASM adapter with sandbox settings
    wasm_adapter = QwenWASMAdapter(
        model_path=model_path,
        wasm_vocab_size=8000,
        cross_modal_layers=[3, 7, 11],  # Every 4 layers (Flamingo style)
        freeze_text_layers=False,  # Allow fine-tuning
        use_sandbox=use_sandbox,
        sandbox_config=sandbox_config
    )
    
    # Load text tokenizer
    text_tokenizer = wasm_adapter.text_tokenizer
    
    print(f"✅ Model components loaded:")
    print(f"   Text vocab: {len(text_tokenizer):,}")
    print(f"   WASM vocab: {wasm_tokenizer.vocab_size:,}")
    print(f"   Cross-modal layers: {[3, 7, 11]}")
    print(f"   Sandbox mode: {'Enabled' if use_sandbox else 'Disabled'}")
    
    # Set WASM tokenizer on adapter for token-to-WAT conversion during forward pass
    wasm_adapter.set_wasm_tokenizer(wasm_tokenizer)
    print(f"   WASM execution: Enabled during forward pass")
    
    return wasm_adapter, text_tokenizer, wasm_tokenizer

def main():
    """Main WASM training function."""
    
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description="Train WASM WorldModel")
    parser.add_argument('--model', default="/home/bigattichouse/workspace/model/Qwen3-0.6B",
                       help='Path to base model')
    parser.add_argument('--data', default="./data/converted", 
                       help='Path to training data directory')
    parser.add_argument('--output', default="./wasm_worldmodel_output",
                       help='Output directory for trained model')
    parser.add_argument('--no-sandbox', action='store_true',
                       help='Disable QEMU sandbox for API calls (faster, less secure)')
    parser.add_argument('--sandbox-memory', default='512M',
                       help='Memory allocation for sandbox VM (default: 512M)')
    parser.add_argument('--sandbox-timeout', type=int, default=30,
                       help='Sandbox execution timeout in seconds (default: 30)')
    parser.add_argument('--epochs', type=int, default=10,
                       help='Number of training epochs (default: 10)')
    parser.add_argument('--batch-size', type=int, default=4,
                       help='Training batch size (default: 4)')
    
    args = parser.parse_args()
    
    # Configuration
    MODEL_NAME = args.model
    DATA_DIR = args.data
    OUTPUT_DIR = args.output
    USE_SANDBOX = not args.no_sandbox
    
    # Sandbox configuration
    sandbox_config = {
        'vm_name': 'wasm-worldmodel-training',
        'memory': args.sandbox_memory,
        'timeout': args.sandbox_timeout,
        'persistent': False
    }
    
    if USE_SANDBOX:
        print("🔥 WASM WorldModel Training (Secure Mode)")
        print(f"   Sandbox: VM memory={args.sandbox_memory}, timeout={args.sandbox_timeout}s")
    else:
        print("🔥 WASM WorldModel Training (Direct Mode)")
        print("   WARNING: External API calls will execute directly on host")
    print("=" * 60)
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Setup model and tokenizers with sandbox config
    wasm_adapter, text_tokenizer, wasm_tokenizer = setup_wasm_model_and_tokenizers(
        MODEL_NAME, USE_SANDBOX, sandbox_config
    )
    
    # Load curriculum datasets
    print(f"🔄 Loading training data...")
    curriculum = WASMCurriculumDataset(
        data_dir=DATA_DIR,
        text_tokenizer=text_tokenizer,
        wasm_tokenizer=wasm_tokenizer,
        max_text_length=400,  # Optimized length (from successful training)
        max_wasm_length=256,
        include_execution_results=True
    )
    
    # Start with basic arithmetic (Stage 1)
    train_dataset = curriculum.get_stage_dataset("basic_arithmetic")
    if not train_dataset:
        print("❌ No basic arithmetic dataset found!")
        return
    
    # Create small eval dataset from first stage
    eval_size = min(50, len(train_dataset) // 10)
    eval_dataset = WASMModalDataset.__new__(WASMModalDataset)
    eval_dataset.examples = train_dataset.examples[:eval_size]
    eval_dataset.text_tokenizer = text_tokenizer
    eval_dataset.wasm_tokenizer = wasm_tokenizer
    eval_dataset.max_text_length = 400
    eval_dataset.max_wasm_length = 256
    eval_dataset.include_execution_results = True
    
    # Adjust training dataset
    train_dataset.examples = train_dataset.examples[eval_size:]
    
    print(f"📊 Dataset split:")
    print(f"   Training: {len(train_dataset)} examples")
    print(f"   Evaluation: {len(eval_dataset)} examples")
    
    # Data collator
    data_collator = WASMDataCollator(
        text_tokenizer=text_tokenizer,
        wasm_tokenizer=wasm_tokenizer,
        mlm=False  # Causal LM
    )
    
    # Training arguments optimized for long training runs
    # Adaptive save frequency based on dataset size (conservative for stability)
    save_frequency = max(200, len(train_dataset) // (args.batch_size * 5))  # Save ~5 times per epoch  
    eval_frequency = save_frequency  # Eval as often as saving (not more frequent)
    
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=args.epochs,
        
        # Same optimized batch settings from successful training
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=2,
        
        learning_rate=5e-5,
        warmup_steps=min(100, len(train_dataset) // (args.batch_size * 5)),  # Adaptive warmup
        weight_decay=0.01,
        
        # Robust checkpointing for long training runs
        logging_steps=10,
        eval_steps=eval_frequency,
        save_steps=save_frequency,
        
        eval_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        
        # Long training stability
        save_total_limit=5,  # Keep more checkpoints for long runs
        
        # ROCm optimizations with file descriptor leak prevention
        dataloader_num_workers=2,           # Reduced to prevent fd leaks
        dataloader_pin_memory=True,         # Memory efficiency  
        dataloader_persistent_workers=False, # Disable to prevent fd leaks during evaluation
        
        # Same stable precision settings
        fp16=False,                         # FP32 for stability
        bf16=False,
        
        # Same optimization features
        gradient_checkpointing=False,       # Disabled for speed
        dataloader_drop_last=True,         # Consistent batch sizes
        
        # ROCm compatibility
        report_to=None,                    
        push_to_hub=False,
        remove_unused_columns=False,       # Keep WASM components
    )
    
    # Performance callback
    performance_callback = WASMTrainingCallback()
    
    # Create WASM trainer
    trainer = WASMTrainer(
        wasm_adapter=wasm_adapter,
        model=wasm_adapter,  # Pass adapter as model
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        callbacks=[performance_callback],
    )
    
    # Check for existing checkpoints
    checkpoint_dir = None
    checkpoints = [d for d in os.listdir(OUTPUT_DIR) if d.startswith("checkpoint-")] if os.path.exists(OUTPUT_DIR) else []
    if checkpoints:
        # Find latest checkpoint
        checkpoint_nums = [int(cp.split("-")[-1]) for cp in checkpoints if cp.split("-")[-1].isdigit()]
        if checkpoint_nums:
            latest_checkpoint = f"checkpoint-{max(checkpoint_nums)}"
            checkpoint_dir = os.path.join(OUTPUT_DIR, latest_checkpoint)
            print(f"🔄 Found existing checkpoint: {checkpoint_dir}")
            print("   Training will resume from this checkpoint")
    
    print(f"🚀 Starting WASM training...")
    print(f"   Device: {device}")
    print(f"   Model: Qwen3-0.6B + WASM stream")
    print(f"   Batch size: {training_args.per_device_train_batch_size}")
    print(f"   Effective batch: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
    print(f"   Learning rate: {training_args.learning_rate}")
    print(f"   Epochs: {training_args.num_train_epochs}")
    print(f"   Cross-modal attention: Every 4 layers")
    
    # Memory check (same as successful training)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"💾 Memory status:")
        print(f"   Total VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
        print(f"   Available: {(torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()) / 1e9:.1f}GB")
    
    # Training with error recovery
    start_time = time.time()
    try:
        if checkpoint_dir and os.path.exists(checkpoint_dir):
            print(f"   Resuming from: {checkpoint_dir}")
            trainer.train(resume_from_checkpoint=checkpoint_dir)
        else:
            print(f"   Starting fresh training")
            trainer.train()
            
        training_time = time.time() - start_time
        
    except KeyboardInterrupt:
        print(f"\n⚠️  Training interrupted by user")
        training_time = time.time() - start_time
        print(f"   Partial training time: {training_time/3600:.2f} hours")
        
        # Save current state
        print(f"   Saving current state...")
        trainer.save_model(f"{OUTPUT_DIR}/interrupted_model")
        print(f"   Model saved to: {OUTPUT_DIR}/interrupted_model")
        return
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        training_time = time.time() - start_time
        print(f"   Training time before failure: {training_time/3600:.2f} hours")
        
        # Save current state if possible
        try:
            trainer.save_model(f"{OUTPUT_DIR}/failed_model")
            print(f"   Emergency save completed: {OUTPUT_DIR}/failed_model")
        except:
            print(f"   Emergency save failed")
        raise
    
    print(f"\n✅ WASM training completed in {training_time/3600:.2f} hours")
    if performance_callback.monitor.iteration_times:
        avg_time = sum(performance_callback.monitor.iteration_times) / len(performance_callback.monitor.iteration_times)
        print(f"📊 Average iteration time: {avg_time:.2f}s")
        print(f"🚀 Speedup vs 6.0s baseline: {6.0/avg_time:.1f}x")
    
    # Save the model
    print(f"💾 Saving WASM model to {OUTPUT_DIR}/final_model")
    trainer.save_model(f"{OUTPUT_DIR}/final_model")
    text_tokenizer.save_pretrained(f"{OUTPUT_DIR}/final_model")
    
    # Save WASM tokenizer
    import pickle
    with open(f"{OUTPUT_DIR}/final_model/wasm_tokenizer.pkl", "wb") as f:
        pickle.dump(wasm_tokenizer, f)
    
    # Save comprehensive training metadata
    import json
    metadata = {
        "model_type": "QwenWASMAdapter",
        "base_model": MODEL_NAME,
        "training_data": DATA_DIR,
        "cross_modal_layers": [3, 7, 11],
        "sandbox_enabled": USE_SANDBOX,
        "sandbox_config": sandbox_config if USE_SANDBOX else None,
        "training_args": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": training_args.learning_rate,
            "warmup_steps": training_args.warmup_steps,
            "weight_decay": training_args.weight_decay
        },
        "dataset_info": {
            "train_size": len(train_dataset),
            "eval_size": len(eval_dataset) if eval_dataset else 0,
            "curriculum_stages": ["basic_arithmetic", "system_operations", "complex_logic"]
        },
        "performance": {
            "training_time_hours": training_time/3600,
            "average_iteration_time": avg_time if performance_callback.monitor.iteration_times else None,
            "speedup_vs_baseline": 6.0/avg_time if performance_callback.monitor.iteration_times else None
        },
        "architecture": {
            "modal_type": "text_wasm_cross_attention",
            "execution_engine": "wasmtime",
            "api_integration": "qemu_sandbox" if USE_SANDBOX else "direct"
        },
        "training_date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "version": "1.0.0"
    }
    
    with open(f"{OUTPUT_DIR}/final_model/training_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✅ WASM training completed successfully!")
    print(f"   Model saved to: {OUTPUT_DIR}/final_model")
    print(f"   Training metadata saved to: {OUTPUT_DIR}/final_model/training_metadata.json")
    print(f"   Use: python run_wasm_inference.py --model {OUTPUT_DIR}/final_model")

if __name__ == "__main__":
    main()