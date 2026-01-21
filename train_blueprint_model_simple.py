#!/usr/bin/env python3
"""
Simple BluePrint WorldModel Training Script
==========================================

Streamlined training script for BluePrint methodology using existing src components.
Trains thinking → blueprint pattern only (no WASM/ByteLogic complexity).
"""

import torch
import sys
import argparse
import logging
from pathlib import Path
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments, 
    Trainer,
    DataCollatorForLanguageModeling
)

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from training.blueprint_dataset import load_blueprint_datasets, validate_blueprint_syntax

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Hardware detection
print("=== Simple BluePrint WorldModel Training ===")
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

# ROCm optimizations from proven working script
print("🚀 Configuring attention backends for ROCm stability...")
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)  
print("✅ Attention: Using stable SDPA (ROCm-optimized)")


def setup_model_and_tokenizer(model_path: str):
    """Load and prepare model and tokenizer."""
    logger.info(f"🧠 Loading model from {model_path}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Add pad token if missing
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model - ROCm optimized settings from proven script
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.float32,  # Stable FP32 for ROCm
        low_cpu_mem_usage=True,
        device_map={"": 0}  # Direct device mapping
    )
    
    # Disable gradient checkpointing for speedup
    if hasattr(model, 'gradient_checkpointing_disable'):
        model.gradient_checkpointing_disable()
        logger.info("✅ Gradient checkpointing: DISABLED (30-40% speedup)")
    
    logger.info(f"✅ Model loaded with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    return model, tokenizer


def test_generation(model_path: str, test_query: str = "Design a temperature conversion service"):
    """Test the trained model with a sample query."""
    logger.info(f"🧪 Testing model generation...")
    
    model, tokenizer = setup_model_and_tokenizer(model_path)
    
    # Format input
    input_text = f"User: {test_query}\n\nAssistant: "
    
    # Tokenize and move to device
    inputs = tokenizer(input_text, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_new_tokens=512,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decode
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=False)
    response = generated_text[len(input_text):]
    
    logger.info(f"📋 Test Query: {test_query}")
    logger.info(f"📝 Generated Response:")
    logger.info(f"---")
    logger.info(response)
    logger.info(f"---")
    
    # Validate format
    is_valid, errors = validate_blueprint_syntax(response)
    
    logger.info(f"✅ Validation:")
    logger.info(f"   Format valid: {is_valid}")
    if errors:
        for error in errors:
            logger.info(f"   ❌ {error}")


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Simple BluePrint WorldModel Training")
    parser.add_argument("--model", required=True, help="Path to base model (e.g., Qwen3-0.6B)")
    parser.add_argument("--datasets_dir", default="training/datasets", help="Path to datasets directory")
    parser.add_argument("--output_dir", default="blueprint_model_output", help="Output directory")
    parser.add_argument("--epochs", type=int, default=15, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--max_length", type=int, default=768, help="Maximum sequence length")
    parser.add_argument("--gradient_accumulation", type=int, default=8, help="Gradient accumulation steps")
    parser.add_argument("--curriculum", choices=["foundation", "business", "technical", "domain", "advanced", "security", "complete", "all"], 
                       default="all", help="Curriculum stage (default: all - trains on everything)")
    parser.add_argument("--test_only", action="store_true", help="Only run generation test")
    parser.add_argument("--test_model", type=str, help="Model path for testing")

    args = parser.parse_args()

    if args.test_only:
        test_model_path = args.test_model or args.model
        test_generation(test_model_path)
        return True

    logger.info(f"🎯 Simple BluePrint WorldModel Training")
    logger.info(f"   Model: {args.model}")
    logger.info(f"   Datasets: {args.datasets_dir}")
    logger.info(f"   Output: {args.output_dir}")
    logger.info(f"   Curriculum: {args.curriculum}")

    try:
        # Setup model and tokenizer
        model, tokenizer = setup_model_and_tokenizer(args.model)
        
        # Load datasets
        train_dataset, val_dataset = load_blueprint_datasets(
            datasets_dir=args.datasets_dir,
            tokenizer=tokenizer,
            max_length=args.max_length,
            curriculum_stage=args.curriculum
        )
        
        # Resize token embeddings for special tokens
        model.resize_token_embeddings(len(tokenizer))
        
        # Calculate training parameters
        total_steps = len(train_dataset) // (args.batch_size * args.gradient_accumulation) * args.epochs
        warmup_steps = int(0.1 * total_steps)
        
        # Training arguments - ROCm optimized from proven working script
        training_args = TrainingArguments(
            output_dir=args.output_dir,
            overwrite_output_dir=True,
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation,
            learning_rate=args.learning_rate,
            weight_decay=0.01,
            warmup_steps=warmup_steps,
            lr_scheduler_type="cosine",
            
            # Performance monitoring
            logging_steps=50,
            eval_steps=500,
            save_steps=1000,
            eval_strategy="steps",
            save_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            
            # ROCm optimizations from proven script
            dataloader_num_workers=4,           # Parallel loading
            dataloader_pin_memory=True,         # Memory efficiency  
            dataloader_persistent_workers=True, # Reuse workers
            dataloader_drop_last=True,         # Consistent batch sizes
            
            # STABLE: FP32 training (no precision issues)
            fp16=False,                         # Disabled for ROCm stability
            bf16=False,                         # Disabled for ROCm stability
            
            # Features that speed up training
            gradient_checkpointing=False,       # Disabled for speed
            
            # ROCm compatibility
            report_to=None,                    # Disable wandb/tensorboard
            remove_unused_columns=False,
            max_grad_norm=1.0,
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,
            return_tensors="pt"
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator,
            tokenizer=tokenizer,
        )
        
        # Run training
        logger.info(f"🏋️ Starting training...")
        trainer.train()
        
        # Save final model
        final_model_path = f"{args.output_dir}/final_blueprint_model"
        trainer.save_model(final_model_path)
        tokenizer.save_pretrained(final_model_path)
        
        logger.info(f"🎉 Training completed!")
        logger.info(f"   Final model saved to: {final_model_path}")
        
        # Test the model
        test_generation(final_model_path)
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)