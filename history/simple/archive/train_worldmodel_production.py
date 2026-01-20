#!/usr/bin/env python3
"""
WorldModel Production Training Configuration
===========================================

Production-grade training settings for WorldModel LLMs:
- Much higher epochs for proper learning
- Larger effective batch sizes
- Better evaluation and monitoring
- Learning rate scheduling
- Early stopping
- Comprehensive metrics
"""

import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, TrainingArguments, 
    Trainer, DataCollatorForLanguageModeling, TrainerCallback,
    get_linear_schedule_with_warmup
)
import json
import os
import time
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass
from torch.utils.data import Dataset
import logging
import numpy as np

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ROCm Detection
print("=== WorldModel Production Training ===")
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

print("=" * 50)

@dataclass
class WorldModelExample:
    user_input: str
    assistant_output: str
    
    @classmethod
    def from_training_text(cls, text_block: str):
        """Parse training data text block."""
        lines = text_block.strip().split('\n')
        user_input = ""
        assistant_output = ""
        
        current_section = None
        
        for line in lines:
            line = line.strip()
            if line.startswith("User:"):
                user_input = line[5:].strip()
                current_section = "user"
            elif line.startswith("Assistant:"):
                assistant_output = line[10:].strip()
                current_section = "assistant"
            elif current_section == "assistant" and line:
                assistant_output += "\n" + line
        
        return cls(user_input=user_input, assistant_output=assistant_output)

class WorldModelDataset(Dataset):
    """Dataset for WorldModel training with proper formatting."""
    
    def __init__(self, examples: List[WorldModelExample], tokenizer: AutoTokenizer, 
                 max_length: int = 768):  # Increased context length
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Ensure pad token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        
        print(f"Processing {len(examples)} examples for training...")
        self.processed_examples = self._process_examples()
    
    def _format_conversation(self, example: WorldModelExample, model_name: str) -> str:
        """Format conversation for specific model type."""
        # Detect model type and use appropriate format
        if "qwen" in model_name.lower():
            # Qwen format
            formatted = f"<|im_start|>user\n{example.user_input}<|im_end|>\n<|im_start|>assistant\n{example.assistant_output}<|im_end|>"
        elif "gemma" in model_name.lower():
            # Gemma format  
            formatted = f"<start_of_turn>user\n{example.user_input}<end_of_turn>\n<start_of_turn>model\n{example.assistant_output}<end_of_turn>"
        else:
            # Generic format
            formatted = f"### Human: {example.user_input}\n\n### Assistant: {example.assistant_output}"
        
        return formatted
    
    def _process_examples(self) -> List[Dict[str, torch.Tensor]]:
        """Process examples with correct tokenization."""
        processed = []
        model_name = getattr(self.tokenizer, 'name_or_path', 'unknown')
        
        for i, example in enumerate(self.examples):
            if i % 100 == 0:
                print(f"  Processing {i}/{len(self.examples)}...")
            
            # Format conversation
            formatted_text = self._format_conversation(example, model_name)
            
            # Tokenize
            encoding = self.tokenizer(
                formatted_text,
                truncation=True,
                max_length=self.max_length,
                padding="max_length",
                return_tensors="pt"
            )
            
            input_ids = encoding['input_ids'].squeeze()
            attention_mask = encoding['attention_mask'].squeeze()
            
            # Create labels
            labels = input_ids.clone()
            
            # Mask user tokens (only train on assistant response)
            if "qwen" in model_name.lower():
                user_end_marker = "<|im_start|>assistant\n"
            elif "gemma" in model_name.lower():
                user_end_marker = "<start_of_turn>model\n"
            else:
                user_end_marker = "### Assistant:"
            
            # Find assistant start in the formatted text
            marker_pos = formatted_text.find(user_end_marker)
            if marker_pos > 0:
                user_part = formatted_text[:marker_pos + len(user_end_marker)]
                user_tokens = self.tokenizer.encode(user_part, add_special_tokens=False)
                user_len = min(len(user_tokens), len(labels))
                if user_len > 0:
                    labels[:user_len] = -100
            
            # Mask padding tokens
            labels[attention_mask == 0] = -100
            
            processed.append({
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'labels': labels
            })
        
        print(f"✅ Processed {len(processed)} examples")
        return processed
    
    def __len__(self):
        return len(self.processed_examples)
    
    def __getitem__(self, idx):
        return self.processed_examples[idx]

def load_training_data(data_file: str) -> List[WorldModelExample]:
    """Load training data from file."""
    print(f"Loading training data from: {data_file}")
    
    with open(data_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Split into examples
    raw_examples = content.split('\n\n')
    examples = []
    
    for raw in raw_examples:
        if raw.strip() and 'User:' in raw and 'Assistant:' in raw:
            try:
                example = WorldModelExample.from_training_text(raw)
                if example.user_input and example.assistant_output:
                    examples.append(example)
            except Exception as e:
                logger.warning(f"Failed to parse example: {e}")
                continue
    
    print(f"✅ Loaded {len(examples)} valid examples")
    
    # Show sample
    if examples:
        sample = examples[0]
        print(f"\n📝 Sample example:")
        print(f"   User: {sample.user_input[:80]}...")
        print(f"   Assistant: {sample.assistant_output[:80]}...")
    
    return examples

def setup_model_and_tokenizer(model_name: str):
    """Load model and tokenizer with ROCm optimizations."""
    print(f"Loading model: {model_name}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        padding_side="right"
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Load model with ROCm-friendly settings
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.float32,  # ROCm stability
        device_map={"": 0} if torch.cuda.is_available() else None,
        low_cpu_mem_usage=True,
        use_cache=False,  # Disable for training
        attn_implementation="eager"  # ROCm compatibility
    )
    
    print(f"✅ Model loaded:")
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"   Vocab size: {len(tokenizer)}")
    
    return model, tokenizer

class ProductionTrainingCallback(TrainerCallback):
    """Enhanced callback for production training monitoring."""
    
    def __init__(self):
        self.start_time = time.time()
        self.best_loss = float('inf')
        self.loss_history = []
        self.eval_history = []
        self.lr_history = []
        self.step_times = []
        
    def on_step_begin(self, args, state, control, **kwargs):
        self.step_start_time = time.time()
    
    def on_step_end(self, args, state, control, **kwargs):
        if hasattr(self, 'step_start_time'):
            step_time = time.time() - self.step_start_time
            self.step_times.append(step_time)
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        
        current_time = time.time()
        elapsed = current_time - self.start_time
        
        if 'train_loss' in logs:
            loss = logs['train_loss']
            lr = logs.get('learning_rate', 0)
            step = state.global_step
            epoch = state.epoch
            
            self.loss_history.append(loss)
            self.lr_history.append(lr)
            
            # Calculate ETA
            if len(self.step_times) > 10:
                avg_step_time = np.mean(self.step_times[-10:])
                remaining_steps = state.max_steps - step if state.max_steps > 0 else (args.num_train_epochs * len(state.train_dataloader) - step)
                eta_seconds = avg_step_time * remaining_steps
                eta_str = f"{eta_seconds/3600:.1f}h" if eta_seconds > 3600 else f"{eta_seconds/60:.0f}m"
            else:
                eta_str = "calculating..."
            
            print(f"Step {step:4d} | Epoch {epoch:.2f} | Loss: {loss:.6f} | LR: {lr:.2e} | ETA: {eta_str}")
        
        if 'eval_loss' in logs:
            eval_loss = logs['eval_loss']
            self.eval_history.append(eval_loss)
            
            if eval_loss < self.best_loss:
                self.best_loss = eval_loss
                improvement = "🎯 NEW BEST"
            else:
                improvement = f"(best: {self.best_loss:.6f})"
            
            print(f"📊 Eval Loss: {eval_loss:.6f} {improvement}")
    
    def on_train_end(self, args, state, control, **kwargs):
        total_time = time.time() - self.start_time
        print(f"\n📈 Training Summary:")
        print(f"   Total time: {total_time/3600:.1f}h ({total_time/60:.1f}m)")
        print(f"   Steps completed: {state.global_step}")
        print(f"   Best eval loss: {self.best_loss:.6f}")
        
        if len(self.loss_history) > 10:
            initial_loss = np.mean(self.loss_history[:10])
            final_loss = np.mean(self.loss_history[-10:])
            improvement = (initial_loss - final_loss) / initial_loss * 100
            print(f"   Loss improvement: {improvement:.1f}%")

def evaluate_worldmodel_quality(model, tokenizer, eval_examples):
    """Evaluate how well the model generates WorldModel format."""
    print("\n🧪 Evaluating WorldModel Format Quality...")
    
    test_prompts = [
        "Calculate 25% of 400",
        "Count vowels in 'hello world'",
        "What is 15 × 8?",
        "Find area of circle radius 7"
    ]
    
    model.eval()
    structure_scores = []
    
    for prompt in test_prompts:
        # Format prompt
        formatted = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        
        # Generate
        inputs = tokenizer(formatted, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=200,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        generated = response[len(formatted):].strip()
        
        # Score structure
        has_think = "<think>" in generated and "</think>" in generated
        has_model = "<model>" in generated and "</model>" in generated
        has_requires = "<requires>" in generated and "</requires>" in generated
        
        structure_score = sum([has_think, has_model, has_requires])
        structure_scores.append(structure_score)
        
        print(f"   '{prompt[:30]}...': {structure_score}/3")
    
    avg_structure = np.mean(structure_scores)
    print(f"📊 Average WorldModel structure score: {avg_structure:.2f}/3.0")
    
    return avg_structure

def train_worldmodel_production():
    """Production training function with proper parameters."""
    
    # PRODUCTION Configuration
    MODEL_NAME = "/home/bigattichouse/workspace/model/Qwen3-0.6B"
    DATA_FILE = "/home/bigattichouse/workspace/worldmodel/data/worldmodel_training_1000.txt"
    OUTPUT_DIR = "./worldmodel_production_training"
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load training data
    examples = load_training_data(DATA_FILE)
    if not examples:
        raise ValueError("No training examples loaded!")
    
    print(f"\n📊 Production Training Configuration")
    print(f"   Dataset size: {len(examples)} examples")
    print(f"   Model: {MODEL_NAME}")
    print(f"   Output: {OUTPUT_DIR}")
    
    # Split data
    split_idx = int(0.85 * len(examples))  # 85% train, 15% eval for production
    train_examples = examples[:split_idx]
    eval_examples = examples[split_idx:]
    
    print(f"   Training: {len(train_examples)} examples")
    print(f"   Evaluation: {len(eval_examples)} examples")
    
    # Setup model
    model, tokenizer = setup_model_and_tokenizer(MODEL_NAME)
    
    # Create datasets
    print(f"\n🔄 Creating datasets...")
    train_dataset = WorldModelDataset(train_examples, tokenizer, max_length=768)
    eval_dataset = WorldModelDataset(eval_examples, tokenizer, max_length=768)
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        return_tensors="pt"
    )
    
    # Calculate training steps
    train_steps_per_epoch = len(train_dataset) // (2 * 8)  # batch_size * grad_accumulation
    total_steps = train_steps_per_epoch * 15  # 15 epochs
    warmup_steps = int(0.1 * total_steps)  # 10% warmup
    
    print(f"\n📈 Training Schedule:")
    print(f"   Steps per epoch: {train_steps_per_epoch}")
    print(f"   Total steps: {total_steps}")
    print(f"   Warmup steps: {warmup_steps}")
    
    # PRODUCTION Training Arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=15,  # PRODUCTION: Much more training
        per_device_train_batch_size=2,  # Conservative for MI50 stability
        per_device_eval_batch_size=4,   # Can be larger for eval
        gradient_accumulation_steps=8,  # Effective batch size of 16
        learning_rate=2e-4,  # Slightly higher for more training
        warmup_steps=warmup_steps,
        weight_decay=0.01,
        max_grad_norm=1.0,  # Gradient clipping
        
        # Logging and evaluation
        logging_steps=25,
        eval_steps=100,  # More frequent evaluation
        save_steps=200,
        
        # Strategy
        eval_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        
        # Early stopping patience (stop if no improvement for 5 evaluations)
        # Note: Transformers doesn't have built-in early stopping, we'd need to add custom callback
        
        # System settings
        dataloader_num_workers=0,  # ROCm compatibility
        fp16=False,
        bf16=False,
        gradient_checkpointing=True,
        remove_unused_columns=False,
        dataloader_pin_memory=False,
        
        # Output settings
        report_to=None,
        push_to_hub=False,
        save_total_limit=5,  # Keep more checkpoints for production
        
        # Learning rate scheduling
        lr_scheduler_type="cosine",  # Better than linear for longer training
        
        # Reproducibility
        seed=42,
        data_seed=42,
    )
    
    # Create trainer with enhanced callback
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        callbacks=[ProductionTrainingCallback()]
    )
    
    print(f"\n🚀 Starting PRODUCTION training...")
    print(f"   Device: {device}")
    print(f"   Epochs: {training_args.num_train_epochs}")
    print(f"   Effective batch size: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
    print(f"   Learning rate: {training_args.learning_rate}")
    print(f"   Total steps: {total_steps}")
    print(f"   Expected time: ~{total_steps * 7 / 3600:.1f} hours")
    
    start_time = time.time()
    
    # Train the model
    train_result = trainer.train()
    
    training_time = time.time() - start_time
    
    print(f"\n🎉 Production training completed!")
    print(f"   Final training loss: {train_result.training_loss:.6f}")
    print(f"   Training time: {training_time/3600:.1f} hours")
    print(f"   Steps completed: {train_result.global_step}")
    
    # Evaluate WorldModel quality
    structure_quality = evaluate_worldmodel_quality(model, tokenizer, eval_examples[:10])
    
    # Save final model
    final_save_path = Path(OUTPUT_DIR) / "final_model"
    trainer.save_model(str(final_save_path))
    tokenizer.save_pretrained(str(final_save_path))
    
    # Save training metadata
    metadata = {
        "training_time_hours": training_time / 3600,
        "final_loss": train_result.training_loss,
        "total_steps": train_result.global_step,
        "structure_quality": structure_quality,
        "training_args": training_args.to_dict(),
        "dataset_size": len(examples),
        "model_name": MODEL_NAME
    }
    
    with open(final_save_path / "training_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✅ Model saved to: {final_save_path}")
    print(f"📊 WorldModel structure quality: {structure_quality:.2f}/3.0")
    
    return train_result, final_save_path

if __name__ == "__main__":
    print("🔥 WorldModel Production Training")
    print("=" * 60)
    
    try:
        # Source ROCm environment variables
        os.environ['HSA_OVERRIDE_GFX_VERSION'] = '9.0.6'
        os.environ['HIP_VISIBLE_DEVICES'] = '0'
        os.environ['ROCBLAS_LAYER'] = '0'
        os.environ['PYTORCH_HIP_ALLOC_CONF'] = 'max_split_size_mb:128'
        
        result, model_path = train_worldmodel_production()
        
        print(f"\n🎊 PRODUCTION TRAINING SUCCESS!")
        print(f"   Model: {model_path}")
        print(f"   Final loss: {result.training_loss:.6f}")
        print(f"\n💡 To use the production model:")
        print(f"   python3 run_worldmodel_inference.py --model {model_path} --interactive")
        
    except Exception as e:
        print(f"❌ Production training failed: {e}")
        import traceback
        traceback.print_exc()