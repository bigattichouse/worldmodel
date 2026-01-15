#!/usr/bin/env python3
"""
Working WorldModel Training for ROCm MI50
=========================================

Full training script that works with the available environment.
Based on successful test results from simple_rocm_test.py

Key fixes implemented:
1. Proper Qwen3 chat template formatting
2. Correct loss masking (no zero loss)
3. ROCm MI50 compatible settings
4. No PEFT dependencies (uses full fine-tuning)
"""

import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, TrainingArguments, 
    Trainer, DataCollatorForLanguageModeling, TrainerCallback
)
import json
import os
import time
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass
from torch.utils.data import Dataset
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ROCm Detection
print("=== WorldModel ROCm Training ===")
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
                 max_length: int = 512):
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
            if i % 50 == 0:
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

class TrainingCallback(TrainerCallback):
    """Custom callback to track training progress."""
    
    def __init__(self):
        self.start_time = time.time()
        self.best_loss = float('inf')
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        
        # Track progress
        current_time = time.time()
        elapsed = current_time - self.start_time
        
        if 'train_loss' in logs:
            loss = logs['train_loss']
            step = state.global_step
            epoch = state.epoch
            
            print(f"Step {step:4d} | Epoch {epoch:.1f} | Loss: {loss:.6f} | Time: {elapsed:.1f}s")
        
        if 'eval_loss' in logs:
            eval_loss = logs['eval_loss']
            if eval_loss < self.best_loss:
                self.best_loss = eval_loss
                print(f"🎯 New best eval loss: {eval_loss:.6f}")

def train_worldmodel():
    """Main training function."""
    
    # Configuration
    MODEL_NAME = "/home/bigattichouse/workspace/model/Qwen3-0.6B"
    DATA_FILE = "/home/bigattichouse/workspace/worldmodel/data/worldmodel_training.txt"
    OUTPUT_DIR = "./worldmodel_rocm_output"
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load training data
    examples = load_training_data(DATA_FILE)
    if not examples:
        raise ValueError("No training examples loaded!")
    
    # Limit data for testing (remove this for full training)
    # examples = examples[:100]  # Comment out for full training
    
    # Split data
    split_idx = int(0.9 * len(examples))
    train_examples = examples[:split_idx]
    eval_examples = examples[split_idx:]
    
    print(f"\n📊 Data split:")
    print(f"   Training: {len(train_examples)} examples")
    print(f"   Evaluation: {len(eval_examples)} examples")
    
    # Setup model
    model, tokenizer = setup_model_and_tokenizer(MODEL_NAME)
    
    # Create datasets
    print(f"\n🔄 Creating datasets...")
    train_dataset = WorldModelDataset(train_examples, tokenizer, max_length=512)
    eval_dataset = WorldModelDataset(eval_examples, tokenizer, max_length=512)
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # Causal LM
        return_tensors="pt"
    )
    
    # Training arguments - Conservative for ROCm MI50
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,  # Start with fewer epochs
        per_device_train_batch_size=2,  # Small batch for MI50
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=4,  # Effective batch size of 8
        learning_rate=1e-4,  # Conservative learning rate
        warmup_steps=100,
        weight_decay=0.01,
        logging_steps=20,
        eval_steps=200,
        save_steps=400,
        eval_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        dataloader_num_workers=0,  # ROCm compatibility
        fp16=False,  # Disable FP16 for stability
        bf16=False,  # Disable BF16
        gradient_checkpointing=True,  # Memory efficiency
        remove_unused_columns=False,
        dataloader_pin_memory=False,  # ROCm compatibility
        report_to=None,  # Disable reporting
        push_to_hub=False,
        save_total_limit=3,  # Keep only 3 checkpoints
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        callbacks=[TrainingCallback()]
    )
    
    print(f"\n🚀 Starting training...")
    print(f"   Device: {device}")
    print(f"   Batch size: {training_args.per_device_train_batch_size}")
    print(f"   Gradient accumulation: {training_args.gradient_accumulation_steps}")
    print(f"   Effective batch size: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
    print(f"   Learning rate: {training_args.learning_rate}")
    print(f"   Epochs: {training_args.num_train_epochs}")
    
    start_time = time.time()
    
    # Train the model
    train_result = trainer.train()
    
    training_time = time.time() - start_time
    
    print(f"\n🎉 Training completed!")
    print(f"   Final training loss: {train_result.training_loss:.6f}")
    print(f"   Training time: {training_time:.1f} seconds")
    print(f"   Steps completed: {train_result.global_step}")
    
    # Save final model
    final_save_path = Path(OUTPUT_DIR) / "final_model"
    trainer.save_model(str(final_save_path))
    tokenizer.save_pretrained(str(final_save_path))
    
    print(f"✅ Model saved to: {final_save_path}")
    
    # Test the model
    test_trained_model(model, tokenizer)
    
    return train_result, final_save_path

def test_trained_model(model, tokenizer):
    """Test the trained model with WorldModel examples."""
    print(f"\n🧪 Testing trained model...")
    
    test_prompts = [
        "Calculate 25% of 80",
        "What is 12 × 7?",
        "Count the R's in strawberry"
    ]
    
    model.eval()
    
    for i, prompt in enumerate(test_prompts):
        print(f"\n--- Test {i+1} ---")
        print(f"Prompt: {prompt}")
        
        # Format prompt
        formatted = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        
        # Tokenize
        inputs = tokenizer(formatted, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=200,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        # Decode
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        new_text = response[len(formatted):].strip()
        
        print(f"Response: {new_text}")
        
        # Check for WorldModel structure
        has_think = "<think>" in new_text
        has_model = "<model>" in new_text
        has_requires = "<requires>" in new_text
        
        structure_score = sum([has_think, has_model, has_requires])
        print(f"WorldModel structure: {structure_score}/3 ({'✅' if structure_score >= 2 else '⚠️'})")

if __name__ == "__main__":
    print("🔥 WorldModel Training for ROCm MI50")
    print("=" * 60)
    
    try:
        # Source ROCm environment variables
        os.environ['HSA_OVERRIDE_GFX_VERSION'] = '9.0.6'
        os.environ['HIP_VISIBLE_DEVICES'] = '0'
        os.environ['ROCBLAS_LAYER'] = '0'
        os.environ['PYTORCH_HIP_ALLOC_CONF'] = 'max_split_size_mb:128'
        
        result, model_path = train_worldmodel()
        
        print(f"\n🎊 SUCCESS!")
        print(f"   Model trained and saved to: {model_path}")
        print(f"   Final loss: {result.training_loss:.6f}")
        print(f"\n💡 To use the trained model:")
        print(f"   python3 -c \"from transformers import AutoModelForCausalLM, AutoTokenizer\"")
        print(f"   model = AutoModelForCausalLM.from_pretrained('{model_path}')\"")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()