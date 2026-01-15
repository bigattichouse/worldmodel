#!/usr/bin/env python3
"""
Fixed ROCm WorldModel Training Script
====================================

Addresses all major issues found:
1. Proper Qwen2.5/Qwen3 chat template formatting
2. ROCm MI50 compatible settings (gfx906)
3. Correct tokenization and loss calculation
4. Proper learning rate and batch size for fine-tuning
"""

import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, TrainingArguments, 
    Trainer, DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
import json
import os
import time
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass
from torch.utils.data import Dataset

# ROCm Detection and Setup
print("=== ROCm GPU Detection ===")
print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")

if torch.cuda.is_available():
    device = torch.device("cuda")
    try:
        gpu_name = torch.cuda.get_device_name()
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"✅ GPU detected: {gpu_name}")
        print(f"Memory: {total_memory:.1f}GB")
        
        # Test basic GPU operations
        test_tensor = torch.randn(100, 100).cuda()
        torch.cuda.empty_cache()
        print("✅ GPU context initialized successfully")
        del test_tensor
        torch.cuda.empty_cache()
        
    except Exception as e:
        print(f"⚠️  GPU detected but context failed: {e}")
        device = torch.device("cpu")
else:
    device = torch.device("cpu")
    print("❌ Using CPU")

print(f"Training device: {device}")
print("=" * 50)

@dataclass
class WorldModelExample:
    user_input: str
    assistant_output: str
    
    @classmethod
    def from_training_text(cls, text_block: str):
        """Parse training data text block into structured example."""
        lines = text_block.strip().split('\n')
        user_input = ""
        assistant_output = ""
        
        in_user = False
        in_assistant = False
        
        for line in lines:
            line = line.strip()
            if line.startswith("User:"):
                user_input = line[5:].strip()
                in_user = True
                in_assistant = False
            elif line.startswith("Assistant:"):
                assistant_output = line[10:].strip()
                in_user = False
                in_assistant = True
            elif in_assistant and line:
                assistant_output += "\n" + line
        
        return cls(user_input=user_input, assistant_output=assistant_output)

class WorldModelDataset(Dataset):
    """Fixed dataset for WorldModel training with proper Qwen formatting."""
    
    def __init__(self, examples: List[WorldModelExample], tokenizer: AutoTokenizer, max_length: int = 1024):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Ensure pad token exists
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        
        print(f"Processing {len(examples)} examples...")
        self.processed_examples = self._process_examples()
    
    def _format_conversation(self, example: WorldModelExample) -> str:
        """Format using proper Qwen chat template."""
        messages = [
            {"role": "user", "content": example.user_input},
            {"role": "assistant", "content": example.assistant_output}
        ]
        
        # Use tokenizer's chat template if available
        if hasattr(self.tokenizer, 'apply_chat_template') and self.tokenizer.chat_template:
            try:
                formatted = self.tokenizer.apply_chat_template(
                    messages, 
                    tokenize=False, 
                    add_generation_prompt=False
                )
                return formatted
            except:
                pass
        
        # Fallback to manual Qwen format
        if "qwen" in self.tokenizer.name_or_path.lower():
            # Qwen format
            formatted = f"<|im_start|>user\n{example.user_input}<|im_end|>\n<|im_start|>assistant\n{example.assistant_output}<|im_end|>"
        else:
            # Generic format
            formatted = f"### Human: {example.user_input}\n\n### Assistant: {example.assistant_output}"
        
        return formatted
    
    def _process_examples(self) -> List[Dict[str, torch.Tensor]]:
        """Process examples with correct tokenization and loss masking."""
        processed = []
        
        for i, example in enumerate(self.examples):
            if i % 50 == 0:
                print(f"Processing example {i}/{len(self.examples)}...")
            
            # Format conversation
            formatted_text = self._format_conversation(example)
            
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
            
            # Create labels - initially copy input_ids
            labels = input_ids.clone()
            
            # Mask user tokens (only train on assistant response)
            user_part = f"<|im_start|>user\n{example.user_input}<|im_end|>\n<|im_start|>assistant\n"
            user_tokens = self.tokenizer.encode(user_part, add_special_tokens=False)
            
            if len(user_tokens) < len(input_ids):
                # Mask everything before assistant response
                labels[:len(user_tokens)] = -100
            
            # Also mask padding tokens
            labels[attention_mask == 0] = -100
            
            processed.append({
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'labels': labels
            })
        
        print(f"✅ Processed {len(processed)} examples successfully")
        return processed
    
    def __len__(self):
        return len(self.processed_examples)
    
    def __getitem__(self, idx):
        return self.processed_examples[idx]

def load_training_data(data_file: str) -> List[WorldModelExample]:
    """Load and parse training data."""
    print(f"Loading training data from: {data_file}")
    
    with open(data_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Split into examples (double newline separated)
    raw_examples = content.split('\n\n')
    examples = []
    
    for raw in raw_examples:
        if raw.strip() and 'User:' in raw and 'Assistant:' in raw:
            try:
                example = WorldModelExample.from_training_text(raw)
                if example.user_input and example.assistant_output:
                    examples.append(example)
            except Exception as e:
                print(f"Warning: Failed to parse example: {e}")
                continue
    
    print(f"✅ Loaded {len(examples)} valid examples")
    
    # Show a sample
    if examples:
        print(f"\nSample example:")
        print(f"User: {examples[0].user_input[:100]}...")
        print(f"Assistant: {examples[0].assistant_output[:100]}...")
    
    return examples

def setup_model_and_tokenizer(model_name: str):
    """Setup model and tokenizer with ROCm optimizations."""
    print(f"Loading model: {model_name}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        padding_side="right"
    )
    
    # Setup special tokens
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Load model with ROCm-friendly settings
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.float32,  # Use float32 for ROCm stability
        device_map={"": 0} if torch.cuda.is_available() else None,
        low_cpu_mem_usage=True,
        use_cache=False,  # Disable for training
        attn_implementation="eager"  # ROCm compatibility
    )
    
    # Resize embeddings if needed (shouldn't be for standard models)
    original_vocab_size = len(tokenizer)
    if len(tokenizer) != model.config.vocab_size:
        print(f"Resizing embeddings: {model.config.vocab_size} -> {len(tokenizer)}")
        model.resize_token_embeddings(len(tokenizer))
    
    return model, tokenizer

def setup_lora(model, lora_config):
    """Setup LoRA with conservative settings for ROCm."""
    print("Setting up LoRA...")
    
    # Prepare model for LoRA training
    model = prepare_model_for_kbit_training(model)
    
    # Apply LoRA
    model = get_peft_model(model, lora_config)
    
    # Print parameter info
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in model.parameters())
    
    print(f"LoRA setup complete:")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Total parameters: {all_params:,}")
    print(f"  Percentage trainable: {100 * trainable_params / all_params:.2f}%")
    
    return model

def train_worldmodel():
    """Main training function with fixed ROCm settings."""
    
    # Configuration - Conservative settings for ROCm MI50
    MODEL_NAME = "/home/bigattichouse/workspace/model/Qwen3-0.6B"  # Start small
    DATA_FILE = "/home/bigattichouse/workspace/worldmodel/data/worldmodel_training.txt"
    OUTPUT_DIR = "./worldmodel_rocm_training"
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load data
    examples = load_training_data(DATA_FILE)
    if not examples:
        raise ValueError("No training examples loaded!")
    
    # Split data
    split_idx = int(0.9 * len(examples))  # 90% train, 10% eval
    train_examples = examples[:split_idx]
    eval_examples = examples[split_idx:]
    
    print(f"Training examples: {len(train_examples)}")
    print(f"Eval examples: {len(eval_examples)}")
    
    # Setup model
    model, tokenizer = setup_model_and_tokenizer(MODEL_NAME)
    
    # LoRA configuration - Very conservative for MI50
    lora_config = LoraConfig(
        r=8,  # Small rank
        lora_alpha=16,  # Conservative alpha
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    
    # Apply LoRA
    model = setup_lora(model, lora_config)
    
    # Create datasets
    train_dataset = WorldModelDataset(train_examples, tokenizer)
    eval_dataset = WorldModelDataset(eval_examples, tokenizer)
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        return_tensors="pt"
    )
    
    # Training arguments - ROCm MI50 optimized
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=2,  # Conservative
        per_device_train_batch_size=1,  # Very small for MI50
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,  # Effective batch size of 8
        learning_rate=1e-4,  # Conservative learning rate
        warmup_steps=50,
        weight_decay=0.01,
        logging_steps=10,
        eval_steps=100,
        save_steps=200,
        evaluation_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        dataloader_num_workers=0,  # No multiprocessing for ROCm
        fp16=False,  # Disable FP16 for ROCm stability
        bf16=False,  # Disable BF16
        gradient_checkpointing=True,
        remove_unused_columns=False,
        dataloader_pin_memory=False,  # ROCm compatibility
        report_to=None,  # Disable wandb
        push_to_hub=False
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer
    )
    
    print("🚀 Starting training...")
    start_time = time.time()
    
    # Train
    train_result = trainer.train()
    
    training_time = time.time() - start_time
    
    # Save model
    final_save_path = Path(OUTPUT_DIR) / "final_model"
    model.save_pretrained(str(final_save_path))
    tokenizer.save_pretrained(str(final_save_path))
    
    # Save merged model for inference
    merged_save_path = Path(OUTPUT_DIR) / "merged_model"
    merged_model = model.merge_and_unload()
    merged_model.save_pretrained(str(merged_save_path))
    tokenizer.save_pretrained(str(merged_save_path))
    
    # Print results
    print("\n" + "="*50)
    print("🎉 Training Complete!")
    print("="*50)
    print(f"Final training loss: {train_result.training_loss:.6f}")
    print(f"Training time: {training_time:.1f} seconds")
    print(f"Model saved to: {final_save_path}")
    print(f"Merged model saved to: {merged_save_path}")
    
    # Test the model with a sample
    test_model_output(merged_model, tokenizer)
    
    return train_result

def test_model_output(model, tokenizer):
    """Test the trained model with a sample WorldModel prompt."""
    print("\n" + "="*50)
    print("🧪 Testing Trained Model")
    print("="*50)
    
    test_prompt = "Calculate 15% of 200"
    
    # Format as conversation
    messages = [{"role": "user", "content": test_prompt}]
    
    if hasattr(tokenizer, 'apply_chat_template'):
        formatted = tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
    else:
        formatted = f"<|im_start|>user\n{test_prompt}<|im_end|>\n<|im_start|>assistant\n"
    
    print(f"Input: {test_prompt}")
    print(f"Formatted prompt: {formatted}")
    
    # Tokenize
    inputs = tokenizer(formatted, return_tensors="pt").to(model.device)
    
    # Generate
    model.eval()
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
    
    # Extract just the new part
    new_text = response[len(formatted):].strip()
    
    print(f"Model response: {new_text}")
    
    # Check for WorldModel tags
    has_think = "<think>" in new_text
    has_model = "<model>" in new_text  
    has_requires = "<requires>" in new_text
    
    print(f"\n🔍 WorldModel Structure Check:")
    print(f"  <think> tags: {'✅' if has_think else '❌'}")
    print(f"  <model> tags: {'✅' if has_model else '❌'}")
    print(f"  <requires> tags: {'✅' if has_requires else '❌'}")
    
    if has_think and has_model and has_requires:
        print("🎉 SUCCESS: Model generating WorldModel format!")
    else:
        print("⚠️  Model needs more training to generate WorldModel format")

if __name__ == "__main__":
    # Source ROCm environment first
    print("🔥 Fixed ROCm WorldModel Training")
    print("Make sure to run: source /home/bigattichouse/workspace/rocm/setup-rocm-env.sh")
    print("=" * 60)
    
    try:
        train_result = train_worldmodel()
        print("✅ Training completed successfully!")
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()