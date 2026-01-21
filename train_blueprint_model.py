#!/usr/bin/env python3
"""
BluePrint WorldModel Training Script
===================================

Trains a model on BluePrint methodology using the thinking → blueprint token pattern.
Automatically scans training/datasets/ for JSONL files and creates progressive curriculum.

This creates a model that:
1. Generates <thinking> tokens for problem understanding
2. Generates <blueprint> tokens with proper BluePrint notation
3. Follows the collaborative specification framework

Based on docs/worldmodel-blueprint-plan.md Phase 1 implementation.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import os
import sys
import json
import argparse
import logging
import time
import glob
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments, 
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import Dataset as HFDataset

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Hardware detection
print("=== BluePrint WorldModel Training ===")
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


class BluePrintDatasetScanner:
    """Scans training/datasets/ directory and loads all JSONL files."""
    
    def __init__(self, datasets_dir: str = "training/datasets"):
        self.datasets_dir = Path(datasets_dir)
        self.categories = {}
        
    def scan_datasets(self) -> Dict[str, List[Dict]]:
        """Scan all JSONL files and organize by category."""
        logger.info(f"🔍 Scanning datasets in {self.datasets_dir}")
        
        all_examples = []
        category_counts = {}
        
        # Find all JSONL files recursively
        jsonl_files = list(self.datasets_dir.glob("**/*.jsonl"))
        
        for jsonl_file in jsonl_files:
            category = jsonl_file.parent.name
            logger.info(f"   Loading {jsonl_file.name} from {category}/")
            
            try:
                examples = self._load_jsonl_file(jsonl_file)
                all_examples.extend(examples)
                
                if category not in category_counts:
                    category_counts[category] = 0
                category_counts[category] += len(examples)
                
            except Exception as e:
                logger.warning(f"   ⚠️  Failed to load {jsonl_file}: {e}")
        
        # Log summary
        logger.info(f"✅ Loaded {len(all_examples)} total examples:")
        for category, count in category_counts.items():
            logger.info(f"   {category}: {count} examples")
            
        return {
            'all_examples': all_examples,
            'categories': category_counts
        }
    
    def _load_jsonl_file(self, file_path: Path) -> List[Dict]:
        """Load examples from a JSONL file."""
        examples = []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                    
                try:
                    example = json.loads(line)
                    
                    # Validate required fields
                    required_fields = ['id', 'user_query', 'response']
                    if not all(field in example for field in required_fields):
                        logger.warning(f"   Missing required fields in {file_path}:{line_num}")
                        continue
                    
                    # Validate BluePrint format
                    if not self._validate_blueprint_format(example['response']):
                        logger.warning(f"   Invalid BluePrint format in {file_path}:{line_num}")
                        continue
                        
                    examples.append(example)
                    
                except json.JSONDecodeError as e:
                    logger.warning(f"   Invalid JSON in {file_path}:{line_num}: {e}")
                    continue
        
        return examples
    
    def _validate_blueprint_format(self, response: str) -> bool:
        """Validate that response contains proper <thinking> and <blueprint> tags."""
        has_thinking = '<thinking>' in response and '</thinking>' in response
        has_blueprint = '<blueprint>' in response and '</blueprint>' in response
        return has_thinking and has_blueprint


class BluePrintDataset(Dataset):
    """Dataset for BluePrint training with thinking → blueprint pattern."""
    
    def __init__(self, examples: List[Dict], tokenizer, max_length: int = 768):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Add special tokens if not present
        special_tokens = ['<thinking>', '</thinking>', '<blueprint>', '</blueprint>']
        self.tokenizer.add_tokens(special_tokens)
        
        logger.info(f"📚 BluePrintDataset initialized with {len(examples)} examples")
        logger.info(f"   Max length: {max_length} tokens")
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        example = self.examples[idx]
        
        # Format the training text
        text = f"Query: {example['user_query']}\n\nResponse: {example['response']}"
        
        # Tokenize
        encoded = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Prepare inputs and labels
        input_ids = encoded['input_ids'].squeeze()
        attention_mask = encoded['attention_mask'].squeeze()
        
        # For causal LM, labels are the same as input_ids
        labels = input_ids.clone()
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }


class BluePrintTrainer:
    """Trainer for BluePrint methodology using progressive curriculum."""
    
    def __init__(self,
                 model_path: str,
                 datasets_dir: str = "training/datasets",
                 output_dir: str = "blueprint_model_output",
                 max_length: int = 768,
                 learning_rate: float = 2e-4,
                 batch_size: int = 2,
                 gradient_accumulation_steps: int = 8):
        
        self.model_path = model_path
        self.datasets_dir = datasets_dir
        self.output_dir = Path(output_dir)
        self.max_length = max_length
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        
        # Create output directory
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize components
        self._setup_model_and_tokenizer()
        self._load_datasets()
        
        logger.info(f"🏗️ BluePrintTrainer initialized")
        logger.info(f"   Model: {model_path}")
        logger.info(f"   Output: {output_dir}")
        logger.info(f"   Training examples: {len(self.train_examples)}")
        logger.info(f"   Validation examples: {len(self.val_examples)}")
    
    def _setup_model_and_tokenizer(self):
        """Initialize model and tokenizer."""
        logger.info(f"🧠 Loading model and tokenizer from {self.model_path}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        
        # Add pad token if missing
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16,  # Use FP16 for efficiency
            device_map="auto"
        )
        
        # Resize embeddings for new special tokens
        self.model.resize_token_embeddings(len(self.tokenizer))
        
        logger.info(f"✅ Model loaded with {sum(p.numel() for p in self.model.parameters()):,} parameters")
    
    def _load_datasets(self):
        """Load and prepare datasets."""
        scanner = BluePrintDatasetScanner(self.datasets_dir)
        dataset_info = scanner.scan_datasets()
        
        all_examples = dataset_info['all_examples']
        
        # Split into train/validation (80/20)
        split_idx = int(len(all_examples) * 0.8)
        
        # Shuffle examples for good distribution
        import random
        random.shuffle(all_examples)
        
        self.train_examples = all_examples[:split_idx]
        self.val_examples = all_examples[split_idx:]
        
        # Create datasets
        self.train_dataset = BluePrintDataset(
            self.train_examples, 
            self.tokenizer, 
            self.max_length
        )
        
        self.val_dataset = BluePrintDataset(
            self.val_examples, 
            self.tokenizer, 
            self.max_length
        )
        
        logger.info(f"✅ Datasets prepared")
    
    def train(self, epochs: int = 15):
        """Run BluePrint training with HuggingFace Trainer."""
        logger.info(f"🚀 Starting BluePrint training")
        logger.info(f"   Epochs: {epochs}")
        logger.info(f"   Learning rate: {self.learning_rate}")
        logger.info(f"   Batch size: {self.batch_size}")
        logger.info(f"   Gradient accumulation: {self.gradient_accumulation_steps}")
        
        # Calculate total steps
        total_steps = len(self.train_dataset) // (self.batch_size * self.gradient_accumulation_steps) * epochs
        warmup_steps = int(0.1 * total_steps)
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            overwrite_output_dir=True,
            num_train_epochs=epochs,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            learning_rate=self.learning_rate,
            weight_decay=0.01,
            warmup_steps=warmup_steps,
            lr_scheduler_type="cosine",
            logging_dir=self.output_dir / "logs",
            logging_steps=50,
            eval_steps=500,
            save_steps=1000,
            evaluation_strategy="steps",
            save_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            report_to="none",  # Disable wandb
            fp16=True,  # Use FP16 for efficiency
            dataloader_pin_memory=False,  # May help with ROCm
            remove_unused_columns=False,
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,  # We're doing causal LM, not masked LM
            return_tensors="pt"
        )
        
        # Convert to HuggingFace datasets
        train_hf_dataset = HFDataset.from_list([
            {
                'input_ids': self.train_dataset[i]['input_ids'],
                'attention_mask': self.train_dataset[i]['attention_mask'],
                'labels': self.train_dataset[i]['labels']
            }
            for i in range(len(self.train_dataset))
        ])
        
        val_hf_dataset = HFDataset.from_list([
            {
                'input_ids': self.val_dataset[i]['input_ids'],
                'attention_mask': self.val_dataset[i]['attention_mask'],
                'labels': self.val_dataset[i]['labels']
            }
            for i in range(len(self.val_dataset))
        ])
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_hf_dataset,
            eval_dataset=val_hf_dataset,
            data_collator=data_collator,
            tokenizer=self.tokenizer,
        )
        
        # Run training
        logger.info(f"🏋️ Starting training...")
        trainer.train()
        
        # Save final model
        final_model_path = self.output_dir / "final_blueprint_model"
        trainer.save_model(final_model_path)
        self.tokenizer.save_pretrained(final_model_path)
        
        logger.info(f"🎉 Training completed!")
        logger.info(f"   Final model saved to: {final_model_path}")
        
        return trainer


def test_generation(model_path: str, test_query: str = "Design a temperature conversion service"):
    """Test the trained model with a sample query."""
    logger.info(f"🧪 Testing model generation...")
    
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, device_map="auto")
    
    # Format input
    input_text = f"Query: {test_query}\n\nResponse: "
    
    # Tokenize
    inputs = tokenizer(input_text, return_tensors="pt")
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            attention_mask=inputs.attention_mask,
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
    has_thinking = '<thinking>' in response and '</thinking>' in response
    has_blueprint = '<blueprint>' in response and '</blueprint>' in response
    
    logger.info(f"✅ Validation:")
    logger.info(f"   Contains <thinking> tags: {has_thinking}")
    logger.info(f"   Contains <blueprint> tags: {has_blueprint}")
    logger.info(f"   Format valid: {has_thinking and has_blueprint}")


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="BluePrint WorldModel Training")
    parser.add_argument("--model", required=True, help="Path to base model (e.g., Qwen3-0.6B)")
    parser.add_argument("--datasets_dir", default="training/datasets", help="Path to datasets directory")
    parser.add_argument("--output_dir", default="blueprint_model_output", help="Output directory")
    parser.add_argument("--epochs", type=int, default=15, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--max_length", type=int, default=768, help="Maximum sequence length")
    parser.add_argument("--gradient_accumulation", type=int, default=8, help="Gradient accumulation steps")
    parser.add_argument("--test_only", action="store_true", help="Only run generation test")
    parser.add_argument("--test_model", type=str, help="Model path for testing (if different from --model)")

    args = parser.parse_args()

    if args.test_only:
        test_model_path = args.test_model or args.model
        test_generation(test_model_path)
        return True

    logger.info(f"🎯 BluePrint WorldModel Training")
    logger.info(f"   Model: {args.model}")
    logger.info(f"   Datasets: {args.datasets_dir}")
    logger.info(f"   Output: {args.output_dir}")

    # Initialize trainer
    trainer = BluePrintTrainer(
        model_path=args.model,
        datasets_dir=args.datasets_dir,
        output_dir=args.output_dir,
        max_length=args.max_length,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation
    )

    # Run training
    try:
        hf_trainer = trainer.train(epochs=args.epochs)
        
        # Test the trained model
        final_model_path = Path(args.output_dir) / "final_blueprint_model"
        if final_model_path.exists():
            test_generation(str(final_model_path))
        
        return True
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)