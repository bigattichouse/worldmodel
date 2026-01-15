import asyncio
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, TrainingArguments, 
    Trainer, DataCollatorForLanguageModeling, TrainerCallback
)
try:
    from peft import (
        LoraConfig, get_peft_model, TaskType, 
        PeftModel, PeftConfig, prepare_model_for_kbit_training
    )
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    LoraConfig = None

try:
    from bitsandbytes import BitsAndBytesConfig
    BNBCONFIG_AVAILABLE = True
except ImportError:
    BNBCONFIG_AVAILABLE = False
    BitsAndBytesConfig = None

import json
import numpy as np
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union
import logging
import os
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None

from tqdm.auto import tqdm

from ..utils.config import TrainingConfig
from ..utils.logging import get_logger
from ..training.dataGenerator import TrainingExample, DataGenerator


@dataclass
class SFTConfig:
    model_name: str = "../model/Qwen2.5-3B-Instruct"  # Use Qwen2.5-3B-Instruct path
    max_sequence_length: int = 2048  # Reduced for memory efficiency on ROCm
    learning_rate: float = 5e-5  # Conservative learning rate for fine-tuning
    batch_size: int = 1  # Very small batch for ROCm MI50
    gradient_accumulation_steps: int = 16  # Compensate with more accumulation
    num_epochs: int = 1  # Single epoch for ROCm testing
    warmup_steps: int = 10  # Minimal warmup
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    save_steps: int = 50  # Save more frequently
    eval_steps: int = 25   # Evaluate more frequently
    logging_steps: int = 5   # More frequent logging
    output_dir: str = "./qwen2.5_sft_checkpoints"
    use_wandb: bool = False
    wandb_project: str = "worldmodel_qwen2.5_sft"
    
    # LoRA configuration
    use_lora: bool = True
    lora_rank: int = 8  # Reduced rank for memory efficiency
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    lora_target_modules: List[str] = None
    
    # Quantization configuration - disabled for ROCm compatibility
    use_4bit: bool = False
    use_8bit: bool = False
    bnb_4bit_compute_dtype: str = "float16"
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True
    
    def __post_init__(self):
        if self.lora_target_modules is None:
            # Default LoRA targets for Qwen2.5 models  
            self.lora_target_modules = ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    
    def to_training_args(self) -> TrainingArguments:
        """Convert to HuggingFace TrainingArguments."""
        return TrainingArguments(
            output_dir=self.output_dir,
            learning_rate=self.learning_rate,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            num_train_epochs=self.num_epochs,
            warmup_steps=self.warmup_steps,
            weight_decay=self.weight_decay,
            max_grad_norm=self.max_grad_norm,
            save_steps=self.save_steps,
            eval_steps=self.eval_steps,
            logging_steps=self.logging_steps,
            eval_strategy="steps",
            save_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            report_to="wandb" if self.use_wandb else None,
            dataloader_num_workers=0,  # Disable multiprocessing for ROCm stability
            fp16=False,  # Disable FP16 to avoid gradient scaling issues
            bf16=False,  # Disable BF16 for ROCm compatibility
            gradient_checkpointing=True,
            remove_unused_columns=False,
            dataloader_pin_memory=False,  # Disable pin memory for ROCm
            max_steps=-1,  # No step limit
            push_to_hub=False,  # Disable hub operations
            hub_model_id=None
        )

@dataclass
class TrainingMetrics:
    epoch: int
    step: int
    train_loss: float
    eval_loss: float
    learning_rate: float
    throughput: float  # examples per second
    timestamp: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data

class WorldModelDataset(Dataset):
    def __init__(self, examples: List[TrainingExample], tokenizer: AutoTokenizer, 
                 max_length: int = 2048, include_thinking: bool = True):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.include_thinking = include_thinking
        
        # Set padding token if it doesn't exist
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        self.processed_examples = self._process_examples()
    
    def _format_example(self, example: TrainingExample) -> str:
        """Format training example into a conversation format."""
        # Create Phi-4 conversation format using chat template
        messages = [
            {'role': 'user', 'content': example.input_text},
            {'role': 'assistant', 'content': example.target_output}
        ]
        
        # Use tokenizer's chat template if available
        if hasattr(self.tokenizer, 'apply_chat_template') and self.tokenizer.chat_template:
            conversation = self.tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=False
            )
        else:
            # Fallback format
            conversation = f"<|user|>{example.input_text}<|end|><|assistant|>{example.target_output}<|end|><|endoftext|>"
        
        return conversation
    
    def _process_examples(self) -> List[Dict[str, torch.Tensor]]:
        """Pre-process all examples for faster training."""
        processed = []
        
        for example in tqdm(self.examples, desc="Processing training data"):
            # Format the conversation
            text = self._format_example(example)
            
            # Tokenize
            encoding = self.tokenizer(
                text,
                truncation=True,
                max_length=self.max_length,
                padding="max_length",  # Pad to max_length for consistent batching
                return_tensors="pt"
            )
            
            input_ids = encoding['input_ids'].squeeze()
            attention_mask = encoding['attention_mask'].squeeze()
            
            # For causal LM, labels are the same as input_ids
            labels = input_ids.clone()
            
            # Find the model response start to only compute loss on model tokens
            model_start = self._find_assistant_start(input_ids)
            if model_start > 0:
                # Mask user tokens by setting them to -100 (ignored in loss)
                labels[:model_start] = -100
            
            processed.append({
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'labels': labels
            })
        
        return processed
    
    def _find_assistant_start(self, input_ids: torch.Tensor) -> int:
        """Find where the model response starts."""
        # Convert to text and look for assistant marker
        text = self.tokenizer.decode(input_ids, skip_special_tokens=False)
        assistant_pos = text.find("<|assistant|>")
        
        if assistant_pos == -1:
            return 0  # Fallback to start if marker not found
        
        # Find the token position corresponding to assistant start
        assistant_tokens = self.tokenizer.encode("<|assistant|>", add_special_tokens=False)
        
        # Simple approach: find the assistant tokens in input_ids
        for i in range(len(input_ids) - len(assistant_tokens) + 1):
            if torch.equal(input_ids[i:i+len(assistant_tokens)], torch.tensor(assistant_tokens)):
                return i + len(assistant_tokens)
        
        return 0
    
    def __len__(self) -> int:
        return len(self.processed_examples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.processed_examples[idx]

class SFTTrainer:
    def __init__(self, config: SFTConfig):
        self.config = config
        self.logger = get_logger(__name__)
        
        # Initialize tokenizer and model
        self.tokenizer = None
        self.model = None
        self.trainer = None
        
        # Training state
        self.training_metrics: List[TrainingMetrics] = []
        self.best_eval_loss = float('inf')
        
        self.logger.info("SFT Trainer initialized")
    
    def _setup_model_and_tokenizer(self):
        """Initialize the model and tokenizer."""
        try:
            # Initialize CUDA/ROCm context first
            if torch.cuda.is_available():
                torch.cuda.init()
                torch.cuda.empty_cache()
                # Force GPU context creation
                dummy = torch.zeros(1).cuda()
                del dummy
                torch.cuda.empty_cache()
            
            self.logger.info(f"Loading model: {self.config.model_name}")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_name,
                trust_remote_code=True,
                padding_side="right"  # Important for causal LM training
            )
            
            # Set special tokens if missing
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Add special tokens for our format - use Qwen2.5 compatible tokens
            special_tokens = ["<|user|>", "<|assistant|>", "<|end|>"]
            new_tokens = []
            for token in special_tokens:
                if token not in self.tokenizer.get_vocab():
                    new_tokens.append(token)
            
            if new_tokens:
                self.tokenizer.add_tokens(new_tokens)
                self.logger.info(f"Added {len(new_tokens)} special tokens")
            
            # Setup quantization if enabled
            quantization_config = None
            if self.config.use_4bit and BNBCONFIG_AVAILABLE:
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=getattr(torch, self.config.bnb_4bit_compute_dtype),
                    bnb_4bit_quant_type=self.config.bnb_4bit_quant_type,
                    bnb_4bit_use_double_quant=self.config.bnb_4bit_use_double_quant,
                )
            elif self.config.use_8bit and BNBCONFIG_AVAILABLE:
                quantization_config = BitsAndBytesConfig(load_in_8bit=True)
            
            # Load model with ROCm-specific optimizations
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                trust_remote_code=True,
                torch_dtype=torch.float32,  # Use float32 for ROCm stability
                device_map={"": 0} if torch.cuda.is_available() else None,
                quantization_config=quantization_config,
                low_cpu_mem_usage=True,  # Reduce CPU memory usage
                use_cache=False,  # Disable KV cache for training
                attn_implementation="eager"  # Use eager attention for ROCm compatibility
            )
            
            # Resize embeddings if we added tokens
            if new_tokens:
                self.model.resize_token_embeddings(len(self.tokenizer))
            
            # Prepare model for training if using quantization
            if quantization_config is not None and PEFT_AVAILABLE:
                self.model = prepare_model_for_kbit_training(self.model)
            
            # Setup LoRA if enabled
            if self.config.use_lora and PEFT_AVAILABLE:
                self._setup_lora()
            
            # Enable gradient checkpointing for memory efficiency
            if hasattr(self.model, 'gradient_checkpointing_enable'):
                self.model.gradient_checkpointing_enable()
            
            self.logger.info("Model and tokenizer loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            raise
    
    def _setup_lora(self):
        """Setup LoRA configuration for the model."""
        if not PEFT_AVAILABLE:
            self.logger.error("PEFT library not available. Install with: pip install peft")
            raise ImportError("PEFT library required for LoRA training")
        
        self.logger.info("Setting up LoRA configuration")
        
        # Create LoRA config
        lora_config = LoraConfig(
            r=self.config.lora_rank,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=self.config.lora_target_modules,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        
        # Apply LoRA to model
        self.model = get_peft_model(self.model, lora_config)
        
        # Print trainable parameters
        trainable_params, all_param = self._get_trainable_params()
        self.logger.info(f"LoRA setup complete. Trainable params: {trainable_params:,} ({100 * trainable_params / all_param:.2f}%) out of {all_param:,}")
        
        # Enable training mode for LoRA layers
        self.model.enable_input_require_grads()
    
    def _get_trainable_params(self) -> tuple:
        """Get count of trainable vs total parameters."""
        trainable_params = 0
        all_param = 0
        
        for _, param in self.model.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        
        return trainable_params, all_param
    
    def _create_data_collator(self) -> DataCollatorForLanguageModeling:
        """Create data collator for language modeling."""
        return DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,  # We're doing causal LM, not masked LM
            return_tensors="pt"
        )
    
    def _split_dataset(self, examples: List[TrainingExample], 
                      eval_split: float = 0.1) -> Tuple[List[TrainingExample], List[TrainingExample]]:
        """Split dataset into train and eval sets."""
        if eval_split <= 0:
            return examples, []
        
        split_idx = int(len(examples) * (1 - eval_split))
        train_examples = examples[:split_idx]
        eval_examples = examples[split_idx:]
        
        self.logger.info(f"Split dataset: {len(train_examples)} train, {len(eval_examples)} eval")
        return train_examples, eval_examples
    
    async def train(self, examples: List[TrainingExample], 
                   eval_split: float = 0.1,
                   resume_from_checkpoint: Optional[str] = None) -> Dict[str, Any]:
        """Train the model on the provided examples."""
        
        if not examples:
            raise ValueError("No training examples provided")
        
        # Setup model and tokenizer if not already done
        if self.model is None:
            self._setup_model_and_tokenizer()
        
        # Initialize wandb if enabled
        if self.config.use_wandb:
            if not WANDB_AVAILABLE:
                self.logger.warning("Wandb not available, disabling wandb logging")
                self.config.use_wandb = False
            else:
                wandb.init(
                    project=self.config.wandb_project,
                    config=asdict(self.config),
                    name=f"sft_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                )
        
        try:
            # Split data
            train_examples, eval_examples = self._split_dataset(examples, eval_split)
            
            # Create datasets
            train_dataset = WorldModelDataset(
                train_examples, 
                self.tokenizer, 
                self.config.max_sequence_length
            )
            
            eval_dataset = None
            if eval_examples:
                eval_dataset = WorldModelDataset(
                    eval_examples,
                    self.tokenizer,
                    self.config.max_sequence_length
                )
            
            # Create data collator
            data_collator = self._create_data_collator()
            
            # Setup training arguments
            training_args = self.config.to_training_args()
            
            # Create trainer
            self.trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                data_collator=data_collator,
                callbacks=[SFTTrainingCallback(self)]
            )
            
            self.logger.info(f"Starting training with {len(train_examples)} examples")
            
            # Train the model
            train_result = self.trainer.train(
                resume_from_checkpoint=resume_from_checkpoint
            )
            
            # Save the final model
            final_save_path = Path(self.config.output_dir) / "final_model"
            
            if self.config.use_lora and PEFT_AVAILABLE:
                # For LoRA, save both the adapter and the merged model
                self.model.save_pretrained(str(final_save_path))
                self.tokenizer.save_pretrained(str(final_save_path))
                
                # Also save the merged model for easier inference
                merged_save_path = Path(self.config.output_dir) / "merged_model"
                merged_model = self.model.merge_and_unload()
                merged_model.save_pretrained(str(merged_save_path))
                self.tokenizer.save_pretrained(str(merged_save_path))
                self.logger.info(f"LoRA adapter saved to {final_save_path}, merged model to {merged_save_path}")
            else:
                # Standard model saving
                self.trainer.save_model(str(final_save_path))
                self.tokenizer.save_pretrained(str(final_save_path))
                self.logger.info(f"Model saved to {final_save_path}")
            
            # Calculate final metrics
            final_metrics = {
                'train_loss': train_result.training_loss,
                'train_samples': len(train_examples),
                'eval_samples': len(eval_examples) if eval_examples else 0,
                'total_steps': train_result.global_step,
                'best_eval_loss': self.best_eval_loss,
                'training_runtime': train_result.metrics.get('train_runtime', 0.0),
                'model_save_path': str(final_save_path)
            }
            
            self.logger.info(f"Training completed. Final loss: {train_result.training_loss:.4f}")
            
            return final_metrics
            
        except Exception as e:
            self.logger.error(f"Training failed: {e}")
            raise
        
        finally:
            if self.config.use_wandb and WANDB_AVAILABLE:
                wandb.finish()
    
    def save_metrics(self, output_path: str):
        """Save training metrics to file."""
        metrics_data = {
            'config': asdict(self.config),
            'metrics': [metric.to_dict() for metric in self.training_metrics],
            'best_eval_loss': self.best_eval_loss,
            'saved_at': datetime.now(timezone.utc).isoformat()
        }
        
        with open(output_path, 'w') as f:
            json.dump(metrics_data, f, indent=2)
        
        self.logger.info(f"Saved metrics to {output_path}")
    
    def load_model(self, model_path: str):
        """Load a trained model."""
        try:
            self.logger.info(f"Loading trained model from {model_path}")
            
            # Load tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True
            )
            
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                trust_remote_code=True,
                torch_dtype=torch.float32,  # Use float32 for ROCm stability
                device_map={"": 0} if torch.cuda.is_available() else None,  # Force single GPU
                low_cpu_mem_usage=True
            )
            
            self.logger.info("Model loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            raise
    
    async def evaluate(self, examples: List[TrainingExample]) -> Dict[str, float]:
        """Evaluate the model on a set of examples."""
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model not loaded. Call train() or load_model() first.")
        
        eval_dataset = WorldModelDataset(
            examples,
            self.tokenizer,
            self.config.max_sequence_length
        )
        
        # Use trainer for evaluation if available
        if self.trainer is not None:
            eval_results = self.trainer.evaluate(eval_dataset)
            return eval_results
        
        # Manual evaluation
        self.model.eval()
        total_loss = 0.0
        total_samples = 0
        
        dataloader = DataLoader(
            eval_dataset, 
            batch_size=self.config.batch_size,
            collate_fn=self._create_data_collator()
        )
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                # Move to device
                if torch.cuda.is_available():
                    batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v 
                            for k, v in batch.items()}
                
                # Forward pass
                outputs = self.model(**{k: v for k, v in batch.items() 
                                      if k in ['input_ids', 'attention_mask', 'labels']})
                
                total_loss += outputs.loss.item() * len(batch['input_ids'])
                total_samples += len(batch['input_ids'])
        
        avg_loss = total_loss / total_samples
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        
        return {
            'eval_loss': avg_loss,
            'eval_perplexity': perplexity,
            'eval_samples': total_samples
        }
    
    async def generate_sample(self, prompt: str, max_new_tokens: int = 512,
                             temperature: float = 0.7, top_p: float = 0.9) -> str:
        """Generate a response for a given prompt."""
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model not loaded")
        
        # Format prompt
        formatted_prompt = f"<user>\n{prompt}\n</user>\n<assistant>\n"
        
        # Tokenize
        inputs = self.tokenizer(
            formatted_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_sequence_length - max_new_tokens
        )
        
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        # Generate
        self.model.eval()
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode response
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract just the assistant response
        if "<assistant>" in response:
            response = response.split("<assistant>")[-1].strip()
        
        return response

class SFTTrainingCallback(TrainerCallback):
    """Custom callback to track training metrics."""
    
    def __init__(self, trainer: SFTTrainer):
        self.sft_trainer = trainer
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        """Called when logging occurs."""
        if logs is None:
            return
        
        # Create training metric
        metric = TrainingMetrics(
            epoch=state.epoch,
            step=state.global_step,
            train_loss=logs.get('train_loss', 0.0),
            eval_loss=logs.get('eval_loss', 0.0),
            learning_rate=logs.get('learning_rate', 0.0),
            throughput=logs.get('train_samples_per_second', 0.0),
            timestamp=datetime.now(timezone.utc)
        )
        
        self.sft_trainer.training_metrics.append(metric)
        
        # Update best eval loss
        if 'eval_loss' in logs:
            eval_loss = logs['eval_loss']
            if eval_loss < self.sft_trainer.best_eval_loss:
                self.sft_trainer.best_eval_loss = eval_loss

# Convenience functions
async def quick_sft_train(examples: List[TrainingExample], 
                         model_name: str = "google/gemma-2-2b-it",
                         output_dir: str = "./sft_output") -> SFTTrainer:
    """Quickly train a model with default settings."""
    config = SFTConfig(
        model_name=model_name,
        output_dir=output_dir,
        num_epochs=1,
        batch_size=2,  # Small batch for quick training
        gradient_accumulation_steps=2
    )
    
    trainer = SFTTrainer(config)
    await trainer.train(examples)
    
    return trainer

async def train_from_dataset_file(dataset_file: str, config: SFTConfig = None) -> SFTTrainer:
    """Train a model from a saved dataset file."""
    if config is None:
        config = SFTConfig()
    
    # Load dataset
    generator = DataGenerator(TrainingConfig())
    examples = generator.load_dataset(dataset_file)
    
    if not examples:
        raise ValueError(f"No examples loaded from {dataset_file}")
    
    # Train model
    trainer = SFTTrainer(config)
    await trainer.train(examples)
    
    return trainer