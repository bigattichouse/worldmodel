#!/usr/bin/env python3
"""
ByteLogic Integrated WorldModel Training
========================================

Trains a WorldModel with integrated ByteLogic computation during forward pass.
Uses QwenWASMAdapter architecture with cross-modal computation layers.

This creates a model that:
1. Generates ByteLogic code at multiple layers (3, 7, 11)
2. Executes code during forward pass
3. Injects results into token stream
4. Learns computation-augmented reasoning

Based on WorldModel LLM specification with ByteLogic integration.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import sys
import json
import argparse
import logging
import time
from typing import Dict, List, Optional, Any
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from models.qwen_wasm_adapter import QwenWASMAdapter
from training.bytelogic_dataset_new import load_bytelogic_dataset, ByteLogicDataset
from execution.bytelogic_executor import ByteLogicExecutor
from tokenization.bytelogic_tokenizer import ByteLogicTokenizer

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Hardware detection
print("=== ByteLogic Integrated WorldModel Training ===")
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


class ByteLogicComputationTrainer:
    """Trainer for ByteLogic-integrated WorldModel."""
    
    def __init__(self, 
                 model_path: str,
                 dataset_path: str,
                 output_dir: str,
                 computation_layers: List[int] = [3, 7, 11, 15, 19, 23, 27],
                 learning_rate: float = 1e-5,
                 batch_size: int = 2,
                 max_length: int = 512):
        """
        Initialize integrated trainer.
        
        Args:
            model_path: Path to base model
            dataset_path: Path to ByteLogic training dataset
            output_dir: Output directory for checkpoints
            computation_layers: Layers for ByteLogic computation
            learning_rate: Training learning rate
            batch_size: Training batch size
            max_length: Maximum sequence length
        """
        self.model_path = model_path
        self.dataset_path = dataset_path
        self.output_dir = Path(output_dir)
        self.computation_layers = computation_layers
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_length = max_length
        
        # Create output directory
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize components
        self.device = device
        self._setup_model()
        self._setup_datasets()
        self._setup_optimizer()
        
        logger.info(f"🏗️ Integrated trainer initialized")
        logger.info(f"   Computation layers: {computation_layers}")
        logger.info(f"   Training with execution: True")
    
    def _setup_model(self):
        """Initialize the QwenWASMAdapter with ByteLogic integration."""
        logger.info(f"🧠 Initializing QwenWASMAdapter...")
        
        # Initialize with ByteLogic computation layers
        self.model = QwenWASMAdapter(
            model_path=self.model_path,
            cross_modal_layers=self.computation_layers,
            freeze_text_layers=False,  # Allow full training
            use_sandbox=False  # Disable sandbox for training speed
        )
        
        # Move to device
        self.model.to(self.device)
        
        # Set up ByteLogic tokenizer
        self.bytelogic_tokenizer = ByteLogicTokenizer()
        
        # Get text tokenizer for dataset processing
        self.text_tokenizer = self.model.text_tokenizer
        
        logger.info(f"✅ Model initialized with {sum(p.numel() for p in self.model.parameters()):,} parameters")
        logger.info(f"   Computation layers: {self.computation_layers}")
    
    def _setup_datasets(self):
        """Load and setup training datasets."""
        logger.info(f"📖 Loading ByteLogic datasets from {self.dataset_path}")
        
        # Load training dataset
        self.train_dataset = load_bytelogic_dataset(
            data_file=self.dataset_path,
            tokenizer=self.text_tokenizer,
            max_length=self.max_length,
            validation_mode=False
        )
        
        # Load validation dataset
        self.val_dataset = load_bytelogic_dataset(
            data_file=self.dataset_path,
            tokenizer=self.text_tokenizer,
            max_length=self.max_length,
            validation_mode=True
        )
        
        # Create data loaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self._collate_fn
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self._collate_fn
        )
        
        logger.info(f"✅ Datasets loaded")
        logger.info(f"   Training examples: {len(self.train_dataset)}")
        logger.info(f"   Validation examples: {len(self.val_dataset)}")
    
    def _collate_fn(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """Collate function for batch processing."""
        # Standard collation for now - can be enhanced for computation-specific needs
        input_ids = torch.stack([item['input_ids'] for item in batch])
        attention_mask = torch.stack([item['attention_mask'] for item in batch])
        labels = torch.stack([item['labels'] for item in batch])
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }
    
    def _setup_optimizer(self):
        """Setup optimizer and scheduler."""
        # Use AdamW with lower learning rate for integrated training
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=0.01,
            eps=1e-8
        )
        
        # Cosine annealing scheduler
        total_steps = len(self.train_loader) * 5  # 5 epochs
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=total_steps,
            eta_min=1e-7
        )
        
        logger.info(f"🔧 Optimizer configured: AdamW (lr={self.learning_rate})")
    
    def _compute_loss(self, outputs: Dict[str, Any], labels: torch.Tensor) -> torch.Tensor:
        """
        Compute loss for integrated ByteLogic training.
        
        Includes both language modeling loss and computation consistency loss.
        """
        # Standard language modeling loss
        logits = outputs['logits']
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        # Flatten for cross entropy
        loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
        lm_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        
        # Computation consistency loss (if execution results available)
        computation_loss = 0.0
        execution_results = outputs.get('execution_results', [])
        
        if execution_results:
            # Encourage successful execution and valid ByteLogic syntax
            for result in execution_results:
                if result.get('executed', False) and result.get('success', False):
                    # Small reward for successful execution
                    computation_loss -= 0.1
                else:
                    # Small penalty for failed execution
                    computation_loss += 0.1
            
            computation_loss = torch.tensor(computation_loss, device=self.device, requires_grad=True)
        
        # Combine losses
        total_loss = lm_loss + 0.1 * computation_loss
        
        return total_loss, lm_loss, computation_loss
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train one epoch with integrated ByteLogic execution."""
        self.model.train()
        
        total_loss = 0.0
        total_lm_loss = 0.0
        total_comp_loss = 0.0
        num_batches = 0
        
        logger.info(f"🏋️ Training epoch {epoch + 1}")
        
        for batch_idx, batch in enumerate(self.train_loader):
            # Move batch to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Forward pass with ByteLogic execution
            self.optimizer.zero_grad()
            
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                execute_wasm=True,  # CRITICAL: Enable computation during training
                return_dict=True
            )
            
            # Compute loss
            loss, lm_loss, comp_loss = self._compute_loss(outputs, labels)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # Optimizer step
            self.optimizer.step()
            self.scheduler.step()
            
            # Update metrics
            total_loss += loss.item()
            total_lm_loss += lm_loss.item()
            if isinstance(comp_loss, torch.Tensor):
                total_comp_loss += comp_loss.item()
            num_batches += 1
            
            # Log progress
            if batch_idx % 10 == 0:
                logger.info(f"  Batch {batch_idx}/{len(self.train_loader)}: loss={loss.item():.4f}, lm_loss={lm_loss.item():.4f}")
        
        # Average losses
        avg_loss = total_loss / num_batches
        avg_lm_loss = total_lm_loss / num_batches
        avg_comp_loss = total_comp_loss / num_batches
        
        return {
            'total_loss': avg_loss,
            'lm_loss': avg_lm_loss,
            'computation_loss': avg_comp_loss
        }
    
    def validate(self) -> Dict[str, float]:
        """Validate model with ByteLogic execution."""
        self.model.eval()
        
        total_loss = 0.0
        total_lm_loss = 0.0
        total_comp_loss = 0.0
        num_batches = 0
        successful_executions = 0
        total_executions = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                # Move batch to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Forward pass with ByteLogic execution
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    execute_wasm=True,  # Execute during validation too
                    return_dict=True
                )
                
                # Compute loss
                loss, lm_loss, comp_loss = self._compute_loss(outputs, labels)
                
                # Update metrics
                total_loss += loss.item()
                total_lm_loss += lm_loss.item()
                if isinstance(comp_loss, torch.Tensor):
                    total_comp_loss += comp_loss.item()
                num_batches += 1
                
                # Track execution success
                execution_results = outputs.get('execution_results', [])
                for result in execution_results:
                    total_executions += 1
                    if result.get('executed', False) and result.get('success', False):
                        successful_executions += 1
        
        # Average metrics
        avg_loss = total_loss / num_batches
        avg_lm_loss = total_lm_loss / num_batches
        avg_comp_loss = total_comp_loss / num_batches
        execution_success_rate = successful_executions / max(total_executions, 1)
        
        return {
            'val_loss': avg_loss,
            'val_lm_loss': avg_lm_loss,
            'val_computation_loss': avg_comp_loss,
            'execution_success_rate': execution_success_rate
        }
    
    def save_checkpoint(self, epoch: int, metrics: Dict[str, float]):
        """Save model checkpoint."""
        checkpoint_path = self.output_dir / f"checkpoint_epoch_{epoch + 1}.pt"
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics,
            'computation_layers': self.computation_layers
        }, checkpoint_path)
        
        logger.info(f"💾 Saved checkpoint: {checkpoint_path}")
    
    def train(self, epochs: int = 5):
        """Run complete integrated training."""
        logger.info(f"🚀 Starting integrated ByteLogic training")
        logger.info(f"   Epochs: {epochs}")
        logger.info(f"   Computation layers: {self.computation_layers}")
        logger.info(f"   Execution during training: True")
        
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            # Train epoch
            train_metrics = self.train_epoch(epoch)
            
            # Validate
            val_metrics = self.validate()
            
            # Log metrics
            logger.info(f"📊 Epoch {epoch + 1}/{epochs}")
            logger.info(f"   Train Loss: {train_metrics['total_loss']:.4f}")
            logger.info(f"   Val Loss: {val_metrics['val_loss']:.4f}")
            logger.info(f"   Execution Success: {val_metrics['execution_success_rate']:.2%}")
            
            # Save checkpoint if best validation loss
            if val_metrics['val_loss'] < best_val_loss:
                best_val_loss = val_metrics['val_loss']
                self.save_checkpoint(epoch, {**train_metrics, **val_metrics})
                logger.info(f"✅ New best validation loss: {best_val_loss:.4f}")
        
        # Save final model
        final_path = self.output_dir / "final_integrated_model.pt"
        torch.save(self.model.state_dict(), final_path)
        
        logger.info(f"🎉 Training completed!")
        logger.info(f"   Final model: {final_path}")
        logger.info(f"   Best validation loss: {best_val_loss:.4f}")


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="ByteLogic Integrated WorldModel Training")
    parser.add_argument("--model", required=True, help="Path to base model")
    parser.add_argument("--dataset", required=True, help="Path to ByteLogic dataset")
    parser.add_argument("--output_dir", default="integrated_worldmodel_output", help="Output directory")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--computation_layers", nargs='+', type=int, default=[3, 7, 11, 15, 19, 23, 27], help="Computation layer indices")
    
    args = parser.parse_args()
    
    logger.info(f"🎯 Integrated ByteLogic WorldModel Training")
    logger.info(f"   Model: {args.model}")
    logger.info(f"   Dataset: {args.dataset}")
    logger.info(f"   Output: {args.output_dir}")
    logger.info(f"   Computation layers: {args.computation_layers}")
    
    # Initialize trainer
    trainer = ByteLogicComputationTrainer(
        model_path=args.model,
        dataset_path=args.dataset,
        output_dir=args.output_dir,
        computation_layers=args.computation_layers,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size
    )
    
    # Run training
    try:
        trainer.train(epochs=args.epochs)
        return True
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)