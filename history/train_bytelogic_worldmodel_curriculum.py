#!/usr/bin/env python3
"""
ByteLogic Integrated WorldModel Training with Curriculum Learning
===============================================================

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
import re

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from models.qwen_wasm_adapter import QwenWASMAdapter
from training.bytelogic_dataset_new import load_bytelogic_dataset, ByteLogicDataset
from execution.bytelogic_executor import ByteLogicExecutor
from tokenization.bytelogic_tokenizer import ByteLogicTokenizer

# Import the multi-dataset loader
sys.path.insert(0, str(Path(__file__).parent / "tools"))
from multi_dataset_loader import DynamicDatasetLoader

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Hardware detection
print("=== ByteLogic Integrated WorldModel Training with Curriculum Learning ===")
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


class ThreeTierByteLogicDataset(ByteLogicDataset):
    """Dataset for training with three-tier approach: thinking -> pseudocode -> computation."""

    def _load_examples(self, data_file: str) -> List[Dict[str, Any]]:
        """Load examples from JSONL or JSON file."""
        examples = []

        if data_file.endswith('.jsonl'):
            # Load JSONL format
            with open(data_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        example = json.loads(line)
                        processed = self._process_example(example, line_num)
                        if processed:
                            examples.append(processed)
                    except json.JSONDecodeError as e:
                        logger.warning(f"Failed to parse line {line_num + 1}: {e}")
        else:
            # Load JSON format
            with open(data_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            if isinstance(data, dict):
                # Check for our specific dataset structure: {"dataset_info": {}, "examples": [...]}
                if 'examples' in data:
                    dataset_examples = data['examples']
                elif 'train' in data:
                    # Dataset with train/val/test splits
                    split_name = 'validation' if self.validation_mode else 'train'
                    dataset_examples = data.get(split_name, data.get('train', []))
                else:
                    # Single example in a dict
                    dataset_examples = [data]
            elif isinstance(data, list):
                # Direct list of examples
                dataset_examples = data
            else:
                # Single example
                dataset_examples = [data]

            for idx, example in enumerate(dataset_examples):
                processed = self._process_example(example, idx)
                if processed:
                    examples.append(processed)

        return examples

    def _process_example(self, example: Dict[str, Any], idx: int) -> Optional[Dict[str, Any]]:
        """Process a single training example with three-tier format."""
        try:
            # Handle both old format and new three-tier format
            if 'complete_response' in example:
                # New format: complete_response contains thinking, pseudocode, computation
                complete_response = example['complete_response']

                # Extract thinking section
                thinking_match = re.search(r'<thinking>(.*?)</thinking>', complete_response, re.DOTALL)
                thinking_section = thinking_match.group(1).strip() if thinking_match else ""

                # Extract pseudocode section
                pseudocode_match = re.search(r'<pseudocode>(.*?)</pseudocode>', complete_response, re.DOTALL)
                pseudocode_section = pseudocode_match.group(1).strip() if pseudocode_match else ""

                # Extract computation section
                computation_match = re.search(r'<computation>(.*?)</computation>', complete_response, re.DOTALL)
                bytelogic_code = computation_match.group(1).strip() if computation_match else ""

                # Combine all sections for the full response
                full_response = ""
                if thinking_section:
                    full_response += f"<thinking>{thinking_section}</thinking>\n"
                if pseudocode_section:
                    full_response += f"<pseudocode>{pseudocode_section}</pseudocode>\n"
                if bytelogic_code:
                    full_response += f"<computation>{bytelogic_code}</computation>"

                input_text = example.get('user_query', '')
                output_text = full_response.strip()
            else:
                # Old format: input/output with computation
                input_text = example.get('input', '')
                output_text = example.get('output', '')

                # Extract ByteLogic computation if present
                computation_match = re.search(
                    r'<computation>\s*(.*?)\s*</computation>',
                    output_text,
                    re.DOTALL
                )

                bytelogic_code = None
                if computation_match:
                    bytelogic_code = computation_match.group(1).strip()

            if not input_text and not output_text:
                # Try alternative field names
                input_text = example.get('question', '') or example.get('query', '') or example.get('user_query', '')
                output_text = example.get('response', '') or example.get('answer', '') or example.get('complete_response', '')

                # Extract ByteLogic computation if present in alternative format
                if output_text:
                    computation_match = re.search(
                        r'<computation>\s*(.*?)\s*</computation>',
                        output_text,
                        re.DOTALL
                    )

                    if computation_match:
                        bytelogic_code = computation_match.group(1).strip()

            if not input_text or not output_text:
                logger.warning(f"Example {idx} missing input or output")
                return None

            # Validate ByteLogic syntax if tokenizer available
            if bytelogic_code and self.bytelogic_tokenizer:
                is_valid, error = self.bytelogic_tokenizer.validate_bytelogic_syntax(bytelogic_code)
                if not is_valid:
                    logger.warning(f"Example {idx} has invalid ByteLogic syntax: {error}")
                    # Don't skip invalid examples, just note them

            # Extract expected result from metadata if available
            metadata = example.get('metadata', {})
            expected_result = metadata.get('expected_result', [])

            return {
                "input": input_text,
                "output": output_text,
                "bytelogic_code": bytelogic_code,
                "expected_result": expected_result,
                "metadata": metadata,
                "category": metadata.get('category', 'unknown'),
                "difficulty": metadata.get('difficulty', 'unknown')
            }

        except Exception as e:
            logger.error(f"Error processing example {idx}: {e}")
            return None


class ThreeTierByteLogicCurriculumDataset(ThreeTierByteLogicDataset):
    """Three-tier dataset with curriculum learning for different phases."""
    
    def __init__(self, *args, curriculum_phase: str = "integrated", **kwargs):
        self.curriculum_phase = curriculum_phase
        super().__init__(*args, **kwargs)
        
        # Filter examples based on curriculum phase
        if curriculum_phase != "integrated":
            self.examples = self._filter_by_curriculum_phase(self.examples)
            logger.info(f"Filtered to {len(self.examples)} examples for phase '{curriculum_phase}'")
    
    def _filter_by_curriculum_phase(self, examples: List[Dict]) -> List[Dict]:
        """Filter examples based on curriculum phase."""
        if self.curriculum_phase == "abstract_planning":
            # Only include thinking section in output
            filtered_examples = []
            for ex in examples:
                if '<thinking>' in ex['output']:
                    # Extract only the thinking part
                    thinking_match = re.search(r'<thinking>(.*?)</thinking>', ex['output'], re.DOTALL)
                    if thinking_match:
                        new_ex = ex.copy()
                        new_ex['output'] = f"<thinking>{thinking_match.group(1)}</thinking>"
                        filtered_examples.append(new_ex)
            return filtered_examples
        elif self.curriculum_phase == "algorithm_design":
            # Only include thinking and pseudocode sections
            filtered_examples = []
            for ex in examples:
                # Check if the output contains the required tags
                has_thinking = '<thinking>' in ex['output']
                has_pseudocode = '<pseudocode>' in ex['output']

                if has_thinking and has_pseudocode:
                    # Extract thinking and pseudocode parts
                    thinking_match = re.search(r'<thinking>(.*?)</thinking>', ex['output'], re.DOTALL)
                    pseudocode_match = re.search(r'<pseudocode>(.*?)</pseudocode>', ex['output'], re.DOTALL)

                    new_output = ""
                    if thinking_match:
                        new_output += f"<thinking>{thinking_match.group(1)}</thinking>\n"
                    if pseudocode_match:
                        new_output += f"<pseudocode>{pseudocode_match.group(1)}</pseudocode>"

                    if new_output.strip():
                        new_ex = ex.copy()
                        new_ex['output'] = new_output.strip()
                        filtered_examples.append(new_ex)
            return filtered_examples
        elif self.curriculum_phase == "implementation_translation":
            # Only include pseudocode and computation sections
            filtered_examples = []
            for ex in examples:
                # Check if the output contains the required tags
                has_pseudocode = '<pseudocode>' in ex['output']
                has_computation = '<computation>' in ex['output']

                if has_pseudocode and has_computation:
                    # Extract pseudocode and computation parts
                    pseudocode_match = re.search(r'<pseudocode>(.*?)</pseudocode>', ex['output'], re.DOTALL)
                    computation_match = re.search(r'<computation>(.*?)</computation>', ex['output'], re.DOTALL)

                    new_output = ""
                    if pseudocode_match:
                        new_output += f"<pseudocode>{pseudocode_match.group(1)}</pseudocode>\n"
                    if computation_match:
                        new_output += f"<computation>{computation_match.group(1)}</computation>"

                    if new_output.strip():
                        new_ex = ex.copy()
                        new_ex['output'] = new_output.strip()
                        filtered_examples.append(new_ex)
            return filtered_examples
        elif self.curriculum_phase == "integrated":
            # All sections included
            return examples
        else:
            # Unknown phase, return all
            return examples


class ByteLogicComputationTrainer:
    """Trainer for ByteLogic-integrated WorldModel with curriculum learning."""

    def __init__(self,
                 model_path: str,
                 dataset_path: str,
                 output_dir: str,
                 curriculum_phase: str = "integrated",  # Options: abstract_planning, algorithm_design, implementation_translation, integrated
                 computation_layers: List[int] = [3, 7, 11, 15, 19, 23, 27],
                 learning_rate: float = 1e-5,
                 batch_size: int = 2,
                 max_length: int = 512):
        """
        Initialize integrated trainer with curriculum learning.

        Args:
            model_path: Path to base model
            dataset_path: Path to ByteLogic training dataset
            output_dir: Output directory for checkpoints
            curriculum_phase: Which phase of curriculum to train on
            computation_layers: Layers for ByteLogic computation
            learning_rate: Training learning rate
            batch_size: Training batch size
            max_length: Maximum sequence length
        """
        self.model_path = model_path
        self.dataset_path = dataset_path
        self.output_dir = Path(output_dir)
        self.curriculum_phase = curriculum_phase
        self.computation_layers = computation_layers
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_length = max_length

        # Initialize failure tracking
        self.total_executions = 0
        self.failed_executions = 0

        # Create output directory
        self.output_dir.mkdir(exist_ok=True)

        # Initialize components
        self.device = device
        self._setup_model()
        self._setup_datasets()
        self._setup_optimizer()

        logger.info(f"🏗️ Integrated trainer initialized")
        logger.info(f"   Curriculum phase: {curriculum_phase}")
        logger.info(f"   Computation layers: {computation_layers}")
        logger.info(f"   Training with execution: True")

    def get_execution_stats(self) -> str:
        """Get execution statistics."""
        if self.total_executions == 0:
            return "0/0"
        return f"{self.failed_executions}/{self.total_executions}"

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
        """Load and setup training datasets with curriculum support."""
        if self.dataset_path.lower() in ["auto", "all"]:
            logger.info(f"📖 Loading ALL ByteLogic datasets automatically")

            # Import and use the function directly
            import importlib.util
            from pathlib import Path
            tools_path = Path(__file__).parent / "tools" / "multi_dataset_loader.py"
            spec = importlib.util.spec_from_file_location("multi_dataset_loader_helper", tools_path)
            multi_loader_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(multi_loader_module)

            # Use the function
            all_examples = multi_loader_module.load_all_datasets()

            # Split into train and validation sets
            split_idx = int(len(all_examples) * 0.8)
            train_examples = all_examples[:split_idx]
            val_examples = all_examples[split_idx:]

            logger.info(f"   Total loaded examples: {len(all_examples)}")
            logger.info(f"   Training examples: {len(train_examples)}")
            logger.info(f"   Validation examples: {len(val_examples)}")

            # We need to create a custom dataset implementation since we can't pass examples directly
            # Let's create the datasets with temporary files
            import tempfile
            import json

            # Create temporary files for the train and val data
            with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json', dir='/tmp') as f:
                json.dump({"train": train_examples}, f)
                temp_train_file = f.name

            with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json', dir='/tmp') as f:
                json.dump({"train": val_examples}, f)
                temp_val_file = f.name

            try:
                # Create ThreeTierByteLogic datasets using the temporary files
                self.train_dataset = ThreeTierByteLogicCurriculumDataset(
                    data_file=temp_train_file,
                    text_tokenizer=self.text_tokenizer,
                    max_length=self.max_length,
                    validation_mode=False,
                    curriculum_phase=self.curriculum_phase
                )
                self.val_dataset = ThreeTierByteLogicCurriculumDataset(
                    data_file=temp_val_file,
                    text_tokenizer=self.text_tokenizer,
                    max_length=self.max_length,
                    validation_mode=True,
                    curriculum_phase=self.curriculum_phase
                )
            finally:
                # Clean up temporary files
                import os
                os.remove(temp_train_file)
                os.remove(temp_val_file)

        else:
            logger.info(f"📖 Loading ByteLogic datasets from {self.dataset_path}")
            
            # Check if this is a three-tier dataset by looking at the content
            with open(self.dataset_path, 'r', encoding='utf-8') as f:
                sample_data = f.read(2000)  # Read first 2000 chars to check format
            
            if '"complete_response"' in sample_data:
                # This is a three-tier dataset
                logger.info(f"   Detected three-tier dataset format for phase: {self.curriculum_phase}")
                
                # Load training dataset using the three-tier approach
                self.train_dataset = ThreeTierByteLogicCurriculumDataset(
                    data_file=self.dataset_path,
                    text_tokenizer=self.text_tokenizer,
                    max_length=self.max_length,
                    validation_mode=False,
                    curriculum_phase=self.curriculum_phase
                )

                # Load validation dataset
                self.val_dataset = ThreeTierByteLogicCurriculumDataset(
                    data_file=self.dataset_path,
                    text_tokenizer=self.text_tokenizer,
                    max_length=self.max_length,
                    validation_mode=True,
                    curriculum_phase=self.curriculum_phase
                )
            else:
                # This is the original format
                logger.info(f"   Using original dataset format")
                
                # Load training dataset using the original approach
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
        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self._collate_fn
        )

        self.val_loader = torch.utils.data.DataLoader(
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
            # Track execution successes/failures for statistics
            for result in execution_results:
                if result.get('executed', False):  # Only track if actually executed
                    self.total_executions += 1
                    if not result.get('success', False):
                        self.failed_executions += 1
                        # Small penalty for failed execution
                        computation_loss += 0.1
                    elif result.get('success', False):
                        # Small reward for successful execution
                        computation_loss -= 0.1

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

        # For progress tracking
        import time
        start_time = time.time()
        total_batches = len(self.train_loader)

        logger.info(f"🏋️ Training epoch {epoch + 1} - {total_batches} batches")
        logger.info(f"   Curriculum phase: {self.curriculum_phase}")

        for batch_idx, batch in enumerate(self.train_loader):
            batch_start_time = time.time()

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

            # Calculate timing stats
            batch_time = time.time() - batch_start_time
            elapsed_time = time.time() - start_time
            avg_time_per_batch = elapsed_time / (batch_idx + 1)
            remaining_batches = total_batches - (batch_idx + 1)
            eta_seconds = remaining_batches * avg_time_per_batch

            # Calculate progress percentage
            progress_pct = ((batch_idx + 1) / total_batches) * 100

            # Get execution failure stats
            fail_stats = self.get_execution_stats()

            # Update progress line (separate from logging)
            prog_line = f"\r  Epoch {epoch + 1} Batch {batch_idx + 1}/{total_batches} | {progress_pct:.1f}% | loss={loss.item():.4f} | Fails: {fail_stats} | {batch_time:.2f}s/batch | ETA: {eta_seconds/60:.1f}m"
            print(prog_line, end="", flush=True)

            # Log progress every 50 batches for detailed tracking
            if (batch_idx + 1) % 50 == 0:
                # Add a newline before log message to separate from progress
                print()  # newline to separate from progress line
                logger.info(f"  Batch {batch_idx + 1}/{total_batches}: loss={loss.item():.4f}, lm_loss={lm_loss.item():.4f}, Fails: {fail_stats}, elapsed={elapsed_time/60:.1f}min")
                # Re-print progress line after log
                print(prog_line, end="", flush=True)

        # Print a newline after progress bar
        print()

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

        # For validation progress tracking
        import time
        start_time = time.time()
        total_batches = len(self.val_loader)

        logger.info(f"🔍 Validation - {total_batches} batches")
        logger.info(f"   Curriculum phase: {self.curriculum_phase}")

        with torch.no_grad():
            for batch_idx, batch in enumerate(self.val_loader):
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

                # Calculate validation timing stats
                elapsed_time = time.time() - start_time
                avg_time_per_batch = elapsed_time / (batch_idx + 1)
                remaining_batches = total_batches - (batch_idx + 1)
                eta_seconds = remaining_batches * avg_time_per_batch

                # Calculate progress percentage
                progress_pct = ((batch_idx + 1) / total_batches) * 100

                # Get execution failure stats
                fail_stats = self.get_execution_stats()

                # Update progress line for validation (separate from logging)
                prog_line = f"\r  Validation Batch {batch_idx + 1}/{total_batches} | {progress_pct:.1f}% | loss={loss.item():.4f} | Fails: {fail_stats} | ETA: {eta_seconds/60:.1f}m"
                print(prog_line, end="", flush=True)

        # Print newline after validation progress bar
        print()

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
        checkpoint_path = self.output_dir / f"checkpoint_epoch_{epoch + 1}_phase_{self.curriculum_phase}.pt"

        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics,
            'computation_layers': self.computation_layers,
            'curriculum_phase': self.curriculum_phase
        }, checkpoint_path)

        logger.info(f"💾 Saved checkpoint: {checkpoint_path}")

    def train(self, epochs: int = 5):
        """Run complete integrated training."""
        logger.info(f"🚀 Starting integrated ByteLogic training")
        logger.info(f"   Curriculum phase: {self.curriculum_phase}")
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
        final_path = self.output_dir / f"final_integrated_model_phase_{self.curriculum_phase}.pt"
        torch.save(self.model.state_dict(), final_path)

        logger.info(f"🎉 Training completed!")
        logger.info(f"   Final model: {final_path}")
        logger.info(f"   Best validation loss: {best_val_loss:.4f}")


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="ByteLogic Integrated WorldModel Training with Curriculum Learning")
    parser.add_argument("--model", required=True, help="Path to base model")
    parser.add_argument("--dataset", default="auto", help="Path to ByteLogic dataset (use 'auto' or 'all' to load all datasets from training/datasets/, DEFAULT: auto)")
    parser.add_argument("--output_dir", default="integrated_worldmodel_output", help="Output directory")
    parser.add_argument("--curriculum-phase", dest="curriculum_phase", default="integrated", 
                        choices=["abstract_planning", "algorithm_design", "implementation_translation", "integrated"],
                        help="Curriculum learning phase: abstract_planning, algorithm_design, implementation_translation, or integrated (DEFAULT: integrated)")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--computation_layers", nargs='+', type=int, default=[3, 7, 11, 15, 19, 23, 27], help="Computation layer indices")

    args = parser.parse_args()

    logger.info(f"🎯 Integrated ByteLogic WorldModel Training with Curriculum Learning")
    logger.info(f"   Model: {args.model}")
    if args.dataset.lower() in ['auto', 'all']:
        logger.info(f"   Dataset: Loading ALL datasets automatically from training/datasets/ (DEFAULT)")
    else:
        logger.info(f"   Dataset: {args.dataset}")
    logger.info(f"   Curriculum phase: {args.curriculum_phase}")
    logger.info(f"   Output: {args.output_dir}")
    logger.info(f"   Computation layers: {args.computation_layers}")

    # Initialize trainer
    trainer = ByteLogicComputationTrainer(
        model_path=args.model,
        dataset_path=args.dataset,
        output_dir=args.output_dir,
        curriculum_phase=args.curriculum_phase,
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