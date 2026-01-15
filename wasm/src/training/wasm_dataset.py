"""
WASM-Aware Dataset
=================

Dataset class for training multimodal text+WASM models.
Handles tokenization of both text and WAT code streams.
"""

import torch
from torch.utils.data import Dataset
from typing import List, Dict, Optional, Any
import json
import re
from transformers import AutoTokenizer

from ..tokenization.wat_tokenizer import WATTokenizer


class WASMModalDataset(Dataset):
    """Dataset for training text+WASM multimodal models."""
    
    def __init__(
        self,
        data_file: str,
        text_tokenizer: AutoTokenizer,
        wasm_tokenizer: WATTokenizer,
        max_text_length: int = 512,
        max_wasm_length: int = 256,
        include_execution_results: bool = True
    ):
        self.text_tokenizer = text_tokenizer
        self.wasm_tokenizer = wasm_tokenizer
        self.max_text_length = max_text_length
        self.max_wasm_length = max_wasm_length
        self.include_execution_results = include_execution_results
        
        # Load and process data
        self.examples = self._load_examples(data_file)
        print(f"Loaded {len(self.examples)} WASM training examples")
    
    def _load_examples(self, data_file: str) -> List[Dict[str, Any]]:
        """Load examples from training file."""
        examples = []
        
        with open(data_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Split by double newlines to get individual examples
        raw_examples = content.split('\n\n')
        
        for raw_example in raw_examples:
            if not raw_example.strip():
                continue
            
            example = self._parse_example(raw_example.strip())
            if example:
                examples.append(example)
        
        return examples
    
    def _parse_example(self, example_text: str) -> Optional[Dict[str, Any]]:
        """Parse a single training example."""
        try:
            # Extract components
            user_match = re.search(r'User:\s*(.*?)(?=Assistant:|$)', example_text, re.DOTALL)
            assistant_match = re.search(r'Assistant:\s*(.*?)(?=$)', example_text, re.DOTALL)
            
            if not user_match or not assistant_match:
                return None
            
            user_text = user_match.group(1).strip()
            assistant_text = assistant_match.group(1).strip()
            
            # Extract WASM model
            wat_match = re.search(r'<wat_model>\s*(.*?)\s*</wat_model>', assistant_text, re.DOTALL)
            wat_code = wat_match.group(1).strip() if wat_match else None
            
            # Extract computed result
            computed_match = re.search(r'<computed>(.*?)</computed>', assistant_text)
            computed_result = computed_match.group(1).strip() if computed_match else None
            
            # Extract think block
            think_match = re.search(r'<think>(.*?)</think>', assistant_text, re.DOTALL)
            think_text = think_match.group(1).strip() if think_match else ""
            
            # Extract requires block
            requires_match = re.search(r'<requires>(.*?)</requires>', assistant_text)
            requires_text = requires_match.group(1).strip() if requires_match else ""
            
            # Extract final answer (text after all blocks)
            final_answer = self._extract_final_answer(assistant_text)
            
            return {
                "user_text": user_text,
                "think_text": think_text,
                "wat_code": wat_code,
                "computed_result": computed_result,
                "requires": requires_text,
                "final_answer": final_answer,
                "full_text": example_text
            }
            
        except Exception as e:
            print(f"Warning: Failed to parse example: {e}")
            return None
    
    def _extract_final_answer(self, assistant_text: str) -> str:
        """Extract final answer after all special blocks."""
        # Remove all special blocks
        cleaned = assistant_text
        
        # Remove blocks in order
        cleaned = re.sub(r'<think>.*?</think>', '', cleaned, flags=re.DOTALL)
        cleaned = re.sub(r'<wat_model>.*?</wat_model>', '', cleaned, flags=re.DOTALL)
        cleaned = re.sub(r'<computed>.*?</computed>', '', cleaned)
        cleaned = re.sub(r'<requires>.*?</requires>', '', cleaned)
        
        # Clean up whitespace
        lines = [line.strip() for line in cleaned.split('\n') if line.strip()]
        return '\n'.join(lines)
    
    def __len__(self) -> int:
        return len(self.examples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a training example."""
        example = self.examples[idx]
        
        # Format text sequence
        text_sequence = self._format_text_sequence(example)
        
        # Tokenize text
        text_encoding = self.text_tokenizer(
            text_sequence,
            max_length=self.max_text_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # Prepare WASM tokens if available
        wasm_input_ids = None
        wasm_attention_mask = None
        
        if example["wat_code"]:
            wasm_tokens = self.wasm_tokenizer.encode_wat(example["wat_code"])
            
            # Pad/truncate WASM tokens
            if len(wasm_tokens) > self.max_wasm_length:
                wasm_tokens = wasm_tokens[:self.max_wasm_length]
            else:
                # Pad with pad token
                padding_length = self.max_wasm_length - len(wasm_tokens)
                wasm_tokens.extend([self.wasm_tokenizer.pad_token_id] * padding_length)
            
            wasm_input_ids = torch.tensor(wasm_tokens, dtype=torch.long)
            wasm_attention_mask = torch.tensor(
                [1 if token_id != self.wasm_tokenizer.pad_token_id else 0 for token_id in wasm_tokens],
                dtype=torch.long
            )
        
        # Create labels for text generation (same as input_ids for causal LM)
        labels = text_encoding["input_ids"].clone()
        
        # Mask user input in labels (only train on assistant response)
        user_end_pos = self._find_assistant_start_position(text_sequence)
        if user_end_pos > 0:
            labels[0, :user_end_pos] = -100
        
        result = {
            "input_ids": text_encoding["input_ids"].squeeze(),
            "attention_mask": text_encoding["attention_mask"].squeeze(),
            "labels": labels.squeeze(),
        }
        
        # Add WASM components if available
        if wasm_input_ids is not None:
            result["wasm_input_ids"] = wasm_input_ids
            result["wasm_attention_mask"] = wasm_attention_mask
        
        # Add execution target if available
        if example["computed_result"] and self.include_execution_results:
            try:
                execution_target = float(example["computed_result"])
                result["execution_target"] = torch.tensor(execution_target, dtype=torch.float32)
            except (ValueError, TypeError):
                # Non-numeric result, encode as text
                result["execution_target_text"] = example["computed_result"]
        
        return result
    
    def _format_text_sequence(self, example: Dict[str, Any]) -> str:
        """Format example as text sequence for training."""
        # Build the sequence
        sequence_parts = [f"User: {example['user_text']}"]
        
        # Add assistant response
        assistant_parts = []
        
        if example["think_text"]:
            assistant_parts.append(f"<think>{example['think_text']}</think>")
        
        if example["wat_code"]:
            assistant_parts.append(f"<wat_model>\n{example['wat_code']}\n</wat_model>")
        
        if example["computed_result"]:
            assistant_parts.append(f"<computed>{example['computed_result']}</computed>")
        
        if example["requires"]:
            assistant_parts.append(f"<requires>{example['requires']}</requires>")
        
        if example["final_answer"]:
            assistant_parts.append(example["final_answer"])
        
        if assistant_parts:
            sequence_parts.append("Assistant: " + "\n".join(assistant_parts))
        
        return "\n".join(sequence_parts)
    
    def _find_assistant_start_position(self, text_sequence: str) -> int:
        """Find token position where assistant response starts."""
        assistant_pos = text_sequence.find("Assistant:")
        if assistant_pos == -1:
            return 0
        
        # Tokenize up to assistant position to get token count
        prefix = text_sequence[:assistant_pos + len("Assistant:")]
        prefix_tokens = self.text_tokenizer.encode(prefix, add_special_tokens=False)
        return len(prefix_tokens)


class WASMCurriculumDataset:
    """Manages curriculum learning across different training stages."""
    
    def __init__(
        self,
        data_dir: str,
        text_tokenizer: AutoTokenizer,
        wasm_tokenizer: WATTokenizer,
        **dataset_kwargs
    ):
        self.data_dir = data_dir
        self.text_tokenizer = text_tokenizer
        self.wasm_tokenizer = wasm_tokenizer
        self.dataset_kwargs = dataset_kwargs
        
        # Initialize stage datasets
        self.stages = {
            "basic_arithmetic": None,
            "system_operations": None,
            "complex_logic": None
        }
        
        self._load_all_stages()
    
    def _load_all_stages(self):
        """Load all curriculum stages."""
        import os
        
        stage_files = {
            "basic_arithmetic": "basic_arithmetic_training.txt",
            "system_operations": "system_operations_training.txt", 
            "complex_logic": "complex_logic_training.txt"
        }
        
        for stage_name, filename in stage_files.items():
            filepath = os.path.join(self.data_dir, filename)
            if os.path.exists(filepath):
                try:
                    dataset = WASMModalDataset(
                        filepath,
                        self.text_tokenizer,
                        self.wasm_tokenizer,
                        **self.dataset_kwargs
                    )
                    self.stages[stage_name] = dataset
                    print(f"✅ Loaded {stage_name}: {len(dataset)} examples")
                except Exception as e:
                    print(f"❌ Failed to load {stage_name}: {e}")
            else:
                print(f"⚠️ Missing file: {filepath}")
    
    def get_stage_dataset(self, stage_name: str) -> Optional[WASMModalDataset]:
        """Get dataset for a specific curriculum stage."""
        return self.stages.get(stage_name)
    
    def get_combined_dataset(self, stages: Optional[List[str]] = None) -> WASMModalDataset:
        """Get combined dataset from multiple stages."""
        if stages is None:
            stages = ["basic_arithmetic", "system_operations", "complex_logic"]
        
        # Combine examples from specified stages
        all_examples = []
        for stage_name in stages:
            stage_dataset = self.stages.get(stage_name)
            if stage_dataset:
                all_examples.extend(stage_dataset.examples)
        
        # Create combined dataset
        combined_dataset = WASMModalDataset.__new__(WASMModalDataset)
        combined_dataset.text_tokenizer = self.text_tokenizer
        combined_dataset.wasm_tokenizer = self.wasm_tokenizer
        combined_dataset.max_text_length = self.dataset_kwargs.get("max_text_length", 512)
        combined_dataset.max_wasm_length = self.dataset_kwargs.get("max_wasm_length", 256)
        combined_dataset.include_execution_results = self.dataset_kwargs.get("include_execution_results", True)
        combined_dataset.examples = all_examples
        
        print(f"✅ Combined dataset: {len(all_examples)} examples from stages: {', '.join(stages)}")
        return combined_dataset