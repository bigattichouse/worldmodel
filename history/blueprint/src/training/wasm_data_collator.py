"""
WASM Data Collator
=================

Custom data collator for text+WASM multimodal training.
Handles batching of both text and WASM token streams.
"""

import torch
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from transformers import DataCollatorForLanguageModeling


@dataclass
class WASMDataCollator:
    """Data collator for WASM multimodal training."""
    
    text_tokenizer: Any
    wasm_tokenizer: Any
    mlm: bool = False
    mlm_probability: float = 0.15
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"
    
    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Collate examples into a batch."""
        batch_size = len(examples)
        
        # Separate text and WASM components
        text_batch = []
        wasm_batch = []
        execution_targets = []
        
        has_wasm = any("wasm_input_ids" in ex for ex in examples)
        has_execution_targets = any("execution_target" in ex for ex in examples)
        
        for example in examples:
            # Text components (required)
            text_batch.append({
                "input_ids": example["input_ids"],
                "attention_mask": example["attention_mask"],
                "labels": example["labels"]
            })
            
            # WASM components (optional)
            if has_wasm:
                if "wasm_input_ids" in example:
                    wasm_batch.append({
                        "input_ids": example["wasm_input_ids"],
                        "attention_mask": example["wasm_attention_mask"]
                    })
                else:
                    # Create empty WASM sequence for consistency
                    max_wasm_length = 256  # Default WASM sequence length
                    wasm_batch.append({
                        "input_ids": torch.full((max_wasm_length,), self.wasm_tokenizer.pad_token_id, dtype=torch.long),
                        "attention_mask": torch.zeros(max_wasm_length, dtype=torch.long)
                    })
            
            # Execution targets (optional)
            if has_execution_targets:
                if "execution_target" in example:
                    execution_targets.append(example["execution_target"])
                else:
                    execution_targets.append(torch.tensor(0.0, dtype=torch.float32))  # Default value
        
        # Batch text components
        text_collated = self._collate_text_batch(text_batch)
        
        # Start with text batch
        batch = {
            "input_ids": text_collated["input_ids"],
            "attention_mask": text_collated["attention_mask"],
            "labels": text_collated["labels"]
        }
        
        # Add WASM components if present
        if has_wasm and wasm_batch:
            wasm_collated = self._collate_wasm_batch(wasm_batch)
            batch["wasm_input_ids"] = wasm_collated["input_ids"]
            batch["wasm_attention_mask"] = wasm_collated["attention_mask"]
        
        # Add execution targets if present
        if has_execution_targets and execution_targets:
            batch["execution_targets"] = torch.stack(execution_targets)
        
        return batch
    
    def _collate_text_batch(self, text_batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Collate text components using standard language modeling collator."""
        # Use standard collator for text
        standard_collator = DataCollatorForLanguageModeling(
            tokenizer=self.text_tokenizer,
            mlm=self.mlm,
            mlm_probability=self.mlm_probability,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors
        )
        
        return standard_collator(text_batch)
    
    def _collate_wasm_batch(self, wasm_batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Collate WASM components."""
        # Extract sequences
        input_ids = [item["input_ids"] for item in wasm_batch]
        attention_masks = [item["attention_mask"] for item in wasm_batch]
        
        # Stack tensors (assuming they're already padded to same length)
        return {
            "input_ids": torch.stack(input_ids),
            "attention_mask": torch.stack(attention_masks)
        }