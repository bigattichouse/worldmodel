"""
ByteLogic Training Dataset
=========================

Dataset class for training ByteLogic-enabled models.
Handles tokenization and processing of computation tokens.
"""

import torch
from torch.utils.data import Dataset
from typing import List, Dict, Optional, Any, Tuple
import json
import re
import logging
from transformers import AutoTokenizer

# Try to import ByteLogic tokenizer, fallback to simple tokenization
try:
    from ..tokenization.bytelogic_tokenizer import ByteLogicTokenizer
except ImportError:
    ByteLogicTokenizer = None

logger = logging.getLogger(__name__)


class ByteLogicDataset(Dataset):
    """Dataset for training ByteLogic-enabled models."""
    
    def __init__(
        self,
        data_file: str,
        text_tokenizer: AutoTokenizer,
        max_length: int = 1024,
        include_computation_details: bool = True,
        validation_mode: bool = False
    ):
        self.text_tokenizer = text_tokenizer
        self.max_length = max_length
        self.include_computation_details = include_computation_details
        self.validation_mode = validation_mode
        
        # Initialize ByteLogic tokenizer if available
        self.bytelogic_tokenizer = ByteLogicTokenizer() if ByteLogicTokenizer else None
        
        # Add special tokens for computation
        self._add_special_tokens()
        
        # Load examples
        self.examples = self._load_examples(data_file)
        logger.info(f"Loaded {len(self.examples)} ByteLogic training examples")
    
    def _add_special_tokens(self):
        """Add ByteLogic-specific special tokens."""
        special_tokens = [
            "<computation>", "</computation>",
            "<result>", "</result>",
            # ByteLogic keywords
            "REL", "FACT", "RULE", "SCAN", "JOIN", "EMIT", "SOLVE", "QUERY",
            "CALC", "INPUT", "LET", "RESULT", "IF", "THEN", "ELSE", "END",
            "FOR", "WHILE", "IN", "RANGE", "LENGTH", "CHAR_AT", "BREAK", "CONTINUE",
            # Mathematical functions
            "POW", "ABS", "MIN", "MAX", "SQRT", "SIN", "COS", "TAN",
            "LOG", "EXP", "CEIL", "FLOOR"
        ]
        
        # Add tokens if they don't exist
        new_tokens = []
        for token in special_tokens:
            if token not in self.text_tokenizer.get_vocab():
                new_tokens.append(token)
        
        if new_tokens:
            self.text_tokenizer.add_tokens(new_tokens)
            logger.info(f"Added {len(new_tokens)} new ByteLogic tokens to tokenizer")
    
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
            
            if isinstance(data, dict) and 'train' in data:
                # Dataset with train/val/test splits
                split_name = 'validation' if self.validation_mode else 'train'
                dataset_examples = data.get(split_name, data.get('train', []))
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
        """Process a single training example."""
        try:
            # Extract input and output
            input_text = example.get('input', '')
            output_text = example.get('output', '')
            
            if not input_text or not output_text:
                logger.warning(f"Example {idx} missing input or output")
                return None
            
            # Extract ByteLogic computation if present
            computation_match = re.search(
                r'<computation>\s*(.*?)\s*</computation>', 
                output_text, 
                re.DOTALL
            )
            
            bytelogic_code = None
            if computation_match:
                bytelogic_code = computation_match.group(1).strip()
                
                # Validate ByteLogic syntax if tokenizer available
                if self.bytelogic_tokenizer:
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
    
    def __len__(self) -> int:
        return len(self.examples)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a training example."""
        example = self.examples[idx]
        
        # Create input-output pair for language modeling
        input_text = example["input"]
        output_text = example["output"]
        
        # Format as conversation
        conversation = f"User: {input_text}\nAssistant: {output_text}"
        
        # Tokenize
        encoding = self.text_tokenizer(
            conversation,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Create labels (same as input_ids for language modeling)
        labels = encoding['input_ids'].clone()
        
        # Mask the user input part in labels (only learn to generate assistant response)
        user_part = f"User: {input_text}\nAssistant: "
        user_encoding = self.text_tokenizer(
            user_part,
            max_length=self.max_length,
            truncation=True
        )
        user_length = len(user_encoding['input_ids'])
        
        # Set labels to -100 for user input part (ignore in loss)
        if user_length < labels.shape[-1]:
            labels[0, :user_length] = -100
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': labels.squeeze(0),
            'bytelogic_code': example.get("bytelogic_code", ""),
            'category': example.get("category", "unknown"),
            'difficulty': example.get("difficulty", "unknown")
        }


# Curriculum learning dataset
class ByteLogicCurriculumDataset(ByteLogicDataset):
    """ByteLogic dataset with curriculum learning."""
    
    def __init__(self, *args, curriculum_stage: str = "all", **kwargs):
        self.curriculum_stage = curriculum_stage
        super().__init__(*args, **kwargs)
        
        # Filter examples based on curriculum stage
        if curriculum_stage != "all":
            self.examples = self._filter_by_curriculum(self.examples)
            logger.info(f"Filtered to {len(self.examples)} examples for stage '{curriculum_stage}'")
    
    def _filter_by_curriculum(self, examples: List[Dict]) -> List[Dict]:
        """Filter examples based on curriculum stage."""
        if self.curriculum_stage == "basic":
            # Only basic examples
            return [ex for ex in examples if ex.get("difficulty") == "beginner"]
        elif self.curriculum_stage == "intermediate":
            # Basic + intermediate
            return [ex for ex in examples if ex.get("difficulty") in ["beginner", "intermediate"]]
        elif self.curriculum_stage == "advanced":
            # All examples
            return examples
        else:
            # Unknown stage, return all
            return examples


def load_bytelogic_dataset(
    data_file: str,
    tokenizer: AutoTokenizer,
    max_length: int = 1024,
    curriculum_stage: str = "all",
    validation_mode: bool = False
) -> ByteLogicDataset:
    """Convenience function to load ByteLogic dataset."""
    
    if curriculum_stage != "all":
        return ByteLogicCurriculumDataset(
            data_file=data_file,
            text_tokenizer=tokenizer,
            max_length=max_length,
            curriculum_stage=curriculum_stage,
            validation_mode=validation_mode
        )
    else:
        return ByteLogicDataset(
            data_file=data_file,
            text_tokenizer=tokenizer,
            max_length=max_length,
            validation_mode=validation_mode
        )