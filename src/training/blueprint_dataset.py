"""
BluePrint Training Dataset
=========================

Dataset class for training BluePrint methodology models.
Handles tokenization and processing of thinking → blueprint patterns.
"""

import torch
from torch.utils.data import Dataset
from typing import List, Dict, Optional, Any, Tuple
import json
import re
import logging
import glob
from pathlib import Path
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)


class BluePrintDatasetScanner:
    """Scans training/datasets/ directory and loads all JSONL files."""
    
    def __init__(self, datasets_dir: str = "training/datasets"):
        self.datasets_dir = Path(datasets_dir)
        
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
        """Load examples from a JSONL file. Handles both structured and simple text formats."""
        examples = []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line or line.startswith('#'):  # Skip empty lines and comments
                    continue
                    
                try:
                    example = json.loads(line)
                    
                    # Handle structured format: {id, user_query, response}
                    structured_fields = ['id', 'user_query', 'response']
                    if all(field in example for field in structured_fields):
                        # Validate BluePrint format
                        if self._validate_blueprint_format(example['response']):
                            examples.append(example)
                        continue
                    
                    # Handle simple text format: {"text": "User: ... <thinking>..."}
                    if 'text' in example:
                        text = example['text']
                        # Check if it's a valid blueprint format with User: prefix
                        if self._validate_blueprint_format(text) and text.startswith('User:'):
                            # Convert to structured format
                            try:
                                # Handle format: "User: query\n\n<thinking>...<blueprint>..."
                                if 'Assistant:' in text:
                                    # Format: "User: ... Assistant: ..."
                                    parts = text.split('Assistant:', 1)
                                    user_part = parts[0].replace('User:', '').strip()
                                    response_part = parts[1].strip()
                                else:
                                    # Format: "User: query\n\n<thinking>..." (no explicit Assistant:)
                                    user_match = text.split('\n\n', 1)
                                    if len(user_match) == 2:
                                        user_part = user_match[0].replace('User:', '').strip()
                                        response_part = user_match[1].strip()
                                    else:
                                        continue
                                
                                converted_example = {
                                    'id': f"converted_{len(examples)}",
                                    'user_query': user_part,
                                    'response': response_part,
                                    'category': file_path.parent.name
                                }
                                examples.append(converted_example)
                            except Exception:
                                continue  # Skip conversion errors
                        continue
                        
                except json.JSONDecodeError as e:
                    # Skip malformed JSON silently to reduce noise
                    continue
                except Exception as e:
                    # Skip any other errors silently
                    continue
        
        return examples
    
    def _validate_blueprint_format(self, response: str) -> bool:
        """Validate that response contains proper <thinking> and <blueprint> tags."""
        has_thinking = '<thinking>' in response and '</thinking>' in response
        has_blueprint = '<blueprint>' in response and '</blueprint>' in response
        return has_thinking and has_blueprint


class BluePrintDataset(Dataset):
    """Dataset for BluePrint training with thinking → blueprint pattern."""
    
    def __init__(self, examples: List[Dict], tokenizer: AutoTokenizer, max_length: int = 768):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Add special tokens for BluePrint methodology
        self._add_special_tokens()
        
        logger.info(f"📚 BluePrintDataset initialized with {len(examples)} examples")
        logger.info(f"   Max length: {max_length} tokens")
    
    def _add_special_tokens(self):
        """Add BluePrint-specific special tokens."""
        special_tokens = [
            '<thinking>', '</thinking>',
            '<blueprint>', '</blueprint>'
        ]
        
        # Add tokens if they don't exist
        new_tokens = []
        for token in special_tokens:
            if token not in self.tokenizer.get_vocab():
                new_tokens.append(token)
        
        if new_tokens:
            self.tokenizer.add_tokens(new_tokens)
            logger.info(f"Added {len(new_tokens)} BluePrint tokens to tokenizer")
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        example = self.examples[idx]
        
        # Format the training text as conversation
        conversation = f"User: {example['user_query']}\n\nAssistant: {example['response']}{self.tokenizer.eos_token}"
        
        # Tokenize
        encoding = self.tokenizer(
            conversation,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Create labels (same as input_ids for causal LM)
        labels = encoding['input_ids'].clone()
        
        # Mask the user input part in labels (only learn to generate assistant response)
        user_part = f"User: {example['user_query']}\n\nAssistant: "
        user_encoding = self.tokenizer(
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
            'labels': labels.squeeze(0)
        }


class BluePrintCurriculumDataset(BluePrintDataset):
    """BluePrint dataset with curriculum learning stages."""
    
    def __init__(self, examples: List[Dict], tokenizer: AutoTokenizer, max_length: int = 768, curriculum_stage: str = "all"):
        self.curriculum_stage = curriculum_stage
        
        # Filter examples based on curriculum stage
        if curriculum_stage != "all":
            examples = self._filter_by_curriculum(examples)
            logger.info(f"Filtered to {len(examples)} examples for curriculum stage '{curriculum_stage}'")
        
        super().__init__(examples, tokenizer, max_length)
    
    def _filter_by_curriculum(self, examples: List[Dict]) -> List[Dict]:
        """
        Filter examples based on curriculum stage following docs/blueprint-test-catalog.md.
        
        Progressive curriculum:
        - foundation: Basic CRUD, simple calculations, data conversions (60 examples)
        - business: Add business logic patterns - financial, e-commerce, HR (80 examples) 
        - technical: Add authentication, data processing, infrastructure (80 examples)
        - domain: Add healthcare, education, logistics, manufacturing (60 examples)
        - advanced: Add microservices, event-driven, architectural patterns (40+ examples)
        - security: Add security patterns and validation edge cases
        """
        
        stage_categories = {
            "foundation": ["basic_systems"],
            "business": ["basic_systems", "business_logic"], 
            "technical": ["basic_systems", "business_logic", "technical_systems"],
            "domain": ["basic_systems", "business_logic", "technical_systems", "domain_specific"],
            "advanced": ["basic_systems", "business_logic", "technical_systems", "domain_specific", "advanced_patterns"],
            "security": ["basic_systems", "business_logic", "technical_systems", "domain_specific", "advanced_patterns", "security_patterns"],
            "complete": ["basic_systems", "business_logic", "technical_systems", "domain_specific", "advanced_patterns", "security_patterns", "validation"]
        }
        
        if self.curriculum_stage not in stage_categories:
            return examples
        
        allowed_categories = stage_categories[self.curriculum_stage]
        
        filtered_examples = []
        for example in examples:
            example_category = example.get('category', 'unknown')
            if example_category in allowed_categories:
                filtered_examples.append(example)
        
        return filtered_examples


def load_blueprint_datasets(
    datasets_dir: str = "training/datasets",
    tokenizer: AutoTokenizer = None,
    max_length: int = 768,
    curriculum_stage: str = "all",
    train_test_split: float = 0.8
) -> Tuple[BluePrintDataset, BluePrintDataset]:
    """
    Load BluePrint datasets with automatic scanning.
    
    Returns:
        Tuple of (train_dataset, val_dataset)
    """
    
    # Scan datasets
    scanner = BluePrintDatasetScanner(datasets_dir)
    dataset_info = scanner.scan_datasets()
    
    all_examples = dataset_info['all_examples']
    
    # Shuffle for good distribution
    import random
    random.shuffle(all_examples)
    
    # Split into train/validation
    split_idx = int(len(all_examples) * train_test_split)
    train_examples = all_examples[:split_idx]
    val_examples = all_examples[split_idx:]
    
    # Create datasets
    if curriculum_stage != "all":
        train_dataset = BluePrintCurriculumDataset(
            train_examples, tokenizer, max_length, curriculum_stage
        )
        val_dataset = BluePrintCurriculumDataset(
            val_examples, tokenizer, max_length, curriculum_stage
        )
    else:
        train_dataset = BluePrintDataset(train_examples, tokenizer, max_length)
        val_dataset = BluePrintDataset(val_examples, tokenizer, max_length)
    
    logger.info(f"✅ Datasets created:")
    logger.info(f"   Training: {len(train_dataset)} examples")
    logger.info(f"   Validation: {len(val_dataset)} examples")
    
    return train_dataset, val_dataset


def validate_blueprint_syntax(response: str) -> Tuple[bool, List[str]]:
    """
    Validate BluePrint syntax in a response.
    
    Returns:
        Tuple of (is_valid, error_messages)
    """
    errors = []
    
    # Check for required tokens
    if '<thinking>' not in response:
        errors.append("Missing <thinking> tag")
    elif '</thinking>' not in response:
        errors.append("Missing </thinking> tag")
    
    if '<blueprint>' not in response:
        errors.append("Missing <blueprint> tag")
    elif '</blueprint>' not in response:
        errors.append("Missing </blueprint> tag")
    
    # Check thinking structure
    thinking_match = re.search(r'<thinking>(.*?)</thinking>', response, re.DOTALL)
    if thinking_match:
        thinking_content = thinking_match.group(1).strip()
        if 'PROBLEM UNDERSTANDING:' not in thinking_content:
            errors.append("Missing PROBLEM UNDERSTANDING section in thinking")
        if 'STRATEGIC APPROACH:' not in thinking_content:
            errors.append("Missing STRATEGIC APPROACH section in thinking")
        if 'DESIGN PREPARATION:' not in thinking_content:
            errors.append("Missing DESIGN PREPARATION section in thinking")
    
    # Check blueprint structure
    blueprint_match = re.search(r'<blueprint>(.*?)</blueprint>', response, re.DOTALL)
    if blueprint_match:
        blueprint_content = blueprint_match.group(1).strip()
        if 'Service ' not in blueprint_content:
            errors.append("Missing Service definition in blueprint")
    
    return len(errors) == 0, errors