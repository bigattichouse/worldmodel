"""
Dataset loading for WorldModel training.

Loads JSONL examples from training/datasets/ and formats them using
Qwen3's chat template. Labels are masked for prompt tokens so the
model only learns to generate the assistant response.
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Optional
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

logger = logging.getLogger(__name__)

DATASETS_ROOT = Path(__file__).parent.parent.parent / "training" / "datasets"


def load_jsonl(path: Path) -> List[Dict]:
    """Load a JSONL file into a list of dicts."""
    examples = []
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                examples.append(json.loads(line))
            except json.JSONDecodeError as e:
                logger.warning(f"Skipping malformed line {lineno} in {path}: {e}")
    return examples


def load_all_datasets(root: Optional[Path] = None, categories: Optional[List[str]] = None) -> List[Dict]:
    """
    Recursively load all JSONL files from the datasets directory.

    Args:
        root: Path to datasets root (defaults to training/datasets/)
        categories: If set, only load these subdirectory names

    Returns:
        List of all examples across all files
    """
    root = root or DATASETS_ROOT
    all_examples = []

    for jsonl_path in sorted(root.rglob("*.jsonl")):
        if categories:
            parts = jsonl_path.relative_to(root).parts
            if not any(c in parts for c in categories):
                continue
        examples = load_jsonl(jsonl_path)
        all_examples.extend(examples)
        logger.info(f"Loaded {len(examples)} examples from {jsonl_path}")

    logger.info(f"Total examples loaded: {len(all_examples)}")
    return all_examples


def validate_example(example: Dict) -> bool:
    """Check that an example has required fields and non-empty content."""
    if "query" not in example or "response" not in example:
        return False
    if not example["query"].strip() or not example["response"].strip():
        return False
    if "<code>" in example["response"] and "<output>" not in example["response"]:
        logger.warning(f"Example {example.get('id', '?')} has <code> but no <output>")
        return False
    return True


class WorldModelDataset(Dataset):
    """
    PyTorch Dataset for WorldModel training.

    Uses Qwen3's chat template and masks prompt tokens in labels so loss
    is only computed on the assistant response (the tags + code + reasoning).
    """

    def __init__(
        self,
        examples: List[Dict],
        tokenizer: PreTrainedTokenizer,
        max_length: int = 1024,
        skip_invalid: bool = True,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []

        skipped = 0
        for ex in examples:
            if skip_invalid and not validate_example(ex):
                skipped += 1
                continue
            self.examples.append(ex)

        if skipped:
            logger.warning(f"Skipped {skipped} invalid examples")
        logger.info(f"Dataset ready: {len(self.examples)} examples")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        query = ex["query"].strip()
        response = ex["response"].strip()

        # Format with Qwen3 chat template to match pre-training format
        prompt_text = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": query}],
            tokenize=False,
            add_generation_prompt=True,
        )
        full_text = self.tokenizer.apply_chat_template(
            [
                {"role": "user", "content": query},
                {"role": "assistant", "content": response},
            ],
            tokenize=False,
            add_generation_prompt=False,
        )

        # Tokenize prompt alone to find its token length
        prompt_len = len(
            self.tokenizer(
                prompt_text,
                add_special_tokens=False,
            ).input_ids
        )

        # Tokenize full sequence
        encoding = self.tokenizer(
            full_text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)

        # Labels: -100 for prompt tokens and padding — only predict the response
        labels = input_ids.clone()
        labels[:min(prompt_len, len(labels))] = -100
        labels[attention_mask == 0] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
