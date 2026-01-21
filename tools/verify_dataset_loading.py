#!/usr/bin/env python3
"""
Verify BluePrint Dataset Loading
=================================

Scans the training datasets directory to ensure all `.jsonl` files are properly
loaded by the `BluePrintDatasetScanner`. It also identifies other JSON-like
files (`.json`) that are being ignored by the current loading process.

This script helps to:
1.  Verify that all intended `.jsonl` training files are readable.
2.  Identify `.jsonl` files with formatting issues.
3.  Flag `.json` files that might need to be converted to `.jsonl`.
"""

import sys
import logging
from pathlib import Path

# Add src to path to allow for local imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from training.blueprint_dataset import BluePrintDatasetScanner

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def verify_datasets(datasets_dir: str = "training/datasets"):
    """
    Scans the dataset directory and reports on the loading status of
    JSON and JSONL files.
    """
    datasets_path = Path(datasets_dir)
    
    logger.info("--- Verifying BluePrint Dataset Loading ---")
    logger.info(f"Target directory: {datasets_path.resolve()}")
    logger.info("-" * 40)
    
    # Use the actual scanner to test loading
    scanner = BluePrintDatasetScanner(datasets_dir)
    scanner.scan_datasets()
    
    # --- Additional check for ignored .json files ---
    logger.info("-" * 40)
    logger.info("--- Checking for ignored .json files ---")
    
    json_files = list(datasets_path.glob("**/*.json"))
    
    if not json_files:
        logger.info("✅ No .json files found. All data files should be in .jsonl format.")
    else:
        logger.warning(f"⚠️ Found {len(json_files)} .json files that are ignored by the loader:")
        for json_file in json_files:
            logger.warning(f"  -> {json_file.relative_to(datasets_path)}")
        logger.warning("   Consider converting these files to .jsonl format if they should be included in the training.")
        
    logger.info("-" * 40)
    logger.info("✅ Verification complete.")


if __name__ == "__main__":
    verify_datasets()