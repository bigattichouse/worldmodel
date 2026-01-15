#!/usr/bin/env python3
"""
WorldModel Training with 1000 Examples
======================================

Updated training script using the expanded dataset.
"""

import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, TrainingArguments, 
    Trainer, DataCollatorForLanguageModeling, TrainerCallback
)
import json
import os
import time
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass
from torch.utils.data import Dataset
import logging

# Use the working training script but with new data
exec(open('train_worldmodel_rocm.py').read().replace(
    'DATA_FILE = "/home/bigattichouse/workspace/worldmodel/data/worldmodel_training.txt"',
    'DATA_FILE = "/home/bigattichouse/workspace/worldmodel/data/worldmodel_training_1000.txt"'
).replace(
    'OUTPUT_DIR = "./worldmodel_rocm_output"',
    'OUTPUT_DIR = "./worldmodel_1000_examples"'
).replace(
    'num_train_epochs=3',
    'num_train_epochs=5'
))