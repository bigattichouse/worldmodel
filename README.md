# WorldModel LLM Training System

A comprehensive system for training language models to use structured reasoning with `<think>`, `<model>`, and `<requires>` tags for computational tasks.

## Overview

This project implements the **WorldModel** approach to LLM training, teaching models to:
- **Think systematically** using `<think>` tags for reasoning
- **Generate executable code** in `<model>` tags  
- **Specify requirements** with `<requires>` tags for execution

The system includes 184 carefully crafted training examples covering mathematics, physics, chemistry, and tricky problems like "count the R's in strawberry."

## Quick Start

### Prerequisites
- Python 3.8+
- ROCm 7.1.1+ (for AMD GPU training)
- llama.cpp (for optimal training performance)

### Installation
```bash
pip install -r requirements.txt
```

### Training Options

**Option 1: llama.cpp + ROCm (Recommended)**
```bash
# Set up environment
source /path/to/rocm/setup-rocm-env.sh
export HSA_OVERRIDE_GFX_VERSION=10.3.0

# Convert model to GGUF format
cd ../llama.cpp
python3 convert_hf_to_gguf.py ../model/Qwen2.5-3B-Instruct/ --outfile ../model/qwen2.5-3b-f32.gguf --outtype f32

# Start training
./build/bin/llama-finetune \
  --file ../worldmodel/data/worldmodel_llama_cpp_training.txt \
  --model ../model/qwen2.5-3b-f32.gguf \
  -ngl 999 -c 512 -b 256
```

**Option 2: PyTorch Training (CPU/CUDA)**
```bash
python3 main.py train sft \
  --data ./data/worldmodel_enhanced_training.json \
  --epochs 1 --batch-size 4 \
  --output ./worldmodel_finetuned
```

## Project Structure

```
worldmodel/
├── main.py                    # Main CLI interface
├── config.json               # Training configuration  
├── src/                       # Core system modules
│   ├── core/                  # Inference and tag parsing
│   ├── training/              # Training implementations
│   ├── execution/             # Code execution system
│   └── utils/                 # Utilities and config
├── data/                      # Training datasets
├── spec/                      # Design documentation
└── scripts/                   # Training and utility scripts
```

## Training Data

The system includes comprehensive training examples:
- **173 base examples**: Computational tasks across multiple domains
- **11 science examples**: Phase change materials, chemistry, physics
- **Tricky examples**: Famous edge cases like "strawberry R counting"

### Data Format
Examples follow the WorldModel structure:
```
User: Calculate 15% tip on $67.50