#!/bin/bash
# Training script with ROCm environment setup

# Activate venv first (has ROCm PyTorch)
source venv/bin/activate

# Source ROCm environment
source /home/bigattichouse/workspace/rocm/setup-rocm-env.sh

# Set gfx906 override for older GPU
export HSA_OVERRIDE_GFX_VERSION=10.3.0

# Disable tokenizer parallelism warnings
export TOKENIZERS_PARALLELISM=false

echo "Starting Phi-4-mini fine-tuning with ROCm..."
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "GPU detected: $(python -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\")')"

# Run training with larger batch size for GPU
python3 main.py train sft \
  --data ./data/worldmodel_final_training.json \
  --epochs 1 \
  --batch-size 2 \
  --learning-rate 5e-5 \
  --output ./phi4_worldmodel_finetuned_rocm