#!/bin/bash
# Enhanced ROCm training script for Qwen2.5 WorldModel fine-tuning

set -e

echo "🔥 Starting Qwen2.5 WorldModel Fine-tuning with ROCm"
echo "=================================================="

# Activate virtual environment
echo "📦 Activating virtual environment..."
source venv/bin/activate

# Setup ROCm environment
echo "🚀 Setting up ROCm 7.1.1 environment..."
source /home/bigattichouse/workspace/rocm/setup-rocm-env.sh

# Set gfx906 override for MI50
echo "🎯 Setting GPU architecture override for MI50 (gfx906)..."
export HSA_OVERRIDE_GFX_VERSION=10.3.0

# Disable tokenizer parallelism warnings
export TOKENIZERS_PARALLELISM=false

# ROCm-specific optimizations
export PYTORCH_ROCM_ARCH=gfx906
export HIP_FORCE_DEV_KERNARG=1
export ROCBLAS_LAYER=0

echo ""
echo "🖥️  System Information:"
echo "   ROCm Version: $(rocminfo | grep 'ROCm Version' | head -1)"
echo "   PyTorch Version: $(python -c 'import torch; print(torch.__version__)')"
echo "   GPU Detection: $(python -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\")')"
echo "   GPU Memory: $(python -c 'import torch; print(f\"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB\" if torch.cuda.is_available() else \"N/A\")')"

echo ""
echo "📊 Training Configuration:"
echo "   Model: Qwen2.5-3B-Instruct"
echo "   Dataset: 184 examples (173 base + 11 science)"
echo "   Batch Size: 8 (ROCm optimized)"
echo "   Gradient Accumulation: 2 steps"
echo "   Learning Rate: 5e-5"
echo "   Epochs: 1"
echo ""

# Test GPU availability
echo "🧪 Testing GPU availability..."
python3 -c "
import torch
print(f'CUDA Available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'Device: {torch.cuda.get_device_name(0)}')
    print(f'Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
    print(f'Capability: {torch.cuda.get_device_capability(0)}')
else:
    print('⚠️ GPU not detected! Training will fall back to CPU.')
"

echo ""
echo "🚀 Starting Fine-tuning..."

# Run training with enhanced dataset
python3 main.py train sft \
  --data ./data/worldmodel_enhanced_training.json \
  --epochs 1 \
  --batch-size 8 \
  --learning-rate 5e-5 \
  --output ./qwen2.5_worldmodel_rocm_finetuned

echo ""
echo "✅ Training completed!"
echo "📁 Model saved to: ./qwen2.5_worldmodel_rocm_finetuned"
echo ""
echo "🧪 To test the fine-tuned model:"
echo "   python3 main.py generate \"Count the R's in strawberry\" --worldmodel --verbose"