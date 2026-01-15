#!/bin/bash
# Optimized ROCm training script based on finetune.md recommendations
# Uses improved PyTorch configuration instead of TorchTune (compatibility issues)

set -e

echo "🔥 Starting Optimized ROCm Fine-tuning"
echo "===================================="

# Cleanup function for fan speed
cleanup() {
    echo ""
    echo "🌪️  Resetting fans to normal speed..."
    sudo echo 128 > /sys/class/hwmon/hwmon3/pwm2 || echo "⚠️  Warning: Could not reset fan speed"
}
trap cleanup EXIT INT TERM

# Set fans to max speed
echo "🌪️  Setting fans to maximum speed..."
sudo echo 255 > /sys/class/hwmon/hwmon3/pwm2 || echo "⚠️  Warning: Could not set fan speed"

# Activate virtual environment
echo "📦 Activating virtual environment..."
source venv/bin/activate

# Source ROCm environment  
echo "🚀 Setting up ROCm environment..."
source /home/bigattichouse/workspace/rocm/setup-rocm-env.sh

# Environment variables for ROCm optimization
export HSA_OVERRIDE_GFX_VERSION=9.0.6
export PYTORCH_ROCM_ARCH=gfx906
export TOKENIZERS_PARALLELISM=false
export HIP_VISIBLE_DEVICES=0
export PYTORCH_HIP_ALLOC_CONF=max_split_size_mb:512
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export OMP_NUM_THREADS=1
export HSA_DISABLE_CACHE=1
export HSA_FORCE_FINE_GRAIN_PCIE=1

echo ""
echo "🖥️  System Information:"
echo "   PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "   ROCm Available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "   GPU: $(python -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")')"

echo ""
echo "📊 Optimized Configuration for MI50 32GB:"
echo "   PyTorch: 2.4.1+rocm6.0 (better MI50 support)"
echo "   Architecture: gfx906 (native MI50)"
echo "   Batch Size: 2 (optimized for 32GB)"
echo "   Gradient Accumulation: 4"  
echo "   Sequence Length: 4096"
echo "   LoRA Rank: 16"
echo "   Learning Rate: 5e-5 (conservative)"
echo "   Epochs: 1 (testing)"

echo ""
echo "🚀 Starting optimized training..."

# Run optimized training
python train_rocm_optimized.py

echo ""
echo "✅ Optimized training script completed!"