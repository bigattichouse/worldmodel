#!/bin/bash
# ROCm WorldModel Training Runner
# ===============================

echo "🔥 Starting WorldModel ROCm Training"
echo "=================================="

# Source ROCm environment
echo "📡 Setting up ROCm environment..."
source /home/bigattichouse/workspace/rocm/setup-rocm-env.sh

# Set additional ROCm optimizations for MI50 (gfx906)
export HSA_OVERRIDE_GFX_VERSION=9.0.6
export HIP_VISIBLE_DEVICES=0
export ROCBLAS_LAYER=0

# PyTorch ROCm optimizations
export PYTORCH_HIP_ALLOC_CONF=max_split_size_mb:128

# Disable parallel tokenizers to avoid issues
export TOKENIZERS_PARALLELISM=false

echo "🌡️  GPU Status:"
rocm-smi --showtemp --showpower --showuse 2>/dev/null | head -5

echo ""
echo "🧪 Running quick test first..."
python quick_rocm_test.py

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Quick test passed! Starting full training..."
    python train_rocm_fixed.py
else
    echo ""
    echo "❌ Quick test failed. Please check the issues above."
    exit 1
fi