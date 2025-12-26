#!/bin/bash
# WorldModel LLM Training with llama.cpp + ROCm
# Optimized for AMD MI50 (gfx906) GPU

set -e

echo "🔥 WorldModel LLM Training with llama.cpp + ROCm"
echo "================================================="

# Setup ROCm environment
echo "🚀 Setting up ROCm environment..."
source /home/bigattichouse/workspace/rocm/setup-rocm-env.sh
export HSA_OVERRIDE_GFX_VERSION=10.3.0

# ROCm stability fixes for gfx906
export HIP_FORCE_DEV_KERNARG=1
export HSA_ENABLE_SDMA=0
export ROCR_VISIBLE_DEVICES=0

# Check GPU detection
echo "🖥️  GPU Detection:"
rocm-smi --showtemp

# Model paths
MODEL_DIR="../model"
TRAINING_DATA="./data/worldmodel_llama_cpp_training.txt"
OUTPUT_MODEL="./qwen2.5-worldmodel-lora.gguf"

# Check if model exists, convert if needed
if [ ! -f "$MODEL_DIR/qwen2.5-3b-f32.gguf" ]; then
    echo "📦 Converting Qwen2.5-3B to GGUF F32 format..."
    cd ../llama.cpp
    python3 convert_hf_to_gguf.py "$MODEL_DIR/Qwen2.5-3B-Instruct/" \
        --outfile "$MODEL_DIR/qwen2.5-3b-f32.gguf" \
        --outtype f32
    cd ../worldmodel
    echo "✅ Model conversion complete"
else
    echo "✅ GGUF model already exists"
fi

# Check training data exists
if [ ! -f "$TRAINING_DATA" ]; then
    echo "❌ Training data not found: $TRAINING_DATA"
    echo "Run: python3 convert_to_llama_cpp.py"
    exit 1
fi

echo ""
echo "📊 Training Configuration:"
echo "   Model: Qwen2.5-3B-Instruct (F32 GGUF)"
echo "   Training Data: 184 WorldModel examples"
echo "   GPU Layers: 999 (all on GPU)"
echo "   Context Size: 512 tokens"
echo "   Batch Size: 256"
echo "   Output: $OUTPUT_MODEL"
echo ""

# Start training
echo "🚀 Starting LoRA fine-tuning..."
cd ../llama.cpp

# Try conservative ROCm settings for gfx906 stability
./build/bin/llama-finetune \
    --file "../worldmodel/$TRAINING_DATA" \
    --model "$MODEL_DIR/qwen2.5-3b-f32.gguf" \
    -ngl 32 \
    -c 512 \
    -b 64 \
    -ub 32 \
    --epochs 1 \
    --learning-rate 5e-5 \
    -o "../worldmodel/$OUTPUT_MODEL"

cd ../worldmodel

echo ""
echo "✅ Training Complete!"
echo "📁 Model saved to: $OUTPUT_MODEL"
echo ""
echo "🧪 Test the model:"
echo "cd ../llama.cpp"
echo "./build/bin/llama-cli --model ../worldmodel/$OUTPUT_MODEL \\"
echo "  --prompt 'Count the R's in strawberry' -ngl 999"
echo ""
echo "Or test with WorldModel system:"
echo "python3 main.py generate 'Calculate 25% of 200' --worldmodel --verbose"