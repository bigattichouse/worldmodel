#!/bin/bash
# WASM WorldModel Setup Script

set -e  # Exit on any error

echo "🚀 Setting up WASM WorldModel environment..."

# Check Python version
python_version=$(python3 --version 2>&1 | grep -o '[0-9]\+\.[0-9]\+' | head -1)
echo "📍 Python version: $python_version"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
else
    echo "✅ Virtual environment already exists"
fi

# Activate virtual environment
echo "🔌 Activating virtual environment..."
source venv/bin/activate

# Detect hardware and install PyTorch
echo "🔍 Detecting hardware..."
if command -v rocm-smi &> /dev/null; then
    echo "🔴 AMD GPU detected - installing ROCm PyTorch..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.0
elif command -v nvidia-smi &> /dev/null; then
    echo "🟢 NVIDIA GPU detected - installing CUDA PyTorch..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
else
    echo "💻 No GPU detected - installing CPU-only PyTorch..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
fi

# Install other dependencies
echo "📚 Installing other dependencies..."
pip install -r requirements.txt

# Download model if it doesn't exist
if [ ! -d "../model/Qwen3-0.6B" ]; then
    echo "🤖 Downloading Qwen3-0.6B model..."
    mkdir -p ../model
    cd ../model
    git clone https://huggingface.co/Qwen/Qwen3-0.6B
    cd ../worldmodel
else
    echo "✅ Qwen3-0.6B model already exists"
fi

# Verify installation
echo "🔧 Verifying installation..."
python3 -c "import torch; print(f'✅ GPU Available: {torch.cuda.is_available()}'); print(f'   GPU Name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU-only\"}')"
python3 -c "import wasmtime; print('✅ WASM runtime ready')"
python3 -c "import transformers; print('✅ Transformers ready')"

echo ""
echo "🎉 Setup complete! To get started:"
echo ""
echo "   # Activate virtual environment:"
echo "   source venv/bin/activate"
echo ""
echo "   # Train model:"
echo "   python train_worldmodel.py"
echo ""
echo "   # Run inference:"
echo "   python run_inference.py \"what is 5 * 3\""
echo ""