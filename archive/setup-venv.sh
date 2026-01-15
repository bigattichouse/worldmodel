#!/bin/bash
# Setup virtual environment with ROCm PyTorch for WorldModel training

set -e

echo "🐍 Setting up Python virtual environment for WorldModel training..."

# Create venv if it doesn't exist
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
else
    echo "✅ Virtual environment already exists"
fi

# Activate venv
echo "🔌 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install ROCm PyTorch first (specific index required)
echo "🔥 Installing ROCm PyTorch..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.0

# Install other requirements
echo "📚 Installing other dependencies..."
pip install transformers>=4.57.0
pip install huggingface_hub>=0.36.0
pip install accelerate>=1.12.0
pip install faiss-cpu>=1.13.0
pip install pytest>=9.0.0
pip install pytest-asyncio>=1.3.0
pip install chromadb>=1.4.0
pip install peft>=0.13.0
pip install bitsandbytes>=0.44.0

echo ""
echo "✅ Virtual environment setup complete!"
echo ""
echo "To activate the environment:"
echo "  source venv/bin/activate"
echo ""
echo "To run training:"
echo "  ./train_rocm.sh"
echo ""
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "ROCm available: $(python -c 'import torch; print(torch.cuda.is_available())')"