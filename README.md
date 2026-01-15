# WorldModel: LLM Training for Structured Reasoning

A system that trains language models to perform computational tasks using structured `<think>`, `<model>`, and `<requires>` tags for systematic reasoning and code execution.

## 🛠️ Installation & Setup

### 1. Prerequisites
- **Python 3.8-3.12** (avoid Python 3.13)
- **ROCm 7.1+** (for AMD GPU training) or **CUDA 11.8+** (for NVIDIA)
- **16GB+ VRAM** (32GB+ recommended for optimal training)
- **20GB+ disk space** (for models and datasets)

### 2. Clone Repository
```bash
git clone <repository-url>
cd worldmodel
```

### 3. Create Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 4. Install Dependencies

**For AMD ROCm (Recommended)**:
```bash
# Install PyTorch with ROCm support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.0

# Install other dependencies
pip install -r requirements.txt
```

**For NVIDIA CUDA**:
```bash
# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
pip install -r requirements.txt
```

**For CPU-Only (Slower)**:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

### 5. Download Base Model
```bash
# Create model directory
mkdir -p ../model

# Download Qwen3-0.6B model (recommended for testing)
cd ../model
git clone https://huggingface.co/Qwen/Qwen3-0.6B

# Alternative: Download larger model for better performance
# git clone https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct
# git clone https://huggingface.co/Qwen/Qwen2.5-3B-Instruct

cd ../worldmodel
```

### 6. Set Environment Variables (AMD ROCm)
```bash
# For AMD ROCm users - add to ~/.bashrc or run before training
export HSA_OVERRIDE_GFX_VERSION=9.0.6  # For MI50
# export HSA_OVERRIDE_GFX_VERSION=10.3.0  # For newer AMD GPUs
export PYTORCH_ROCM_ARCH=gfx906  # For MI50
export HIP_VISIBLE_DEVICES=0
```

### 7. Verify Installation
```bash
# Test GPU detection
python3 -c "import torch; print(f'GPU Available: {torch.cuda.is_available()}'); print(f'GPU Name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"

# Quick ROCm test (AMD users)
python3 -c "import torch; x = torch.randn(100, 100).cuda(); print('ROCm working!') if torch.cuda.is_available() else print('Using CPU')"
```

## 🚀 Quick Start

### Training
```bash
# Complete workflow (training + inference test)
./complete_workflow.sh

# Training only  
python3 train_worldmodel_rocm.py
```

### Inference
```bash
# Interactive session
python3 run_worldmodel_inference.py --interactive

# Single query
python3 run_worldmodel_inference.py "Calculate 25% of 400"
```

## 🧠 How It Works

The system teaches models to generate structured responses:

**Input**: "What's 15% of 200?"

**Model Output**:
```
<think>I need to calculate 15% of 200...</think>
<model>
result = 0.15 * 200
print(f"15% of 200 = {result}")
</model>
<requires>python:math</requires>

15% of 200 equals 30.
```

**System Response**:
1. Parses the structured output
2. Safely executes the generated code
3. Returns results with explanation

## 📁 Structure

```
worldmodel/
├── train_worldmodel_rocm.py    # Main training script
├── run_worldmodel_inference.py # Inference engine with code execution
├── complete_workflow.sh        # One-command training + testing
├── requirements.txt            # Dependencies
├── data/                       # Training datasets (1000+ examples)
├── docs/                       # Documentation and guides
└── archive/                    # Development scripts and variants
```

## 📊 Training Data

**1000+ Examples Across**:
- **Math**: Basic arithmetic, percentages, geometry
- **Text Analysis**: Character counting, string processing
- **System Tasks**: Date/time, file operations, environment info
- **Data Processing**: Statistics, JSON/CSV parsing
- **Advanced Math**: Trigonometry, linear algebra, number theory

## ⚙️ Requirements

- **Python 3.8+**
- **ROCm 7.1+** (for AMD GPU training)
- **PyTorch 2.4.1+rocm6.0** (specific version for MI50 compatibility)
- **Transformers, PEFT, Accelerate**

## 🎯 Training Results

- **Loss reduction**: 1.46 → 0.59 (60% improvement in 3 epochs)
- **Training time**: 
  - **30 epochs**: ~6-10 hours (current default, comprehensive training)
  - **3 epochs**: ~6 minutes (quick test)
- **Structure quality**: 2/3 → 3/3 WorldModel tags with extended training
- **GPU utilization**: 80-95% on AMD MI50 (32GB VRAM)

## 🔧 Troubleshooting

### Common Issues

**Model download fails**:
```bash
# Make sure you have git-lfs installed
sudo apt install git-lfs  # Ubuntu/Debian
brew install git-lfs     # macOS
git lfs install
```

**GPU not detected**:
- **AMD ROCm**: Check `rocm-smi` shows your GPU
- **NVIDIA**: Check `nvidia-smi` shows your GPU  
- **Both**: Verify environment variables are set

**Out of memory errors**:
- Reduce model size: Use Qwen3-0.6B instead of larger models
- Reduce batch size in training script
- Use gradient checkpointing (enabled by default)

**Training loss not decreasing**:
- Check that training data is properly formatted
- Verify model path is correct
- Ensure sufficient training epochs (30 recommended)

**ROCm-specific issues**:
- Use PyTorch 2.4.1+rocm6.0 (not newer versions)
- Set `HSA_OVERRIDE_GFX_VERSION=9.0.6` for MI50
- Check `docs/` for detailed ROCm troubleshooting

### Getting Help
- Check `docs/` directory for detailed guides
- Review training logs in `logs/` directory
- Test with CPU-only mode if GPU issues persist

## 📖 Documentation

See `docs/` directory for:
- **Training guides** and ROCm setup
- **Inference examples** and API reference  
- **Troubleshooting** for common issues
- **Performance optimization** tips

## 🔧 ROCm Compatibility

Optimized for **AMD Instinct MI50** with:
- **Native gfx906 architecture** (HSA_OVERRIDE_GFX_VERSION=9.0.6)
- **Conservative memory settings** for 32GB VRAM
- **Stable PyTorch 2.4.1+rocm6.0** (avoids newer version issues)

## 🎉 Ready to Use

This system is production-ready for:
- **Educational tools** teaching step-by-step reasoning
- **Computational assistants** with code execution
- **Research platforms** for structured AI reasoning
- **Custom applications** requiring reliable code generation

The WorldModel approach bridges natural language understanding with executable computation, making AI reasoning transparent and verifiable.