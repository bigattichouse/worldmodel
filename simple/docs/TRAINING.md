# Training Guide

## Overview

This guide covers training WorldModel LLMs using two approaches:
1. **llama.cpp + ROCm** (recommended for AMD GPUs)
2. **PyTorch** (fallback for CPU/CUDA)

## llama.cpp Training (Recommended)

### Prerequisites
- llama.cpp built with ROCm support
- AMD GPU with ROCm drivers
- Model in GGUF F32 format

### Setup
```bash
# 1. Prepare training data
python3 convert_to_llama_cpp.py

# 2. Convert model to GGUF
cd ../llama.cpp
python3 convert_hf_to_gguf.py ../model/Qwen2.5-3B-Instruct/ \
  --outfile ../model/qwen2.5-3b-f32.gguf --outtype f32

# 3. Set ROCm environment  
source /path/to/rocm/setup-rocm-env.sh
export HSA_OVERRIDE_GFX_VERSION=10.3.0
```

### Training Command
```bash
./build/bin/llama-finetune \
  --file ../worldmodel/data/worldmodel_llama_cpp_training.txt \
  --model ../model/qwen2.5-3b-f32.gguf \
  -ngl 999 \          # Use all GPU layers
  -c 512 \            # Context length
  -b 256 \            # Batch size
  -ub 256 \           # Micro batch size  
  --output-model ../worldmodel/qwen2.5-worldmodel-lora.gguf
```

### Expected Performance
- **Training time**: ~10-30 minutes (depending on GPU)
- **Memory usage**: ~12-14 GB VRAM
- **GPU utilization**: 80-95%

## PyTorch Training (Alternative)

### CPU Training
```bash
python3 main.py train sft \
  --data ./data/worldmodel_enhanced_training.json \
  --epochs 1 \
  --batch-size 1 \
  --learning-rate 5e-5
```

### GPU Training (CUDA)
```bash
python3 main.py train sft \
  --data ./data/worldmodel_enhanced_training.json \
  --epochs 1 \
  --batch-size 8 \
  --learning-rate 5e-5 \
  --use-lora
```

## ROCm Compatibility Notes

### Known Issues
- **PyTorch + ROCm + gfx906**: Documented segfault issues
- **Phi-4 mini**: Requires unreleased transformers features
- **Memory access faults**: Common with older AMD architectures

### Workarounds
1. **Use llama.cpp**: Proven ROCm compatibility
2. **ROCm 5.7.3**: Better performance than newer versions on gfx906
3. **CPU fallback**: Always works as last resort

## Monitoring Training

### GPU Usage
```bash
# Monitor GPU utilization
watch -n 5 rocm-smi --showuse --showtemp --showpower

# Check memory usage
rocm-smi --showmeminfo
```

### Training Progress
- Watch for decreasing loss values
- Monitor GPU temperature (keep under 80°C)
- Check for memory overflow errors

## Troubleshooting

### Common Issues

**GPU not detected:**
```bash
export HSA_OVERRIDE_GFX_VERSION=10.3.0
rocm-smi --showtemp
```

**Out of memory:**
- Reduce batch size (`-b 128` instead of 256)
- Use gradient accumulation
- Enable LoRA training

**Segfaults with PyTorch:**
- Switch to llama.cpp training
- Use CPU-only PyTorch
- Downgrade to ROCm 5.7.3

## Model Testing

### Quick Test
```bash
# Test with llama.cpp
cd ../llama.cpp
./build/bin/llama-cli \
  --model ../worldmodel/qwen2.5-worldmodel-lora.gguf \
  --prompt "Count the R's in strawberry" \
  -ngl 999

# Test with WorldModel system
python3 main.py generate "Calculate 25% of 200" --worldmodel --verbose
```

### Expected Output
The model should generate structured responses:
```
<think>Need to calculate 25% of 200...</think>
<model>
result = 200 * 0.25
print(f"25% of 200 = {result}")
</model>
<requires>python:math</requires>
```