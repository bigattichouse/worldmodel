# WorldModel Fine-tuning Specification for ROCm

## Overview

This document outlines the challenges and solutions for fine-tuning large language models on AMD ROCm hardware, specifically addressing issues encountered with the WorldModel project using Qwen2.5 models.

## Current Issues Identified

### 1. llama.cpp Fine-tuning Limitations
- **Problem**: llama.cpp is excellent for inference on ROCm but problematic for fine-tuning
- **Evidence**: Our `train_llamacpp_rocm.sh` script exists but faces stability issues
- **Root Cause**: llama.cpp's training implementation has limited ROCm optimization compared to PyTorch

### 2. Python Version Compatibility
- **Critical Issue**: Python 3.13 has backwards incompatible C binary interop issues
- **Impact**: Most ML libraries won't work with Python 3.13
- **Recommendation**: Use Python 3.9-3.12 for ROCm fine-tuning

### 3. ROCm-Specific Framework Limitations
- **Unsloth**: No AMD GPU support (consumer or professional)
- **BitsAndBytes**: Limited ROCm support, requires specific ROCm 6.0+ installation
- **Consumer GPU Issues**: Libraries like xformers ROCm variant only support workstation GPUs

## Recommended Fine-tuning Stack for ROCm

### Primary Framework: PyTorch + HuggingFace Ecosystem
```python
# Core dependencies (current requirements.txt shows good selection)
torch>=2.6.0  # ROCm-compatible version
transformers>=4.57.0
accelerate>=1.12.0
peft>=0.13.0  # For LoRA fine-tuning
```

### Recommended Libraries

1. **TorchTune** (AMD's recommended)
   - PyTorch-native library for single/multi-GPU fine-tuning
   - Designed specifically for LLM fine-tuning and inference
   - Strong ROCm compatibility

2. **PEFT Library** (Currently in use)
   - Parameter-Efficient Fine-Tuning with LoRA
   - AdaLoRA and other optimization variants
   - Integrated with Hugging Face ecosystem

3. **Hugging Face Accelerate** (Currently in use)
   - Simplifies multi-GPU scaling
   - Maintains performance and flexibility
   - Integrated with Transformers

## Current Implementation Analysis

### Strengths of Current Setup (`src/training/sftTrainer.py`)
```python
# Good ROCm-specific optimizations already implemented:
- torch_dtype=torch.float32  # ROCm stability
- attn_implementation="eager"  # ROCm compatibility
- dataloader_num_workers=0   # ROCm stability
- fp16=False, bf16=False     # ROCm compatibility
- dataloader_pin_memory=False # ROCm optimization
```

### Configuration Issues to Address

1. **Model Path**: Currently hardcoded to `../model/Qwen2.5-3B-Instruct`
   - Should be configurable for different model sizes
   - Consider using smaller models (1.5B) for initial testing

2. **Memory Configuration**: 
   - Current: `batch_size=1, gradient_accumulation_steps=16`
   - Too conservative for modern ROCm hardware
   - Recommend testing with larger batches

3. **Quantization Disabled**: 
   - `use_4bit=False, use_8bit=False`
   - This is correct for ROCm stability, but consider QLoRA for memory efficiency

## Recommended Migration Strategy

### Phase 1: Optimize Current PyTorch Setup
1. **Environment Standardization**
   ```bash
   # Use system Python 3.9-3.12 (avoid 3.13)
   # ROCm 6.0+ with PyTorch ROCm build
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.0
   ```

2. **Configuration Updates**
   ```python
   # Recommended SFTConfig adjustments for MI50/MI100/MI200 series:
   batch_size: 4-8           # Instead of 1
   gradient_accumulation_steps: 4-8  # Instead of 16
   max_sequence_length: 4096  # Instead of 2048 if memory allows
   learning_rate: 1e-4       # Slightly higher for faster convergence
   ```

### Phase 2: Consider TorchTune Migration
```bash
# Add TorchTune to dependencies
pip install torchtune

# TorchTune provides optimized configs for popular models
tune ls                    # List available model configs
tune download qwen2       # Download model-specific configs
```

### Phase 3: Advanced ROCm Optimizations
1. **Flash Attention**: When ROCm support improves
2. **Gradient Checkpointing**: Already enabled, good
3. **Mixed Precision**: Test bf16 with newer ROCm versions

## Memory Optimization Techniques

### Current Memory Issues
- MI50: 16GB HBM2 memory
- MI100/MI200: 32GB+ HBM2 memory
- Need efficient memory usage for 3B+ parameter models

### Recommended Optimizations
1. **LoRA Configuration** (Currently implemented)
   ```python
   lora_rank: 8-16          # Start with 8, increase if needed
   lora_alpha: 16-32        # 2x rank typically
   target_modules: ["q_proj", "v_proj", "k_proj", "o_proj"]  # Core attention
   ```

2. **Gradient Checkpointing**: Already enabled
3. **Sequence Length Management**: Dynamic padding vs fixed length
4. **Batch Size Optimization**: Test 2^n values (2, 4, 8, 16)

## Testing and Validation Strategy

### Step 1: Environment Validation
```bash
# Test script for ROCm setup validation
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'ROCm Available: {torch.cuda.is_available()}')
print(f'Device: {torch.cuda.get_device_name(0)}' if torch.cuda.is_available() else 'No GPU')
"
```

### Step 2: Memory Benchmarking
- Start with smallest viable batch size
- Gradually increase until OOM
- Monitor with `rocm-smi` during training

### Step 3: Training Validation
- Single epoch validation runs
- Loss convergence monitoring
- Performance benchmarking (samples/sec)

## Alternative Approaches

### Docker-based Training
```bash
# Use official ROCm PyTorch container
docker pull rocm/pytorch:latest
# Pre-configured with ROCm-optimized PyTorch
```

### Cloud-based Training
- AWS EC2 with MI100/MI200 instances
- Google Cloud with A100s (CUDA fallback)
- Azure with MI200 series

### Hybrid Approach
- Use ROCm for inference (llama.cpp excels here)
- Use cloud CUDA instances for fine-tuning
- Convert between formats as needed

## Implementation Priority

### High Priority (Fix Current Issues)
1. ✅ Verify Python version compatibility
2. ✅ Test current PyTorch ROCm installation
3. ✅ Validate GPU memory detection
4. 🔄 Optimize batch size and memory usage
5. 🔄 Test training stability with current config

### Medium Priority (Optimization)
1. 📋 Benchmark different LoRA configurations
2. 📋 Test TorchTune integration
3. 📋 Implement dynamic sequence length
4. 📋 Add comprehensive monitoring

### Low Priority (Advanced Features)
1. 📋 Flash Attention integration (when ROCm ready)
2. 📋 Multi-GPU support
3. 📋 Mixed precision training
4. 📋 Custom kernel optimizations

## Monitoring and Debugging

### Key Metrics to Track
- GPU memory utilization (`rocm-smi`)
- Training loss convergence
- Examples per second throughput
- Memory fragmentation issues
- Temperature and power consumption

### Common ROCm Issues and Solutions
1. **Memory Fragmentation**: Use `torch.cuda.empty_cache()` regularly
2. **Driver Issues**: Ensure ROCm 6.0+ with latest drivers
3. **Kernel Launch Failures**: Reduce batch size, disable mixed precision
4. **Context Switching**: Use `HIP_VISIBLE_DEVICES=0` to force single GPU

## Conclusion

The current WorldModel implementation has a solid foundation with PyTorch + HuggingFace + PEFT. The main issues are configuration optimization rather than fundamental architectural problems. Focus on:

1. **Environment Stability**: Python version, ROCm drivers
2. **Memory Optimization**: Batch sizes, LoRA configuration  
3. **Training Stability**: Conservative precision settings, proper error handling
4. **Performance Monitoring**: Comprehensive metrics and debugging

The llama.cpp approach should be considered primarily for inference, while PyTorch remains the best option for fine-tuning on ROCm hardware.