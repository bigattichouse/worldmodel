# ROCm GPU Memory Access Fault Troubleshooting

## Issue Summary
- **Problem**: Memory access fault when trying to perform GPU tensor operations
- **Error**: `Memory access fault by GPU node-1 on address (nil). Reason: Page not present or supervisor privilege.`
- **Impact**: Prevents any GPU tensor operations, including fine-tuning

## Diagnostic Information
- **GPU**: AMD Instinct MI50/MI60 (32GB)
- **ROCm Version**: 7.1.1
- **PyTorch**: 2.5.1+rocm6.2
- **Detection**: GPU is detected correctly, memory shown as 31.98GB
- **Failure Point**: Any attempt to create tensors on GPU (.cuda() operations)

## TorchTune Compatibility Issue
- **TorchTune Status**: Incompatible with ROCm PyTorch 2.5.1+rocm6.2
- **Root Cause**: torchao dependency requires torch.int1 (not available in ROCm build)
- **Recommendation**: Use PyTorch + HuggingFace + PEFT instead

## Troubleshooting Steps

### 1. ROCm Driver Reset
```bash
# Reset ROCm driver
sudo modprobe -r amdgpu
sudo modprobe amdgpu

# Check ROCm status
rocm-smi
rocminfo | grep "Name:"
```

### 2. GPU Power/Thermal Check
```bash
# Check GPU status
rocm-smi --showtemp --showpower --showmemuse

# Check for thermal throttling
rocm-smi --showclocks

# Reset GPU
rocm-smi --resetclocks
```

### 3. Memory Testing
```bash
# Test GPU memory
rocm-smi --showmeminfo

# Clear memory
echo 3 | sudo tee /proc/sys/vm/drop_caches
```

### 4. ROCm Environment Reset
```bash
# Unset potentially conflicting variables
unset HSA_OVERRIDE_GFX_VERSION
unset PYTORCH_ROCM_ARCH
unset HIP_VISIBLE_DEVICES

# Restart with minimal environment
export HIP_VISIBLE_DEVICES=0
export HSA_OVERRIDE_GFX_VERSION=9.0.6  # Try original gfx906 instead of 10.3.0
```

### 5. PyTorch Reinstallation
```bash
# Reinstall PyTorch ROCm version
pip uninstall torch torchvision torchaudio
pip install torch==2.4.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.0
```

### 6. Alternative Testing
```bash
# Test with different gfx versions
export HSA_OVERRIDE_GFX_VERSION=9.0.6    # MI50 native
export HSA_OVERRIDE_GFX_VERSION=10.3.0   # What we were using
export HSA_OVERRIDE_GFX_VERSION=11.0.0   # Newer target

# Test with different PyTorch configurations
python -c "
import torch
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
print('Tensor test...')
"
```

## Known Working Alternatives

### 1. CPU-Only Training (Fallback)
- Modify SFTConfig to use `device_map="cpu"`
- Very slow but functional for testing

### 2. Docker-based ROCm
```bash
# Use official ROCm PyTorch container
docker run -it --device=/dev/kfd --device=/dev/dri --group-add video rocm/pytorch:latest
```

### 3. Different ROCm Version
- Try ROCm 6.0 with PyTorch 2.4.0
- May have better stability for older MI50 hardware

## Immediate Next Steps

1. **System Restart**: Reboot to clear any GPU state issues
2. **Driver Reset**: Reset ROCm drivers
3. **Conservative Environment**: Try HSA_OVERRIDE_GFX_VERSION=9.0.6
4. **Minimal Test**: Test single tensor creation without ML libraries

## Current Optimized Configuration

The `train_rocm_optimized.py` script has been configured for 32GB memory with:
- batch_size: 2
- max_sequence_length: 4096 
- lora_rank: 16
- gradient_accumulation_steps: 4

This should work once the memory access fault is resolved.

## Hardware Considerations

- MI50 is an older generation GPU (2018)
- May have compatibility issues with newer ROCm 7.x
- Consider testing with ROCm 5.x or 6.x for better MI50 support
- Memory access faults often indicate hardware/driver issues rather than software configuration problems