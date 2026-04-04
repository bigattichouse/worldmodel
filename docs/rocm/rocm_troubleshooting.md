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

## Training Stability Issues

### "Too Many Open Files" Error
**Problem**: Training crashes with file descriptor limit reached, especially on long runs.

**Root Cause**: Evaluation dataloaders accumulate file descriptors with:
- `dataloader_num_workers=4` (multiple processes)
- `dataloader_persistent_workers=True` (workers stay alive)
- Frequent evaluation (every 50 steps)

**Fix**: Modify training script dataloader settings:
```python
# Before (problematic):
dataloader_num_workers=4,           # Too many parallel workers
dataloader_persistent_workers=True, # Workers accumulate file descriptors

# After (stable):
dataloader_num_workers=2,           # Reduced to prevent fd leaks  
dataloader_persistent_workers=False, # Clean shutdown prevents fd leaks
```

**Additional Stability Measures**:
- Increase evaluation interval (every 200 steps vs 50)
- Monitor system file descriptor usage: `lsof | wc -l`
- Check system limits: `ulimit -n`

## Hardware Considerations

- MI50 is an older generation GPU (2018)
- May have compatibility issues with newer ROCm 7.x
- Consider testing with ROCm 5.x or 6.x for better MI50 support
- Memory access faults often indicate hardware/driver issues rather than software configuration problems

## GPU Overheating Protection

### Problem
**Issue**: Training causes GPU to overheat, triggering hardware alarms and forcing system resets.

**Symptoms**:
- System forces reboot during long training runs
- `rocm-smi` shows temperatures > 85°C
- Hardware thermal alarms in system logs

### Solution: Progressive Thermal Throttling

The training script now includes **soft throttling** that reduces GPU power/clocks before resorting to a hard pause:

1. **Power Cap Reduction**: Lowers GPU power limit by 25W steps (225W → 200W → 175W → ...)
2. **Performance Level**: Sets GPU to minimum clocks if power cap hits floor (100W)
3. **Emergency Hard Pause**: Training pauses only at 107°C+ (3°C below the 110°C hardware alarm)
4. **Auto-Restore**: As GPU cools below 90°C, throttling is gradually removed
5. **Training End**: Full GPU settings restored to original values

### Throttle States

| State | What's Happening | Training Continues? |
|-------|-----------------|---------------------|
| `normal` | Full 225W, auto perf | ✅ Yes |
| `reduced_power` | Power cap lowered (200W → 100W) | ✅ Yes |
| `perf_low` | GPU at minimum clocks | ✅ Yes (slower) |
| `hard_pause` | Emergency pause at 107°C+ | ⏸ No |

### Default Behavior

Soft throttling kicks in at **95°C**, eases off at **90°C**. This keeps the GPU in the **90-95°C target band** during heavy training. The emergency hard stop at **107°C** is a safety net 3°C below the hardware alarm.

### Configuration

Command-line options for `train_rocm.sh`:

```bash
# Default — target 90-95°C band, emergency stop at 107°C
./train_rocm.sh --max-temp 95 --safe-temp 90 --cooldown-interval 30

# More conservative (pause earlier, wait longer)
./train_rocm.sh --max-temp 80 --safe-temp 70 --cooldown-interval 60

# More aggressive (higher thresholds, faster checks)
./train_rocm.sh --max-temp 90 --safe-temp 80 --cooldown-interval 15
```

### Monitoring

Temperature is logged automatically:
- At training start
- Every 100 steps (via `ProgressCallback`)
- At each epoch end
- During cooldown waits

Monitor in real-time with:
```bash
watch -n 5 rocm-smi --showtemp
```

### If Overheating Persists

1. **Reduce batch size**: Lower `--batch-size` reduces GPU load
2. **Increase gradient accumulation**: Maintain effective batch with `--grad-accum`
3. **Better cooling**: Ensure adequate case ventilation
4. **Reduce max_length**: Shorter sequences use less memory bandwidth
5. **Add cooling breaks**: Use conservative thresholds to trigger more frequent pauses

```bash
# Conservative settings for thermal management
./train_rocm.sh --batch-size 1 --grad-accum 16 --max-temp 80 --safe-temp 70
```

### Emergency Actions

If GPU overheats during training:
1. Training will automatically pause and attempt to cool down
2. If cooldown timeout is reached, training stops with error message
3. You can manually interrupt with `Ctrl+C` - checkpoint is saved periodically
4. Resume from last checkpoint: `./train_rocm.sh --resume ./output/worldmodel/checkpoint-XXX`