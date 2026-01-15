# Training Crash Fix

## Problem
Your training crashed at epoch 26.44 with "Too many open files" error. This is caused by file descriptor leaks in multiprocessing dataloaders.

## Root Cause
```
dataloader_num_workers=4
dataloader_persistent_workers=True  # Workers stay alive
eval_steps=50  # Frequent evaluation creates new dataloaders
```

Over 26 epochs, evaluation dataloaders accumulated file descriptors and hit system limit.

## Quick Fix (if you want to resume the main training)

Edit `train_worldmodel_rocm_stable.py` line ~250:

**Before:**
```python
dataloader_num_workers=4,           # Parallel loading
dataloader_persistent_workers=True, # Reuse workers
```

**After:**
```python
dataloader_num_workers=2,           # Reduced to prevent fd leaks  
dataloader_persistent_workers=False, # Disable to prevent fd leaks
```

## Already Fixed in WASM Trainer

The WASM trainer has these fixes applied:
- `dataloader_num_workers=2` (reduced from 4)
- `dataloader_persistent_workers=False` (safer for long training)
- Less frequent evaluation (every ~200 steps vs ~50)

## Check Your Model

Your crashed training should have saved a checkpoint at step 5500. Check:

```bash
ls -la worldmodel_rocm_output_stable/
```

You can resume from the latest checkpoint if desired.

## Recommendation

Use the WASM trainer instead - it's more stable for long runs and has the live execution feature!

```bash
cd wasm/
python train_wasm_worldmodel.py --epochs 30
```