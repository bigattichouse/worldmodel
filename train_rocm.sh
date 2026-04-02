#!/usr/bin/env bash
# =============================================================================
# WorldModel Training Launcher for AMD MI50 (ROCm)
# =============================================================================
# Sets required environment variables for gfx906 / ROCm 7.1.1 + PyTorch 2.4.1
# See docs/rocm/ROCm_Training_Success_Guide.md for setup details.
#
# Usage:
#   ./train_rocm.sh [extra args passed to train.py]
#
# Examples:
#   ./train_rocm.sh                          # defaults
#   ./train_rocm.sh --epochs 15 --batch-size 1
#   ./train_rocm.sh --categories arithmetic,algebra,geometry
#   ./train_rocm.sh --resume ./output/worldmodel/checkpoint-400
# =============================================================================

set -e

# ─── ROCm environment (critical for MI50 gfx906) ────────────────────────────
export HSA_OVERRIDE_GFX_VERSION=9.0.6
export PYTORCH_ROCM_ARCH=gfx906
export TOKENIZERS_PARALLELISM=false
export HIP_VISIBLE_DEVICES=0
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True,max_split_size_mb:512
export OMP_NUM_THREADS=1
export HSA_DISABLE_CACHE=1
export HSA_FORCE_FINE_GRAIN_PCIE=1

# ─── Activate venv if present ────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "$SCRIPT_DIR/venv/bin/activate" ]; then
    source "$SCRIPT_DIR/venv/bin/activate"
    echo "Activated venv"
fi

# ─── Verify GPU ──────────────────────────────────────────────────────────────
echo "GPU info:"
rocm-smi 2>/dev/null | head -20 || echo "(rocm-smi not found, continuing)"
echo ""

# ─── Run training ────────────────────────────────────────────────────────────
echo "Starting training..."
echo "Args: $@"
echo ""

python3 "$SCRIPT_DIR/train.py" "$@"
