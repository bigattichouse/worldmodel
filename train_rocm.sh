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
export HIP_PLATFORM=amd
export ROCBLAS_LAYER=0

# ─── ROCm library path (PyTorch 2.4.1+rocm6.0 needs these at runtime) ───────
# Use whichever ROCm version is installed
if [ -d /opt/rocm-7.2.0/lib ]; then
    ROCM_LIB=/opt/rocm-7.2.0/lib
    ROCM_BIN=/opt/rocm-7.2.0/bin
elif [ -d /opt/rocm-7.1.1/lib ]; then
    ROCM_LIB=/opt/rocm-7.1.1/lib
    ROCM_BIN=/opt/rocm-7.1.1/bin
elif [ -d /opt/rocm/lib ]; then
    ROCM_LIB=/opt/rocm/lib
    ROCM_BIN=/opt/rocm/bin
fi

if [ -n "$ROCM_LIB" ]; then
    export LD_LIBRARY_PATH="${ROCM_LIB}:${ROCM_LIB}/llvm/lib:${LD_LIBRARY_PATH}"
    export PATH="${ROCM_BIN}:${PATH}"
    export ROCM_PATH="$(dirname $ROCM_LIB)"
    echo "ROCm library path: ${ROCM_LIB}"
else
    echo "WARNING: Could not find ROCm library directory"
fi

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

# ─── Pre-training temperature check ─────────────────────────────────────────
echo "Checking GPU temperature before training..."
TEMP_OUTPUT=$(rocm-smi --showtemp 2>/dev/null)
if [ $? -eq 0 ]; then
    # Extract junction temperature (hottest sensor)
    JUNCTION_TEMP=$(echo "$TEMP_OUTPUT" | grep "junction" | grep -oP '[\d.]+(?=\s*$)')
    if [ -n "$JUNCTION_TEMP" ]; then
        echo "Current GPU junction temp: ${JUNCTION_TEMP}°C"
        # Check if already too hot (using bc for float comparison)
        TOO_HOT=$(echo "$JUNCTION_TEMP > 99" | bc -l 2>/dev/null)
        if [ "$TOO_HOT" = "1" ]; then
            echo "WARNING: GPU temperature too high (${JUNCTION_TEMP}°C). Waiting to cool down..."
            while true; do
                sleep 30
                NEW_TEMP=$(rocm-smi --showtemp 2>/dev/null | grep "junction" | grep -oP '[\d.]+(?=\s*$)')
                if [ -z "$NEW_TEMP" ]; then
                    echo "Lost temperature reading, proceeding anyway..."
                    break
                fi
                echo "Current temp: ${NEW_TEMP}°C"
                COOLED=$(echo "$NEW_TEMP < 90" | bc -l 2>/dev/null)
                if [ "$COOLED" = "1" ]; then
                    echo "GPU cooled down to ${NEW_TEMP}°C. Starting training."
                    break
                fi
                echo "Still cooling... (${NEW_TEMP}°C)"
            done
        else
            echo "GPU temperature OK. Starting training."
        fi
    else
        echo "Could not parse GPU temperature, proceeding anyway..."
    fi
else
    echo "rocm-smi not available, skipping temperature check."
fi
echo ""

# ─── Run training ────────────────────────────────────────────────────────────
echo "Starting training..."
echo "Args: $@"
echo ""

# Determine log directory from --output arg or default
LOG_DIR="./output/worldmodel"
for arg in "$@"; do
    case "$arg" in
        --output) LOG_DIR="$2"; shift 2 ;;
        --output=*) LOG_DIR="${arg#--output=}" ;;
    esac
done
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/shell.log"

echo "Shell log: $LOG_FILE"
echo "Started: $(date)" >> "$LOG_FILE"
echo "Args: $@" >> "$LOG_FILE"
echo "---" >> "$LOG_FILE"

python3 "$SCRIPT_DIR/train.py" "$@" 2>&1 | tee -a "$LOG_FILE"
