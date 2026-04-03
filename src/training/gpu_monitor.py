"""
GPU Temperature Monitor for ROCm
=================================
Monitors AMD GPU temperatures via rocm-smi and provides throttling
capabilities to prevent overheating during training.

Usage:
    from src.training.gpu_monitor import get_gpu_temp, wait_for_cooldown
    
    # Check current temperature
    temp = get_gpu_temp()
    
    # Wait if too hot
    wait_for_cooldown(max_temp=85.0)
"""

import os
import time
import logging
import subprocess
from typing import Optional, Dict
from pathlib import Path

logger = logging.getLogger(__name__)

# Default thresholds
DEFAULT_MAX_TEMP = 85.0       # Celsius - pause training above this
DEFAULT_SAFE_TEMP = 75.0      # Celsius - resume training below this
DEFAULT_CHECK_INTERVAL = 30   # Seconds between temperature checks
DEFAULT_COOLDOWN_TIMEOUT = 600  # Max seconds to wait for cooldown


def run_rocm_smi(command: str) -> Optional[str]:
    """Execute rocm-smi command and return output."""
    try:
        result = subprocess.run(
            ["rocm-smi", command],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            return result.stdout
        return None
    except (subprocess.TimeoutExpired, FileNotFoundError, Exception) as e:
        logger.debug(f"rocm-smi {command} failed: {e}")
        return None


def get_gpu_temp(gpu_id: int = 0) -> Optional[Dict[str, float]]:
    """
    Get GPU temperatures from rocm-smi.
    
    Returns dict with 'edge', 'junction', 'memory' temperatures in Celsius,
    or None if rocm-smi is unavailable.
    """
    output = run_rocm_smi("--showtemp")
    if output is None:
        return None
    
    temps = {}
    try:
        for line in output.splitlines():
            line = line.strip()
            if f"GPU[{gpu_id}]" not in line:
                continue
            
            if "edge" in line.lower():
                # Parse: GPU[0] : Temperature (Sensor edge) (C): 37.0
                parts = line.split(":")
                if len(parts) >= 2:
                    try:
                        temps["edge"] = float(parts[-1].strip())
                    except ValueError:
                        pass
            
            elif "junction" in line.lower():
                parts = line.split(":")
                if len(parts) >= 2:
                    try:
                        temps["junction"] = float(parts[-1].strip())
                    except ValueError:
                        pass
            
            elif "memory" in line.lower():
                parts = line.split(":")
                if len(parts) >= 2:
                    try:
                        temps["memory"] = float(parts[-1].strip())
                    except ValueError:
                        pass
    except Exception as e:
        logger.warning(f"Failed to parse rocm-smi output: {e}")
        return None
    
    return temps if temps else None


def get_gpu_temp_celsius(gpu_id: int = 0) -> Optional[float]:
    """
    Get the primary GPU temperature (junction/hottest) in Celsius.
    Falls back to edge temp if junction not available.
    """
    temps = get_gpu_temp(gpu_id)
    if temps is None:
        return None
    
    # Junction temp is typically the hottest and most critical
    if "junction" in temps:
        return temps["junction"]
    elif "edge" in temps:
        return temps["edge"]
    elif "memory" in temps:
        return temps["memory"]
    
    return None


def log_gpu_status(gpu_id: int = 0):
    """Log current GPU temperature and memory usage."""
    import torch
    
    temps = get_gpu_temp(gpu_id)
    if temps:
        temp_str = ", ".join([f"{k}: {v:.1f}°C" for k, v in temps.items()])
        logger.info(f"GPU{gpu_id} temps: {temp_str}")
    else:
        logger.warning("Could not read GPU temperatures")
    
    if torch.cuda.is_available():
        used_gb = torch.cuda.memory_allocated() / 1e9
        reserved_gb = torch.cuda.memory_reserved() / 1e9
        logger.info(f"GPU{gpu_id} VRAM: {used_gb:.1f}GB used / {reserved_gb:.1f}GB reserved")


def wait_for_cooldown(
    max_temp: float = DEFAULT_MAX_TEMP,
    safe_temp: float = DEFAULT_SAFE_TEMP,
    check_interval: int = DEFAULT_CHECK_INTERVAL,
    timeout: int = DEFAULT_COOLDOWN_TIMEOUT,
    gpu_id: int = 0,
):
    """
    Wait until GPU temperature drops to safe level.
    
    Args:
        max_temp: Temperature above which to start waiting
        safe_temp: Temperature to wait until before continuing
        check_interval: Seconds between temperature checks
        timeout: Maximum seconds to wait before giving up
        gpu_id: GPU device ID to monitor
    
    Returns:
        True if cooled down successfully, False if timed out
    """
    import torch
    
    current_temp = get_gpu_temp_celsius(gpu_id)
    if current_temp is None:
        logger.warning("Cannot read GPU temperature, skipping cooldown check")
        return True
    
    if current_temp <= safe_temp:
        return True
    
    logger.warning(
        f"GPU temperature {current_temp:.1f}°C exceeds safe threshold {safe_temp:.1f}°C. "
        f"Pausing training to cool down..."
    )
    
    start_time = time.time()
    while True:
        elapsed = time.time() - start_time
        if elapsed > timeout:
            logger.error(
                f"GPU cooldown timed out after {timeout}s. "
                f"Current temp: {current_temp:.1f}°C, target: {safe_temp:.1f}°C"
            )
            return False
        
        time.sleep(check_interval)
        
        # Clear CUDA cache to help with cooling
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        current_temp = get_gpu_temp_celsius(gpu_id)
        if current_temp is None:
            logger.warning("Lost GPU temperature reading during cooldown")
            continue
        
        if current_temp <= safe_temp:
            wait_time = time.time() - start_time
            logger.info(
                f"GPU cooled down to {current_temp:.1f}°C after {wait_time:.0f}s. "
                f"Resuming training."
            )
            return True
        
        if int(elapsed) % 60 == 0:
            logger.info(f"Still cooling... {current_temp:.1f}°C ({int(elapsed)}s elapsed)")


def check_and_throttle(
    max_temp: float = DEFAULT_MAX_TEMP,
    safe_temp: float = DEFAULT_SAFE_TEMP,
    gpu_id: int = 0,
) -> bool:
    """
    Check GPU temperature and throttle if needed.
    
    Returns True if training should pause, False if OK to continue.
    """
    current_temp = get_gpu_temp_celsius(gpu_id)
    if current_temp is None:
        return False
    
    if current_temp >= max_temp:
        logger.warning(
            f"GPU temperature {current_temp:.1f}°C exceeds max threshold {max_temp:.1f}°C"
        )
        return True
    
    return False


def emergency_shutdown(gpu_id: int = 0):
    """Log emergency shutdown message with current temps."""
    temps = get_gpu_temp(gpu_id)
    temp_str = ", ".join([f"{k}: {v:.1f}°C" for k, v in temps.items()]) if temps else "unknown"
    logger.critical(
        f"EMERGENCY: GPU{gpu_id} overheating detected! Temps: {temp_str}. "
        f"Training halted to prevent hardware damage."
    )
