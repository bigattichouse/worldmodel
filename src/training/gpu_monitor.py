"""
GPU Temperature Monitor for ROCm
=================================
Monitors AMD GPU temperatures via rocm-smi and provides throttling
capabilities to prevent overheating during training.

Throttling strategy (progressive):
  1. Reduce power cap (step down by 25W)
  2. Lower GPU performance level to manual/low
  3. If still too hot after all soft steps → hard pause

Usage:
    from src.training.gpu_monitor import GPUThermalController
    
    controller = GPUThermalController(max_temp=99.0, safe_temp=85.0)
    controller.check_and_throttle()  # Call periodically during training
    controller.restore()             # Call at end of training
"""

import os
import time
import logging
import subprocess
from typing import Optional, Dict
from pathlib import Path
from enum import Enum

logger = logging.getLogger(__name__)

# Default thresholds
DEFAULT_MAX_TEMP = 99.0       # Celsius - start soft throttling above this
DEFAULT_SAFE_TEMP = 90.0      # Celsius - ease off throttling below this
DEFAULT_HARD_PAUSE_TEMP = 107.0  # Celsius - emergency hard stop (3C below 110C alarm)
DEFAULT_CHECK_INTERVAL = 30   # Seconds between temperature checks
DEFAULT_COOLDOWN_TIMEOUT = 600  # Max seconds to wait for cooldown

# Throttling steps (progressive)
POWER_STEP_WATTS = 25         # Reduce power cap by 25W per step
MIN_POWER_WATTS = 100         # Don't go below 100W


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


class ThrottleState(Enum):
    """Current throttling level applied to the GPU."""
    NORMAL = "normal"           # Full power, no throttling
    POWER_REDUCED = "reduced_power"  # Power cap lowered
    PERFORMANCE_LOW = "perf_low"      # Perf level set to low
    HARD_PAUSE = "hard_pause"         # Training must stop


class GPUThermalController:
    """
    Progressive GPU thermal controller.
    
    Escalates through soft throttling steps before resorting to hard pause:
      NORMAL → POWER_REDUCED → PERFORMANCE_LOW → HARD_PAUSE
    """

    def __init__(
        self,
        gpu_id: int = 0,
        max_temp: float = DEFAULT_MAX_TEMP,
        safe_temp: float = DEFAULT_SAFE_TEMP,
        hard_pause_temp: float = DEFAULT_HARD_PAUSE_TEMP,
        check_interval: int = DEFAULT_CHECK_INTERVAL,
        cooldown_timeout: int = DEFAULT_COOLDOWN_TIMEOUT,
    ):
        self.gpu_id = gpu_id
        self.max_temp = max_temp
        self.safe_temp = safe_temp
        self.hard_pause_temp = hard_pause_temp
        self.check_interval = check_interval
        self.cooldown_timeout = cooldown_timeout

        # Track original settings for restore
        self._original_power: Optional[float] = None
        self._original_perf: Optional[str] = None
        self._throttle_state = ThrottleState.NORMAL
        self._current_power_cap: Optional[float] = None

        # Save original state on init
        self._save_original_state()

    def _save_original_state(self):
        """Read and store original power cap and perf level."""
        # Get max power
        output = run_rocm_smi("--showmaxpower")
        if output:
            for line in output.splitlines():
                if f"GPU[{self.gpu_id}]" in line and "Power" in line:
                    try:
                        self._original_power = float(line.split(":")[-1].strip())
                    except ValueError:
                        pass

        # Get perf level
        output = run_rocm_smi("--showperflevel")
        if output:
            for line in output.splitlines():
                if f"GPU[{self.gpu_id}]" in line and "Performance" in line:
                    try:
                        self._original_perf = line.split(":")[-1].strip()
                    except ValueError:
                        pass

        self._current_power_cap = self._original_power

    @property
    def state(self) -> ThrottleState:
        return self._throttle_state

    def check_and_throttle(self) -> bool:
        """
        Check GPU temp and apply progressive throttling.

        Returns:
            True if training should pause, False if OK to continue.
        """
        current_temp = get_gpu_temp_celsius(self.gpu_id)
        if current_temp is None:
            return False  # Can't read temp, assume OK

        # Emergency hard stop — immediate pause regardless of current state
        if current_temp >= self.hard_pause_temp:
            self._throttle_state = ThrottleState.HARD_PAUSE
            logger.error(
                f"EMERGENCY: GPU at {current_temp:.1f}°C — hard pausing training "
                f"(alarm threshold: {self.hard_pause_temp:.0f}°C)"
            )
            return True

        # If we're below safe temp, restore towards normal
        if current_temp <= self.safe_temp and self._throttle_state != ThrottleState.NORMAL:
            self._relax_throttle(current_temp)
            return False

        # If above soft throttle temp, escalate throttling
        if current_temp >= self.max_temp:
            return self._escalate_throttle(current_temp)

        return False

    def _escalate_throttle(self, current_temp: float) -> bool:
        """Above max temp — try soft throttle, fall back to hard pause."""
        logger.warning(
            f"GPU at {current_temp:.1f}°C — escalating throttle from {self._throttle_state.value}"
        )

        if self._throttle_state == ThrottleState.NORMAL:
            self._reduce_power()
            # If power succeeded (state changed to POWER_REDUCED), keep training
            if self._throttle_state == ThrottleState.POWER_REDUCED:
                return False
            # Power failed (state set to HARD_PAUSE) — proceed to pause below

        if self._throttle_state == ThrottleState.POWER_REDUCED:
            new_power = (self._current_power_cap or MIN_POWER_WATTS) - POWER_STEP_WATTS
            if new_power >= MIN_POWER_WATTS:
                self._reduce_power()
                if self._throttle_state == ThrottleState.POWER_REDUCED:
                    return False
            # Can't go lower, try perf_low
            self._set_perf_low()
            if self._throttle_state == ThrottleState.PERFORMANCE_LOW:
                return False

        # All soft options exhausted or unavailable (no sudo)
        self._throttle_state = ThrottleState.HARD_PAUSE
        logger.warning(
            f"GPU at {current_temp:.1f}°C — pausing training until "
            f"GPU cools to {self.safe_temp:.1f}°C."
        )
        return True

    def _relax_throttle(self, current_temp: float):
        """Gradually restore throttling as GPU cools."""
        logger.info(
            f"GPU cooled to {current_temp:.1f}°C — relaxing throttle from {self._throttle_state.value}"
        )

        if self._throttle_state == ThrottleState.HARD_PAUSE:
            # Emergency pause released — go back to fully throttled first
            self._throttle_state = ThrottleState.PERFORMANCE_LOW
        elif self._throttle_state == ThrottleState.PERFORMANCE_LOW:
            self._restore_perf_level()
            self._throttle_state = ThrottleState.POWER_REDUCED
        elif self._throttle_state == ThrottleState.POWER_REDUCED:
            self._restore_power_cap()
            self._throttle_state = ThrottleState.NORMAL

    def _reduce_power(self):
        """Lower power cap by POWER_STEP_WATTS."""
        if self._current_power_cap is None:
            logger.warning("Cannot reduce power — original power unknown")
            return

        new_power = self._current_power_cap - POWER_STEP_WATTS
        if new_power < MIN_POWER_WATTS:
            new_power = MIN_POWER_WATTS

        try:
            result = subprocess.run(
                ["rocm-smi", "--setpoweroverdrive", str(int(new_power))],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0:
                self._current_power_cap = new_power
                self._throttle_state = ThrottleState.POWER_REDUCED
                logger.info(
                    f"GPU power cap reduced to {new_power:.0f}W "
                    f"(was {self._original_power:.0f}W)"
                )
            else:
                logger.debug(
                    f"Power overdrive failed (no sudo?): {result.stderr.strip()}"
                )
                # Mark that soft throttle is not available — escalate to hard pause
                self._throttle_state = ThrottleState.HARD_PAUSE
        except Exception as e:
            logger.debug(f"Error setting power overdrive: {e}")
            self._throttle_state = ThrottleState.HARD_PAUSE

    def _set_perf_low(self):
        """Set GPU performance level to low (minimum clocks)."""
        try:
            result = subprocess.run(
                ["rocm-smi", "--setperflevel", "low"],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0:
                self._throttle_state = ThrottleState.PERFORMANCE_LOW
                logger.info("GPU performance level set to 'low' (minimum clocks)")
            else:
                logger.warning(f"Failed to set perf level: {result.stderr}")
        except Exception as e:
            logger.warning(f"Error setting perf level: {e}")

    def _restore_power_cap(self):
        """Restore original power cap."""
        if self._original_power is None:
            return
        try:
            result = subprocess.run(
                ["rocm-smi", "--setpoweroverdrive", str(int(self._original_power))],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0:
                self._current_power_cap = self._original_power
                logger.info(f"GPU power cap restored to {self._original_power:.0f}W")
            else:
                logger.warning(f"Failed to restore power: {result.stderr}")
        except Exception as e:
            logger.warning(f"Error restoring power cap: {e}")

    def _restore_perf_level(self):
        """Restore original performance level."""
        if self._original_perf is None:
            return
        try:
            result = subprocess.run(
                ["rocm-smi", "--setperflevel", self._original_perf],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0:
                logger.info(f"GPU perf level restored to '{self._original_perf}'")
            else:
                logger.warning(f"Failed to restore perf: {result.stderr}")
        except Exception as e:
            logger.warning(f"Error restoring perf level: {e}")

    def restore(self):
        """Fully restore GPU to original state. Call at end of training."""
        logger.info("Restoring GPU to original settings...")
        self._restore_power_cap()
        self._restore_perf_level()
        self._throttle_state = ThrottleState.NORMAL
        self._current_power_cap = self._original_power

    def wait_for_cooldown(self) -> bool:
        """
        Wait until GPU temperature drops to safe level.
        Used when hard pause is triggered.
        """
        import torch

        start_time = time.time()
        while True:
            elapsed = time.time() - start_time
            if elapsed > self.cooldown_timeout:
                logger.error(
                    f"GPU cooldown timed out after {self.cooldown_timeout}s. "
                    f"Consider stopping training."
                )
                return False

            time.sleep(self.check_interval)

            # Clear CUDA cache to help with cooling
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            current_temp = get_gpu_temp_celsius(self.gpu_id)
            if current_temp is None:
                continue

            if current_temp <= self.safe_temp:
                wait_time = time.time() - start_time
                logger.info(
                    f"GPU cooled to {current_temp:.1f}°C after {wait_time:.0f}s. Resuming."
                )
                return True

            if int(elapsed) % 60 == 0:
                logger.info(f"Still cooling... {current_temp:.1f}°C ({int(elapsed)}s)")


# ─── Backwards-compatible convenience functions ─────────────────────────────

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
