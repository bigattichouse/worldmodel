"""
GPU Temperature Monitor for ROCm
=================================
Monitors AMD GPU temperatures via rocm-smi and pauses training when
the GPU gets too hot. Resumes once it cools back down.

No soft throttling (power-capping / perf-level changes require sudo and
are unreliable).  The strategy is simple: keep training at full speed
until the temperature hits the limit, then hard-pause and wait.

Usage:
    from src.training.gpu_monitor import GPUThermalController

    controller = GPUThermalController(max_temp=88.0, safe_temp=80.0)
    controller.check_and_throttle()  # Call periodically during training
    controller.restore()             # Call at end of training (no-op now)
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
DEFAULT_MAX_TEMP = 99.0       # Celsius — hard-pause above this
DEFAULT_SAFE_TEMP = 92.0      # Celsius — resume below this
DEFAULT_CHECK_INTERVAL = 5    # Seconds between temperature checks
DEFAULT_COOLDOWN_TIMEOUT = 600  # Max seconds to wait for cooldown


def _find_sysfs_temp_files(gpu_id: int = 0):
    """Find sysfs temp file paths for the given GPU. Returns list of (path, label)."""
    import glob
    matches = sorted(glob.glob(
        f"/sys/class/drm/card*/device/hwmon/hwmon*/temp*_input"
    ))
    # temp2 is typically junction (hottest) on AMD GPUs
    if len(matches) >= 2:
        return matches  # return all, we'll pick temp2 (index 1)
    return matches


_SYSFS_TEMP_FILES = None

def _get_sysfs_temp(gpu_id: int = 0) -> Optional[float]:
    """Read GPU junction temperature from sysfs (fast, no subprocess)."""
    global _SYSFS_TEMP_FILES
    if _SYSFS_TEMP_FILES is None:
        _SYSFS_TEMP_FILES = _find_sysfs_temp_files(gpu_id)
        if not _SYSFS_TEMP_FILES:
            return None

    # temp2_input is typically the junction/hottest sensor
    if len(_SYSFS_TEMP_FILES) >= 2:
        path = _SYSFS_TEMP_FILES[1]
    else:
        path = _SYSFS_TEMP_FILES[0]

    try:
        with open(path) as f:
            return int(f.read().strip()) / 1000.0
    except (OSError, ValueError):
        return None


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
    """Current throttling state — binary: normal or paused."""
    NORMAL = "normal"
    PAUSED = "paused"


class GPUThermalController:
    """
    Pause-resume thermal controller.

    Checks temperature every *check_interval* seconds.  When the GPU
    exceeds *max_temp* it returns True (caller should pause training).
    Training resumes automatically once the GPU drops below *safe_temp*.
    """

    def __init__(
        self,
        gpu_id: int = 0,
        max_temp: float = DEFAULT_MAX_TEMP,
        safe_temp: float = DEFAULT_SAFE_TEMP,
        check_interval: int = DEFAULT_CHECK_INTERVAL,
        cooldown_timeout: int = DEFAULT_COOLDOWN_TIMEOUT,
    ):
        self.gpu_id = gpu_id
        self.max_temp = max_temp
        self.safe_temp = safe_temp
        self.check_interval = check_interval
        self.cooldown_timeout = cooldown_timeout

        self._throttle_state = ThrottleState.NORMAL
        self._cached_temp: Optional[float] = None
        self._cached_temp_time: float = 0.0

    @property
    def state(self) -> ThrottleState:
        return self._throttle_state

    def _get_temp(self) -> Optional[float]:
        """Get GPU junction temp from sysfs (fast, no subprocess)."""
        return _get_sysfs_temp(self.gpu_id)

    def check_and_throttle(self) -> bool:
        """
        Check GPU temp.  Returns True if we should hard-pause (>= max_temp).
        """
        current_temp = self._get_temp()
        if current_temp is None:
            return False

        if current_temp >= self.max_temp:
            self._throttle_state = ThrottleState.PAUSED
            logger.warning(
                f"GPU at {current_temp:.1f}°C (>= {self.max_temp:.1f}°C) "
                f"— pausing training until it cools to {self.safe_temp:.1f}°C"
            )
            return True

        if current_temp <= self.safe_temp:
            if self._throttle_state == ThrottleState.PAUSED:
                logger.info(
                    f"GPU cooled to {current_temp:.1f}°C (<= {self.safe_temp:.1f}°C) — resuming"
                )
            self._throttle_state = ThrottleState.NORMAL

        return False

    @property
    def sleep_time(self) -> float:
        """Unused — kept for backwards compat."""
        return 0.0

    def restore(self):
        """Restore state marker. No GPU changes needed (no soft throttling)."""
        logger.info("Restoring GPU thermal controller state")
        self._throttle_state = ThrottleState.NORMAL

    def wait_for_cooldown(self) -> bool:
        """
        Wait until GPU temperature drops to safe level.
        Adds a 5-second buffer after reaching safe temp to prevent immediate re-spike.
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

            current_temp = _get_sysfs_temp(self.gpu_id)
            if current_temp is None:
                continue

            if current_temp <= self.safe_temp:
                # Extra buffer cooldown so GPU doesn't immediately spike
                time.sleep(5)
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
