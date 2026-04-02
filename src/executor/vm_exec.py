"""
Scratchpad VM executor for the WorldModel inference engine.

Uses the QEMU-based scratchpad VM at ~/workspace/scratchpad/ for code
that needs: real file I/O, pip packages, subprocess, network access.

This is slower than python_exec.py (VM startup + SSH overhead).
Use it deliberately for complex tasks that exceed inline exec() capabilities.

See: ~/workspace/scratchpad/README.md
"""

import subprocess
import os
import tempfile
from pathlib import Path
from typing import Optional
from .python_exec import ExecutionResult

SCRATCHPAD_PATH = Path.home() / "workspace" / "scratchpad"
SCRATCHPAD_CLI = SCRATCHPAD_PATH / "node" / "node_modules" / ".bin" / "scratchpad"


class VMExecutor:
    """
    Runs Python code in the scratchpad QEMU VM via SSH.

    The VM provides a full Python environment with package access.
    State is NOT shared between calls unless using a persistent VM.
    """

    def __init__(self, vm_name: str = "worldmodel-exec", timeout: float = 30.0):
        self.vm_name = vm_name
        self.timeout = timeout

    def run(self, code: str) -> ExecutionResult:
        """
        Execute Python code in the VM. Writes code to a temp file,
        copies it in, runs it, returns stdout/stderr.
        """
        with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
            f.write(code)
            tmp_path = f.name

        try:
            result = subprocess.run(
                ["scratchpad-cli", "run", "--name", self.vm_name,
                 "--", "python3", tmp_path],
                capture_output=True,
                text=True,
                timeout=self.timeout,
                cwd=str(SCRATCHPAD_PATH / "node")
            )
            success = result.returncode == 0
            return ExecutionResult(
                stdout=result.stdout,
                stderr=result.stderr,
                success=success
            )
        except subprocess.TimeoutExpired:
            return ExecutionResult("", "", False, timed_out=True)
        except FileNotFoundError:
            return ExecutionResult(
                "",
                "VM executor not available (scratchpad-cli not found). "
                "Use python_exec.PythonExecutor for inline execution.",
                False
            )
        finally:
            os.unlink(tmp_path)


def is_available() -> bool:
    """Check whether the scratchpad VM tool is installed."""
    try:
        subprocess.run(["scratchpad-cli", "--version"],
                       capture_output=True, timeout=5)
        return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False
