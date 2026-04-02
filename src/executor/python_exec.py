"""
Inline Python executor for the WorldModel inference engine.

Executes code blocks in a persistent namespace (shared state across blocks
within one response). Uses threading + timeout to prevent runaway code.

For complex tasks needing real file I/O or packages, see vm_exec.py.

Tool request protocol
---------------------
The executor namespace always contains ``use_tool(name, **kwargs)`` and
``ToolNotAvailableError``.  Code can call::

    result = use_tool("web_search", query="...")

If the runtime has registered that tool it will run and return the result.
Otherwise ``ToolNotAvailableError`` is raised, which the model catches and
handles gracefully (explaining what it needs and offering an alternative).

To register tools at runtime::

    executor = PythonExecutor()
    executor.register_tool("web_search", my_search_fn)
"""

import sys
import io
import threading
import traceback
from typing import Dict, Any, Callable, Optional


class ToolNotAvailableError(Exception):
    """Raised when use_tool() is called for a tool not registered in this runtime."""
    pass


def _make_use_tool(registry: Dict[str, Callable]):
    """Return a use_tool() function bound to the given registry."""
    def use_tool(name: str, **kwargs):
        if name not in registry:
            available = list(registry.keys()) or ["(none)"]
            raise ToolNotAvailableError(
                f"Tool '{name}' is not available in this environment. "
                f"Available tools: {available}. "
                f"To use '{name}', the runtime needs to register it."
            )
        return registry[name](**kwargs)
    return use_tool


class ExecutionResult:
    def __init__(self, stdout: str, stderr: str, success: bool, timed_out: bool = False):
        self.stdout = stdout
        self.stderr = stderr
        self.success = success
        self.timed_out = timed_out

    def output_text(self) -> str:
        """Return the text to inject as <output> content."""
        if self.timed_out:
            return "ERROR: Execution timed out (exceeded time limit)"
        if self.stdout and self.stderr:
            return self.stdout.rstrip() + "\nSTDERR: " + self.stderr.rstrip()
        if self.stdout:
            return self.stdout.rstrip()
        if self.stderr:
            return "ERROR: " + self.stderr.rstrip()
        return "(no output)"


class PythonExecutor:
    """
    Executes Python code blocks in a managed namespace.

    Each PythonExecutor instance maintains shared state (namespace) across
    multiple code blocks within a single response. Create a new instance
    for each user turn to reset state.
    """

    DEFAULT_TIMEOUT = 10.0  # seconds

    def __init__(self, timeout: float = DEFAULT_TIMEOUT):
        self.timeout = timeout
        self._tool_registry: Dict[str, Callable] = {}
        self.namespace: Dict[str, Any] = self._make_namespace()

    def _make_namespace(self) -> Dict[str, Any]:
        return {
            "__builtins__": __builtins__,
            "ToolNotAvailableError": ToolNotAvailableError,
            "use_tool": _make_use_tool(self._tool_registry),
        }

    def register_tool(self, name: str, fn: Callable) -> None:
        """Make a tool callable from within code blocks."""
        self._tool_registry[name] = fn
        # Refresh use_tool in namespace so it sees the updated registry
        self.namespace["use_tool"] = _make_use_tool(self._tool_registry)

    def run(self, code: str) -> ExecutionResult:
        """
        Execute a code block. Returns stdout/stderr captured from execution.
        Namespace persists: variables defined here are available in later calls.
        """
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()
        result_container = [None]
        exception_container = [None]

        def target():
            old_stdout = sys.stdout
            old_stderr = sys.stderr
            sys.stdout = stdout_capture
            sys.stderr = stderr_capture
            try:
                exec(code, self.namespace)
                result_container[0] = True
            except Exception:
                exception_container[0] = traceback.format_exc()
                result_container[0] = False
            finally:
                sys.stdout = old_stdout
                sys.stderr = old_stderr

        thread = threading.Thread(target=target, daemon=True)
        thread.start()
        thread.join(timeout=self.timeout)

        if thread.is_alive():
            # Can't kill the thread cleanly in Python, but we can report it
            return ExecutionResult("", "", False, timed_out=True)

        stdout = stdout_capture.getvalue()
        stderr = stderr_capture.getvalue()

        if exception_container[0]:
            return ExecutionResult(stdout, exception_container[0], success=False)

        return ExecutionResult(stdout, stderr, success=True)

    def reset(self):
        """Clear the namespace (call between user turns). Keeps registered tools."""
        self.namespace = self._make_namespace()


def run_once(code: str, timeout: float = PythonExecutor.DEFAULT_TIMEOUT) -> ExecutionResult:
    """Convenience: run code in a fresh namespace, no state persistence."""
    executor = PythonExecutor(timeout=timeout)
    return executor.run(code)
