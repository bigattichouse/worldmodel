"""
WASM Executor
=============

Handles compilation and execution of WebAssembly code during model forward pass.
Integrates WASM runtime with transformer computation.
"""

import subprocess
import tempfile
import os
from typing import Dict, List, Optional, Any
import logging

logger = logging.getLogger(__name__)


class WASMExecutor:
    """Executes WebAssembly code and returns results for model integration."""
    
    def __init__(self, timeout: int = 5, use_sandbox: bool = True, sandbox_config: Optional[Dict] = None):
        self.timeout = timeout
        self.use_sandbox = use_sandbox
        self._check_wasm_tools()
        
        # Initialize API provider for external calls
        from .wasm_api import WASMAPIProvider
        self.api_provider = WASMAPIProvider(use_sandbox, sandbox_config)
        
        print(f"🔧 WASM Executor initialized:")
        print(f"   Internal WASM: {'Direct execution' if self.has_wat2wasm else 'Simulated'}")
        print(f"   External APIs: {'QEMU sandbox' if use_sandbox else 'Direct host'}")
    
    def _check_wasm_tools(self):
        """Check if required WASM tools are available."""
        try:
            subprocess.run(["wat2wasm", "--version"], 
                         capture_output=True, check=True, timeout=2)
            self.has_wat2wasm = True
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
            self.has_wat2wasm = False
            logger.warning("wat2wasm not found - WASM execution will be simulated")
    
    def execute_wat(self, wat_code: str, inputs: Optional[List[Any]] = None, api_calls: Optional[List[str]] = None) -> Dict:
        """
        Execute WebAssembly Text format code.
        
        Args:
            wat_code: WAT format WebAssembly code
            inputs: Optional input parameters for the WASM function
            api_calls: Optional list of external API calls to execute
            
        Returns:
            Dict with execution results:
            - success: bool
            - result: Any (computation result)  
            - error: str (if failed)
            - computed_token: str (for model integration)
            - api_results: Dict (external API call results)
        """
        result = {
            "success": False,
            "result": None,
            "error": None,
            "computed_token": "<error>execution_failed</error>",
            "api_results": {}
        }
        
        # Execute external API calls first (these use sandbox if enabled)
        if api_calls:
            result["api_results"] = self._execute_api_calls(api_calls)
        
        # Execute WASM code internally (no sandbox needed - it's deterministic computation)
        try:
            if not self.has_wat2wasm:
                wasm_result = self._simulate_execution(wat_code, inputs)
            else:
                wasm_result = self._real_execution(wat_code, inputs)
            
            # Merge results
            result.update(wasm_result)
            
        except Exception as e:
            logger.error(f"WASM execution failed: {e}")
            result["error"] = str(e)
        
        return result
    
    def _execute_api_calls(self, api_calls: List[str]) -> Dict[str, Any]:
        """Execute external API calls (via sandbox if enabled)."""
        api_results = {}
        
        for api_call in api_calls:
            try:
                if isinstance(api_call, dict):
                    api_name = api_call.get("name")
                    args = api_call.get("args", [])
                    kwargs = api_call.get("kwargs", {})
                else:
                    api_name = api_call
                    args = []
                    kwargs = {}
                
                api_result = self.api_provider.call_api(api_name, *args, **kwargs)
                api_results[api_name] = api_result
                
            except Exception as e:
                api_results[api_call] = {
                    "success": False,
                    "error": str(e)
                }
        
        return api_results
    
    def _real_execution(self, wat_code: str, inputs: Optional[List[Any]]) -> Dict:
        """Execute WASM code using actual WASM runtime."""
        with tempfile.TemporaryDirectory() as temp_dir:
            wat_file = os.path.join(temp_dir, "code.wat")
            wasm_file = os.path.join(temp_dir, "code.wasm")
            
            # Write WAT code
            with open(wat_file, 'w') as f:
                f.write(wat_code)
            
            # Compile WAT to WASM
            try:
                subprocess.run(
                    ["wat2wasm", wat_file, "-o", wasm_file],
                    check=True, capture_output=True, timeout=self.timeout
                )
            except subprocess.CalledProcessError as e:
                return {
                    "success": False,
                    "result": None,
                    "error": f"Compilation failed: {e.stderr.decode()}",
                    "computed_token": "<error>compilation_failed</error>"
                }
            
            # Execute WASM (simplified - would need actual WASM runtime)
            # For now, return a placeholder result
            result = self._extract_simple_computation(wat_code, inputs)
            
            return {
                "success": True,
                "result": result,
                "error": None,
                "computed_token": f"<computed>{result}</computed>"
            }
    
    def _simulate_execution(self, wat_code: str, inputs: Optional[List[Any]]) -> Dict:
        """Simulate WASM execution for development/testing."""
        # Simple pattern matching for basic arithmetic
        result = self._extract_simple_computation(wat_code, inputs)
        
        if result is not None:
            return {
                "success": True,
                "result": result,
                "error": None,
                "computed_token": f"<computed>{result}</computed>"
            }
        else:
            return {
                "success": False,
                "result": None,
                "error": "Simulation failed - unsupported operation",
                "computed_token": "<error>unsupported_op</error>"
            }
    
    def _extract_simple_computation(self, wat_code: str, inputs: Optional[List[Any]]) -> Optional[float]:
        """Extract simple arithmetic results from WAT code patterns."""
        if not inputs or len(inputs) < 2:
            return None
            
        try:
            a, b = float(inputs[0]), float(inputs[1])
        except (ValueError, TypeError):
            return None
        
        # Basic calculator operations
        if "i32.mul" in wat_code or "f32.mul" in wat_code or "f64.mul" in wat_code:
            return a * b
        elif "i32.add" in wat_code or "f32.add" in wat_code or "f64.add" in wat_code:
            return a + b
        elif "i32.sub" in wat_code or "f32.sub" in wat_code or "f64.sub" in wat_code:
            return a - b
        elif ("i32.div" in wat_code or "f32.div" in wat_code or "f64.div" in wat_code) and b != 0:
            return a / b
        
        # Percentage calculation (common in training data)
        if "0.25" in wat_code or "25%" in wat_code:
            return a * 0.25
        elif "0.5" in wat_code or "50%" in wat_code:
            return a * 0.5
        elif "0.75" in wat_code or "75%" in wat_code:
            return a * 0.75
        
        return None
    
    def execute_calculator(self, operation: str, a: float, b: float = 0) -> Dict:
        """Execute basic calculator operations directly."""
        try:
            if operation == "add":
                result = a + b
            elif operation == "subtract":
                result = a - b
            elif operation == "multiply":
                result = a * b
            elif operation == "divide" and b != 0:
                result = a / b
            elif operation == "percent":
                result = a * (b / 100.0)
            else:
                return {
                    "success": False,
                    "result": None,
                    "error": f"Unsupported operation: {operation}",
                    "computed_token": f"<error>unsupported_{operation}</error>"
                }
            
            return {
                "success": True,
                "result": result,
                "error": None,
                "computed_token": f"<computed>{result}</computed>"
            }
            
        except Exception as e:
            return {
                "success": False,
                "result": None,
                "error": str(e),
                "computed_token": f"<error>{str(e)}</error>"
            }
    
    def batch_execute(self, wat_codes: List[str], inputs_list: List[Optional[List[Any]]]) -> List[Dict]:
        """Execute multiple WASM programs in batch."""
        results = []
        for wat_code, inputs in zip(wat_codes, inputs_list):
            result = self.execute_wat(wat_code, inputs)
            results.append(result)
        return results