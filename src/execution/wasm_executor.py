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
            
            # Execute WASM using wasmtime Python runtime
            try:
                result = self._execute_with_wasmtime(wasm_file, inputs)
                return {
                    "success": True,
                    "result": result,
                    "error": None,
                    "computed_token": f"<computed>{result}</computed>"
                }
            except Exception as e:
                # Don't fallback - show the actual error
                print(f"Error: wasmtime execution failed: {e}")
                return {
                    "success": False,
                    "result": None,
                    "error": f"WASM execution failed: {e}",
                    "computed_token": f"<error>{e}</error>"
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
    
    def _validate_and_select_inputs(self, inputs: Optional[List[Any]], param_count: int, wat_code: str) -> List[float]:
        """
        Intelligently validate and select the correct number of inputs for WASM function.
        
        Args:
            inputs: Raw input list (may contain extra numbers)
            param_count: Expected parameter count from WASM function
            wat_code: WAT code to analyze for operation type
            
        Returns:
            List of validated inputs matching param_count
        """
        if not inputs:
            return []
            
        # Convert all inputs to floats, filter out invalid ones
        valid_numbers = []
        for inp in inputs:
            try:
                num = float(inp)
                # Filter out obvious non-operand numbers (like precision specs, iteration counts)
                if not (-1e10 <= num <= 1e10):  # Reasonable range check
                    continue
                valid_numbers.append(num)
            except (ValueError, TypeError):
                continue
        
        if param_count == 0:
            return []
        elif param_count == 1:
            # For unary operations, select the most relevant number
            if "square" in wat_code or "sqrt" in wat_code:
                # For squares/roots, prefer non-zero positive numbers
                positive_nums = [n for n in valid_numbers if n > 0]
                return [positive_nums[0]] if positive_nums else [valid_numbers[0]] if valid_numbers else [0]
            else:
                # Default: use first number
                return [valid_numbers[0]] if valid_numbers else [0]
        elif param_count == 2:
            # For binary operations, select the two most relevant numbers
            if len(valid_numbers) >= 2:
                # Smart selection based on operation type
                if any(op in wat_code for op in ["f64.div", "div"]):
                    # For division, avoid zero divisor
                    non_zero = [n for n in valid_numbers[1:] if n != 0]
                    if non_zero:
                        return [valid_numbers[0], non_zero[0]]
                    else:
                        # No non-zero divisor found, use first two anyway (will handle error later)
                        return valid_numbers[:2]
                else:
                    # For other operations, use first two numbers
                    return valid_numbers[:2]
            elif len(valid_numbers) == 1:
                # Only one number provided for binary op - duplicate it or use default
                return [valid_numbers[0], valid_numbers[0]]
            else:
                # No valid numbers - return defaults
                return [0.0, 1.0]
        else:
            # More than 2 parameters - use first N
            return (valid_numbers + [0.0] * param_count)[:param_count]
    
    def _execute_with_wasmtime(self, wasm_file: str, inputs: Optional[List[Any]]) -> Optional[float]:
        """Execute WASM file using wasmtime Python runtime."""
        try:
            from wasmtime import Store, Module, Instance, Func, FuncType, ValType
            
            # Load WASM module
            with open(wasm_file, 'rb') as f:
                wasm_bytes = f.read()
            
            store = Store()
            module = Module(store.engine, wasm_bytes)
            instance = Instance(store, module, [])
            
            # Find the compute function (standard name we use)
            compute_func = None
            exports = instance.exports(store)
            
            for name, value in exports.items():
                if name == "compute":
                    compute_func = value
                    break
            
            if not compute_func:
                # Try finding any exported function
                for name, value in exports.items():
                    if isinstance(value, Func):
                        compute_func = value
                        break
            
            if not compute_func:
                raise RuntimeError("No exported function found in WASM module")
            
            # Check function signature to determine parameter count
            func_type = compute_func.type(store)
            param_count = len(func_type.params)
            
            # Validate and select appropriate inputs
            # Note: For wasmtime execution, we can't access the original WAT code,
            # so we pass an empty string for wat_code parameter
            validated_inputs = self._validate_and_select_inputs(inputs, param_count, "")
            print(f"      Function expects {param_count} parameters, got {len(inputs) if inputs else 0}, using {len(validated_inputs)}")
            
            # Execute with validated parameters
            if param_count == 2 and len(validated_inputs) >= 2:
                # Binary operations
                result = compute_func(store, float(validated_inputs[0]), float(validated_inputs[1]))
            elif param_count == 1 and len(validated_inputs) >= 1:
                # Unary operations
                result = compute_func(store, float(validated_inputs[0]))
            elif param_count == 0:
                # No parameters
                result = compute_func(store)
            else:
                raise RuntimeError(f"Parameter validation failed: function expects {param_count}, but validated inputs: {len(validated_inputs)}")
            
            return float(result) if result is not None else None
            
        except ImportError:
            raise RuntimeError("wasmtime package not installed - install with: pip install wasmtime")
        except Exception as e:
            raise RuntimeError(f"WASM execution failed: {e}")
    
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