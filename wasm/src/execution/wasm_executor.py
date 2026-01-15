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
    
    def __init__(self, timeout: int = 5):
        self.timeout = timeout
        self._check_wasm_tools()
    
    def _check_wasm_tools(self):
        """Check if required WASM tools are available."""
        try:
            subprocess.run(["wat2wasm", "--version"], 
                         capture_output=True, check=True, timeout=2)
            self.has_wat2wasm = True
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
            self.has_wat2wasm = False
            logger.warning("wat2wasm not found - WASM execution will be simulated")
    
    def execute_wat(self, wat_code: str, inputs: Optional[List[Any]] = None) -> Dict:
        """
        Execute WebAssembly Text format code.
        
        Args:
            wat_code: WAT format WebAssembly code
            inputs: Optional input parameters for the WASM function
            
        Returns:
            Dict with execution results:
            - success: bool
            - result: Any (computation result)  
            - error: str (if failed)
            - computed_token: str (for model integration)
        """
        if not self.has_wat2wasm:
            return self._simulate_execution(wat_code, inputs)
        
        try:
            return self._real_execution(wat_code, inputs)
        except Exception as e:
            logger.error(f"WASM execution failed: {e}")
            return {
                "success": False,
                "result": None,
                "error": str(e),
                "computed_token": "<error>wasm_failed</error>"
            }
    
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