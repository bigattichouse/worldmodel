"""
Auto-Converter: Python to WAT
=============================

Converts existing WorldModel training examples from Python code to WebAssembly Text format.
Handles the curriculum training pipeline conversion.
"""

import re
import ast
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class ConversionResult:
    """Result of converting Python code to WAT."""
    wat_code: str
    success: bool
    error: Optional[str] = None
    inputs: Optional[List] = None
    expected_output: Optional[float] = None


class PythonToWATConverter:
    """Converts simple Python arithmetic to WebAssembly Text format."""
    
    def __init__(self):
        self.arithmetic_ops = {
            ast.Add: {"i32": "i32.add", "f32": "f32.add", "f64": "f64.add"},
            ast.Sub: {"i32": "i32.sub", "f32": "f32.sub", "f64": "f64.sub"}, 
            ast.Mult: {"i32": "i32.mul", "f32": "f32.mul", "f64": "f64.mul"},
            ast.Div: {"i32": "i32.div_s", "f32": "f32.div", "f64": "f64.div"},
        }
    
    def convert_training_example(self, example_text: str) -> ConversionResult:
        """
        Convert a WorldModel training example to WAT format.
        
        Args:
            example_text: Full training example with User/Assistant format
            
        Returns:
            ConversionResult with WAT code and metadata
        """
        try:
            # Extract the Python code from <model> tags
            python_code = self._extract_model_code(example_text)
            if not python_code:
                return ConversionResult("", False, "No <model> code found")
            
            # Parse and convert to WAT
            wat_result = self._convert_python_to_wat(python_code)
            
            # Extract expected inputs/outputs from the example
            inputs, expected = self._extract_expected_values(example_text, python_code)
            
            return ConversionResult(
                wat_code=wat_result.wat_code,
                success=wat_result.success,
                error=wat_result.error,
                inputs=inputs,
                expected_output=expected
            )
            
        except Exception as e:
            return ConversionResult("", False, f"Conversion error: {e}")
    
    def _extract_model_code(self, example_text: str) -> Optional[str]:
        """Extract Python code from <model> tags."""
        match = re.search(r'<model>\s*(.*?)\s*</model>', example_text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return None
    
    def _extract_expected_values(self, example_text: str, python_code: str) -> Tuple[Optional[List], Optional[float]]:
        """Extract input values and expected output from the example."""
        inputs = []
        expected_output = None
        
        # Look for numbers in the user question
        user_match = re.search(r'User: (.*?)(?=Assistant:|$)', example_text, re.DOTALL)
        if user_match:
            user_text = user_match.group(1)
            numbers = re.findall(r'\b\d+(?:\.\d+)?\b', user_text)
            inputs = [float(n) for n in numbers]
        
        # Try to extract expected result from assistant response
        assistant_match = re.search(r'Assistant: .*?(\d+(?:\.\d+)?)', example_text, re.DOTALL)
        if assistant_match:
            try:
                expected_output = float(assistant_match.group(1))
            except ValueError:
                pass
        
        return inputs if inputs else None, expected_output
    
    def _convert_python_to_wat(self, python_code: str) -> ConversionResult:
        """Convert Python arithmetic expression to WAT."""
        try:
            # Parse Python code
            tree = ast.parse(python_code.strip())
            
            # Find the main computation
            computation_node = self._find_computation_node(tree)
            if not computation_node:
                return ConversionResult("", False, "No arithmetic computation found")
            
            # Convert to WAT
            wat_code = self._ast_to_wat(computation_node)
            
            return ConversionResult(wat_code, True)
            
        except SyntaxError as e:
            return ConversionResult("", False, f"Python syntax error: {e}")
        except Exception as e:
            return ConversionResult("", False, f"Conversion failed: {e}")
    
    def _find_computation_node(self, tree: ast.AST) -> Optional[ast.AST]:
        """Find the main arithmetic computation in the AST."""
        for node in ast.walk(tree):
            if isinstance(node, ast.BinOp):
                return node
            elif isinstance(node, ast.Assign) and isinstance(node.value, ast.BinOp):
                return node.value
        return None
    
    def _ast_to_wat(self, node: ast.AST) -> str:
        """Convert AST node to WebAssembly Text format."""
        if isinstance(node, ast.BinOp):
            return self._binop_to_wat(node)
        elif isinstance(node, ast.Constant):
            return self._constant_to_wat(node)
        elif isinstance(node, ast.Name):
            return f"local.get ${node.id}"
        else:
            raise ValueError(f"Unsupported AST node type: {type(node)}")
    
    def _binop_to_wat(self, node: ast.BinOp) -> str:
        """Convert binary operation to WAT."""
        # Determine number type (simplified - assumes float for now)
        num_type = "f64"
        
        if type(node.op) not in self.arithmetic_ops:
            raise ValueError(f"Unsupported operation: {type(node.op)}")
        
        op_code = self.arithmetic_ops[type(node.op)][num_type]
        
        # Simple two-parameter function
        wat_template = f"""(module
  (func $compute (param f64 f64) (result f64)
    local.get 0
    local.get 1
    {op_code}))"""
        
        return wat_template
    
    def _constant_to_wat(self, node: ast.Constant) -> str:
        """Convert constant to WAT."""
        if isinstance(node.value, (int, float)):
            return f"f64.const {float(node.value)}"
        else:
            raise ValueError(f"Unsupported constant type: {type(node.value)}")
    
    def convert_batch(self, examples: List[str]) -> List[ConversionResult]:
        """Convert multiple examples in batch."""
        results = []
        for example in examples:
            result = self.convert_training_example(example)
            results.append(result)
        return results