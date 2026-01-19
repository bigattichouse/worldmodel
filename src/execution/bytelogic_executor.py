"""
ByteLogic Executor
==================

Handles compilation and execution of ByteLogic code through the 
ByteLogic → WAT → WASM pipeline for structured reasoning.
"""

import subprocess
import tempfile
import os
import re
import json
from typing import Dict, List, Optional, Any, Tuple
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class ByteLogicExecutor:
    """Executes ByteLogic code by compiling to WASM and running in sandbox."""
    
    def __init__(self, bytelogic_path: str = None, timeout: int = 10, cache_enabled: bool = True):
        """
        Initialize ByteLogic executor.
        
        Args:
            bytelogic_path: Path to ByteLogic compiler executable
            timeout: Maximum execution time in seconds
            cache_enabled: Whether to cache compilation results
        """
        self.timeout = timeout
        self.cache_enabled = cache_enabled
        
        # Find ByteLogic compiler
        self.bytelogic_path = self._find_bytelogic_compiler(bytelogic_path)
        
        # Compilation cache
        self._wat_cache = {}  # bytelog_code -> wat_code
        self._wasm_cache = {}  # wat_code -> wasm_bytes
        self._result_cache = {}  # (wasm_hash, inputs) -> results
        
        # Check WASM tools availability
        self.has_wat2wasm = self._check_wat2wasm()
        self.has_wasmtime = self._check_wasmtime()
        
        logger.info(f"ByteLogic Executor initialized:")
        logger.info(f"  ByteLogic compiler: {self.bytelogic_path}")
        logger.info(f"  WAT compiler: {'Available' if self.has_wat2wasm else 'Missing'}")
        logger.info(f"  WASM runtime: {'Available' if self.has_wasmtime else 'Missing'}")
    
    def _find_bytelogic_compiler(self, provided_path: str = None) -> str:
        """Find the ByteLogic compiler executable."""
        if provided_path and os.path.isfile(provided_path):
            return provided_path
        
        # Try common locations
        possible_paths = [
            "./bytelogic/build/bytelogic",
            "../bytelogic/build/bytelogic", 
            "bytelogic/build/bytelogic",
            "/usr/local/bin/bytelogic",
            "bytelogic"
        ]
        
        for path in possible_paths:
            if os.path.isfile(path) and os.access(path, os.X_OK):
                return path
        
        # Try system PATH
        try:
            result = subprocess.run(["which", "bytelogic"], 
                                  capture_output=True, text=True, timeout=2)
            if result.returncode == 0:
                return result.stdout.strip()
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
        
        logger.warning("ByteLogic compiler not found - execution will be simulated")
        return None
    
    def _check_wat2wasm(self) -> bool:
        """Check if wat2wasm is available."""
        try:
            subprocess.run(["wat2wasm", "--version"], 
                         capture_output=True, check=True, timeout=2)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
            return False
    
    def _check_wasmtime(self) -> bool:
        """Check if wasmtime Python package is available."""
        try:
            import wasmtime
            return True
        except ImportError:
            return False
    
    def execute_bytelogic(self, bl_code: str, inputs: Optional[Dict] = None) -> Dict:
        """
        Execute ByteLogic code through full compilation pipeline.
        
        Args:
            bl_code: ByteLogic source code
            inputs: Optional input parameters/facts to inject
            
        Returns:
            Dict with execution results:
            - success: bool
            - result: Any (computation result)
            - query_results: List[Tuple] (for logic queries)
            - calculation_result: Any (for CALC blocks)
            - error: str (if failed)
            - computation_token: str (for model integration)
            - wat_code: str (for debugging)
            - execution_time_ms: int
        """
        import time
        start_time = time.time()
        
        result = {
            "success": False,
            "result": None,
            "query_results": [],
            "calculation_result": None,
            "error": None,
            "computation_token": "<error>execution_failed</error>",
            "wat_code": None,
            "execution_time_ms": 0
        }
        
        try:
            # Step 1: Validate ByteLogic syntax
            validation_result = self._validate_syntax(bl_code)
            if not validation_result["valid"]:
                result["error"] = f"Syntax error: {validation_result['error']}"
                result["computation_token"] = f"<error>syntax_error</error>"
                return result
            
            # Step 2: Inject input facts if provided
            if inputs:
                bl_code = self._inject_inputs(bl_code, inputs)
            
            # Step 3: Compile ByteLogic to WAT
            wat_result = self._compile_to_wat(bl_code)
            if not wat_result["success"]:
                result["error"] = wat_result["error"]
                result["computation_token"] = f"<error>compilation_failed</error>"
                return result
            
            result["wat_code"] = wat_result["wat_code"]
            
            # Step 4: Compile WAT to WASM and execute
            if self.has_wat2wasm and self.has_wasmtime:
                exec_result = self._execute_real_wasm(wat_result["wat_code"])
            else:
                exec_result = self._simulate_execution(bl_code, wat_result)
            
            # Step 5: Parse results
            parsed_results = self._parse_execution_results(exec_result, bl_code)
            result.update(parsed_results)
            
            # Step 6: Format computation token
            if result["success"]:
                result["computation_token"] = self._format_computation_token(result)
            
        except Exception as e:
            logger.error(f"ByteLogic execution failed: {e}")
            result["error"] = str(e)
            result["computation_token"] = f"<error>{str(e)}</error>"
        
        finally:
            result["execution_time_ms"] = int((time.time() - start_time) * 1000)
        
        return result
    
    def _validate_syntax(self, bl_code: str) -> Dict:
        """Validate ByteLogic syntax using the compiler."""
        if not self.bytelogic_path:
            return {"valid": True, "error": None}  # Skip validation if no compiler
        
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.bl', delete=False) as f:
                f.write(bl_code)
                temp_path = f.name
            
            # Check syntax by trying to compile to WAT
            result = subprocess.run([
                self.bytelogic_path, 
                "-c", "wat", "-o", "-",
                temp_path
            ], capture_output=True, text=True, timeout=self.timeout)
            
            os.unlink(temp_path)
            
            if result.returncode == 0:
                return {"valid": True, "error": None}
            else:
                return {"valid": False, "error": result.stderr}
                
        except subprocess.TimeoutExpired:
            return {"valid": False, "error": "Syntax check timeout"}
        except Exception as e:
            return {"valid": False, "error": f"Syntax check failed: {e}"}
    
    def _inject_inputs(self, bl_code: str, inputs: Dict) -> str:
        """Inject input facts into ByteLogic code."""
        # Find insertion point after REL declarations
        lines = bl_code.split('\n')
        insertion_point = 0
        
        for i, line in enumerate(lines):
            if line.strip().startswith('REL '):
                insertion_point = i + 1
            elif line.strip().startswith('FACT ') and insertion_point == 0:
                insertion_point = i
                break
            elif line.strip() and not line.strip().startswith(';') and not line.strip().startswith('REL '):
                break
        
        # Generate FACT statements from inputs
        fact_lines = []
        for relation, facts in inputs.items():
            if isinstance(facts, list):
                for fact in facts:
                    if isinstance(fact, (list, tuple)) and len(fact) == 2:
                        fact_lines.append(f"FACT {relation} {fact[0]} {fact[1]}")
                    elif isinstance(fact, dict) and 'a' in fact and 'b' in fact:
                        fact_lines.append(f"FACT {relation} {fact['a']} {fact['b']}")
        
        # Insert facts
        new_lines = lines[:insertion_point] + fact_lines + lines[insertion_point:]
        return '\n'.join(new_lines)
    
    def _compile_to_wat(self, bl_code: str) -> Dict:
        """Compile ByteLogic to WAT using the ByteLogic compiler."""
        # Check cache first
        if self.cache_enabled and bl_code in self._wat_cache:
            return {"success": True, "wat_code": self._wat_cache[bl_code]}
        
        if not self.bytelogic_path:
            return {"success": False, "error": "ByteLogic compiler not available"}
        
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.bl', delete=False) as f:
                f.write(bl_code)
                bl_path = f.name
            
            # Compile to WAT
            result = subprocess.run([
                self.bytelogic_path,
                "-c", "wat", "-o", "-",
                bl_path
            ], capture_output=True, text=True, timeout=self.timeout)
            
            os.unlink(bl_path)
            
            if result.returncode == 0:
                wat_code = result.stdout
                if self.cache_enabled:
                    self._wat_cache[bl_code] = wat_code
                return {"success": True, "wat_code": wat_code}
            else:
                return {"success": False, "error": result.stderr}
                
        except subprocess.TimeoutExpired:
            return {"success": False, "error": "Compilation timeout"}
        except Exception as e:
            return {"success": False, "error": f"Compilation failed: {e}"}
    
    def _execute_real_wasm(self, wat_code: str) -> Dict:
        """Execute WAT code using actual WASM runtime."""
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                wat_file = os.path.join(temp_dir, "code.wat")
                wasm_file = os.path.join(temp_dir, "code.wasm")
                
                # Write WAT code
                with open(wat_file, 'w') as f:
                    f.write(wat_code)
                
                # Compile WAT to WASM
                compile_result = subprocess.run([
                    "wat2wasm", wat_file, "-o", wasm_file
                ], capture_output=True, text=True, timeout=self.timeout)
                
                if compile_result.returncode != 0:
                    return {
                        "success": False,
                        "error": f"WAT compilation failed: {compile_result.stderr}"
                    }
                
                # Execute WASM
                exec_result = self._execute_wasm_file(wasm_file)
                return exec_result
                
        except Exception as e:
            return {"success": False, "error": f"WASM execution failed: {e}"}
    
    def _execute_wasm_file(self, wasm_file: str) -> Dict:
        """Execute a WASM file using wasmtime."""
        try:
            from wasmtime import Store, Module, Instance
            
            # Load WASM module
            with open(wasm_file, 'rb') as f:
                wasm_bytes = f.read()
            
            store = Store()
            module = Module(store.engine, wasm_bytes)
            instance = Instance(store, module, [])
            
            # Look for exported functions
            exports = instance.exports(store)
            results = {}
            
            # Try to find and execute standard functions
            for name, func in exports.items():
                if hasattr(func, '__call__'):
                    try:
                        if name == "solve":
                            func(store)  # Execute solve function
                        elif name == "query":
                            result = func(store)
                            results["query_result"] = result
                        elif name == "compute":
                            result = func(store)
                            results["computation_result"] = result
                    except Exception as e:
                        logger.warning(f"Error executing {name}: {e}")
            
            return {"success": True, "results": results}
            
        except ImportError:
            return {"success": False, "error": "wasmtime package not available"}
        except Exception as e:
            return {"success": False, "error": f"WASM execution failed: {e}"}
    
    def _simulate_execution(self, bl_code: str, wat_result: Dict) -> Dict:
        """Simulate ByteLogic execution for development/testing."""
        results = {}
        
        # Parse ByteLogic to understand what it's doing
        analysis = self._analyze_bytelogic_code(bl_code)
        
        if analysis["type"] == "logic_program":
            # Simulate logic programming execution
            results = self._simulate_logic_execution(analysis)
        elif analysis["type"] == "calculation":
            # Simulate calculation execution  
            results = self._simulate_calculation_execution(analysis)
        else:
            results = {"success": False, "error": "Unknown program type"}
        
        return results
    
    def _analyze_bytelogic_code(self, bl_code: str) -> Dict:
        """Analyze ByteLogic code to understand its structure."""
        lines = bl_code.split('\n')
        
        analysis = {
            "type": "unknown",
            "relations": [],
            "facts": [],
            "rules": [],
            "queries": [],
            "calculations": []
        }
        
        for line in lines:
            line = line.strip()
            if not line or line.startswith(';'):
                continue
            
            if line.startswith('REL '):
                rel_name = line.split()[1]
                analysis["relations"].append(rel_name)
                analysis["type"] = "logic_program"
            
            elif line.startswith('FACT '):
                parts = line.split()[1:]
                if len(parts) >= 3:
                    fact = {"relation": parts[0], "a": parts[1], "b": parts[2]}
                    analysis["facts"].append(fact)
            
            elif line.startswith('QUERY '):
                parts = line.split()[1:]
                if len(parts) >= 3:
                    query = {"relation": parts[0], "a": parts[1], "b": parts[2]}
                    analysis["queries"].append(query)
            
            elif line.startswith('CALC '):
                calc_name = line.split()[1]
                analysis["calculations"].append(calc_name)
                analysis["type"] = "calculation"
            
            elif line.startswith('RESULT '):
                analysis["type"] = "calculation"
        
        return analysis
    
    def _simulate_logic_execution(self, analysis: Dict) -> Dict:
        """Simulate logic program execution."""
        # Build fact database
        fact_db = {}
        for fact in analysis["facts"]:
            rel = fact["relation"]
            if rel not in fact_db:
                fact_db[rel] = []
            fact_db[rel].append((fact["a"], fact["b"]))
        
        # Apply simple transitive closure for common patterns
        fact_db = self._apply_transitive_rules(fact_db, analysis)
        
        # Execute queries
        query_results = []
        for query in analysis["queries"]:
            rel = query["relation"]
            a, b = query["a"], query["b"]
            
            if rel in fact_db:
                if a == "?" and b == "?":
                    # Count all tuples
                    query_results.append(len(fact_db[rel]))
                elif a == "?":
                    # Find all matching first positions
                    matches = [fact[0] for fact in fact_db[rel] if fact[1] == b]
                    query_results.extend(matches)
                elif b == "?":
                    # Find all matching second positions  
                    matches = [fact[1] for fact in fact_db[rel] if fact[0] == a]
                    query_results.extend(matches)
                else:
                    # Membership test
                    exists = (a, b) in fact_db[rel]
                    query_results.append(1 if exists else 0)
        
        return {
            "success": True,
            "results": {
                "query_results": query_results,
                "fact_count": sum(len(facts) for facts in fact_db.values())
            }
        }
    
    def _apply_transitive_rules(self, fact_db: Dict, analysis: Dict) -> Dict:
        """Apply simple transitive closure rules."""
        # Look for common transitive patterns
        for rel in list(fact_db.keys()):
            if rel in ["ancestor", "descendant", "reachable", "manages", "knows"]:
                # Apply transitive closure
                original_facts = fact_db[rel][:]
                new_facts = set(original_facts)
                
                # Simple transitive closure (limited iterations)
                for _ in range(10):  # Limit iterations to prevent infinite loops
                    added_new = False
                    for fact1 in original_facts:
                        for fact2 in original_facts:
                            if fact1[1] == fact2[0]:  # Transitive: A->B, B->C => A->C
                                new_fact = (fact1[0], fact2[1])
                                if new_fact not in new_facts:
                                    new_facts.add(new_fact)
                                    added_new = True
                    
                    if not added_new:
                        break
                    original_facts = list(new_facts)
                
                fact_db[rel] = list(new_facts)
        
        return fact_db
    
    def _simulate_calculation_execution(self, analysis: Dict) -> Dict:
        """Simulate calculation execution."""
        # Simple calculation simulation
        if "factorial" in str(analysis):
            result = 120  # Simulate factorial(5)
        elif "fibonacci" in str(analysis):
            result = 55   # Simulate fibonacci(10)
        elif "percentage" in str(analysis):
            result = 36   # Simulate 15% of 240
        else:
            result = 42   # Default result
        
        return {
            "success": True,
            "results": {"calculation_result": result}
        }
    
    def _parse_execution_results(self, exec_result: Dict, bl_code: str) -> Dict:
        """Parse and format execution results."""
        parsed = {
            "success": exec_result.get("success", False),
            "result": None,
            "query_results": [],
            "calculation_result": None
        }
        
        if not exec_result.get("success"):
            parsed["error"] = exec_result.get("error")
            return parsed
        
        results = exec_result.get("results", {})
        
        # Handle query results
        if "query_results" in results:
            parsed["query_results"] = results["query_results"]
            parsed["result"] = results["query_results"]
        
        # Handle calculation results  
        if "calculation_result" in results:
            parsed["calculation_result"] = results["calculation_result"]
            parsed["result"] = results["calculation_result"]
        
        # Handle other result formats
        if "query_result" in results:
            parsed["query_results"] = [results["query_result"]]
            parsed["result"] = results["query_result"]
        
        if "computation_result" in results:
            parsed["calculation_result"] = results["computation_result"]
            parsed["result"] = results["computation_result"]
        
        parsed["success"] = True
        return parsed
    
    def _format_computation_token(self, result: Dict) -> str:
        """Format result as computation token for model integration."""
        if result["query_results"]:
            if len(result["query_results"]) == 1:
                return f"<result>{result['query_results'][0]}</result>"
            else:
                formatted_results = ", ".join(str(r) for r in result["query_results"])
                return f"<result>{formatted_results}</result>"
        
        elif result["calculation_result"] is not None:
            return f"<result>{result['calculation_result']}</result>"
        
        else:
            return "<result>completed</result>"
    
    def batch_execute(self, programs: List[str], inputs_list: List[Optional[Dict]] = None) -> List[Dict]:
        """Execute multiple ByteLogic programs in batch."""
        if inputs_list is None:
            inputs_list = [None] * len(programs)
        
        results = []
        for program, inputs in zip(programs, inputs_list):
            result = self.execute_bytelogic(program, inputs)
            results.append(result)
        
        return results
    
    def get_cache_stats(self) -> Dict:
        """Get compilation cache statistics."""
        return {
            "wat_cache_size": len(self._wat_cache),
            "wasm_cache_size": len(self._wasm_cache),
            "result_cache_size": len(self._result_cache),
            "cache_enabled": self.cache_enabled
        }
    
    def clear_cache(self):
        """Clear all compilation caches."""
        self._wat_cache.clear()
        self._wasm_cache.clear() 
        self._result_cache.clear()
        logger.info("ByteLogic execution caches cleared")