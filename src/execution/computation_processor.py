"""
Computation Token Processor
===========================

Handles processing of <computation> tokens in LLM outputs by executing
ByteLogic code and replacing tokens with results. Integrates with both
ByteLogic and legacy WAT execution paths.
"""

import re
import logging
from typing import Dict, List, Optional, Tuple, Any
from .bytelogic_executor import ByteLogicExecutor
from .wasm_executor import WASMExecutor
from ..tokenization.bytelogic_tokenizer import ByteLogicTokenizer

logger = logging.getLogger(__name__)


class ComputationTokenProcessor:
    """Processes computation tokens in LLM-generated text."""
    
    def __init__(self, 
                 bytelogic_executor: Optional[ByteLogicExecutor] = None,
                 wasm_executor: Optional[WASMExecutor] = None,
                 tokenizer: Optional[ByteLogicTokenizer] = None,
                 enable_legacy_support: bool = True):
        """
        Initialize computation token processor.
        
        Args:
            bytelogic_executor: ByteLogic executor instance
            wasm_executor: Legacy WASM executor for <computed> tokens
            tokenizer: ByteLogic tokenizer instance
            enable_legacy_support: Whether to support legacy <computed> tokens
        """
        self.bytelogic_executor = bytelogic_executor or ByteLogicExecutor()
        self.wasm_executor = wasm_executor or WASMExecutor()
        self.tokenizer = tokenizer or ByteLogicTokenizer()
        self.enable_legacy_support = enable_legacy_support
        
        # Compile patterns for token detection
        self._compile_patterns()
        
        # Execution statistics
        self.stats = {
            "computation_tokens_processed": 0,
            "computed_tokens_processed": 0,
            "successful_executions": 0,
            "failed_executions": 0,
            "cache_hits": 0
        }
        
        logger.info("ComputationTokenProcessor initialized")
        logger.info(f"  ByteLogic support: {'Enabled' if self.bytelogic_executor else 'Disabled'}")
        logger.info(f"  Legacy WAT support: {'Enabled' if self.enable_legacy_support else 'Disabled'}")
    
    def _compile_patterns(self):
        """Compile regex patterns for token detection."""
        # New computation tokens
        self.computation_pattern = re.compile(
            r'<computation>\s*(.*?)\s*</computation>',
            re.DOTALL | re.IGNORECASE
        )
        
        # Legacy computed tokens (for backward compatibility)
        self.computed_pattern = re.compile(
            r'<computed>(.*?)</computed>',
            re.DOTALL | re.IGNORECASE
        )
        
        # Error tokens
        self.error_pattern = re.compile(
            r'<error>(.*?)</error>',
            re.DOTALL | re.IGNORECASE
        )
    
    def process_text(self, text: str, context: Optional[Dict] = None) -> str:
        """
        Process all computation tokens in text.
        
        Args:
            text: Text containing potential computation tokens
            context: Optional context for execution (variables, facts, etc.)
            
        Returns:
            Text with computation tokens replaced by results
        """
        # Process new <computation> tokens
        text = self._process_computation_tokens(text, context)
        
        # Process legacy <computed> tokens if enabled
        if self.enable_legacy_support:
            text = self._process_computed_tokens(text)
        
        return text
    
    def _process_computation_tokens(self, text: str, context: Optional[Dict] = None) -> str:
        """Process <computation> tokens containing ByteLogic code."""
        def replace_computation(match):
            bytelog_code = match.group(1).strip()
            self.stats["computation_tokens_processed"] += 1
            
            try:
                # Execute ByteLogic code
                result = self.bytelogic_executor.execute_bytelogic(
                    bytelog_code, 
                    inputs=context
                )
                
                if result["success"]:
                    self.stats["successful_executions"] += 1
                    return self._format_result(result)
                else:
                    self.stats["failed_executions"] += 1
                    error_msg = result.get("error", "unknown_error")
                    logger.warning(f"ByteLogic execution failed: {error_msg}")
                    return f"<error>{error_msg}</error>"
                    
            except Exception as e:
                self.stats["failed_executions"] += 1
                logger.error(f"Computation token processing failed: {e}")
                return f"<error>processing_failed: {str(e)}</error>"
        
        return self.computation_pattern.sub(replace_computation, text)
    
    def _process_computed_tokens(self, text: str) -> str:
        """Process legacy <computed> tokens for backward compatibility."""
        def replace_computed(match):
            wat_code = match.group(1).strip()
            self.stats["computed_tokens_processed"] += 1
            
            try:
                # Execute WAT code using legacy executor
                result = self.wasm_executor.execute_wat(wat_code)
                
                if result["success"]:
                    self.stats["successful_executions"] += 1
                    return f"→ {result['result']}"
                else:
                    self.stats["failed_executions"] += 1
                    error_msg = result.get("error", "unknown_error")
                    logger.warning(f"Legacy WAT execution failed: {error_msg}")
                    return f"<error>{error_msg}</error>"
                    
            except Exception as e:
                self.stats["failed_executions"] += 1
                logger.error(f"Legacy computed token processing failed: {e}")
                return f"<error>legacy_processing_failed: {str(e)}</error>"
        
        return self.computed_pattern.sub(replace_computed, text)
    
    def _format_result(self, result: Dict) -> str:
        """Format execution result for display in text."""
        if result.get("query_results"):
            # Logic programming results
            results = result["query_results"]
            if len(results) == 1:
                if isinstance(results[0], (int, float)) and results[0] in [0, 1]:
                    # Boolean result
                    return "→ " + ("Yes" if results[0] == 1 else "No")
                else:
                    return f"→ {results[0]}"
            else:
                formatted = ", ".join(str(r) for r in results)
                return f"→ {formatted}"
        
        elif result.get("calculation_result") is not None:
            # Mathematical calculation results
            calc_result = result["calculation_result"]
            if isinstance(calc_result, float):
                # Format floats nicely
                if calc_result.is_integer():
                    return f"→ {int(calc_result)}"
                else:
                    return f"→ {calc_result:.2f}"
            else:
                return f"→ {calc_result}"
        
        elif result.get("result") is not None:
            # Generic result
            return f"→ {result['result']}"
        
        else:
            # Execution completed without specific result
            return "→ (completed)"
    
    def extract_computation_tokens(self, text: str) -> List[Dict]:
        """
        Extract all computation tokens from text without executing.
        
        Returns:
            List of token information dicts
        """
        tokens = []
        
        # Extract computation tokens
        for match in self.computation_pattern.finditer(text):
            tokens.append({
                "type": "computation",
                "full_match": match.group(0),
                "code": match.group(1).strip(),
                "start": match.start(),
                "end": match.end()
            })
        
        # Extract legacy computed tokens if enabled
        if self.enable_legacy_support:
            for match in self.computed_pattern.finditer(text):
                tokens.append({
                    "type": "computed",
                    "full_match": match.group(0),
                    "code": match.group(1).strip(),
                    "start": match.start(),
                    "end": match.end()
                })
        
        return sorted(tokens, key=lambda x: x["start"])
    
    def validate_computation_tokens(self, text: str) -> List[Dict]:
        """
        Validate all computation tokens in text without executing.
        
        Returns:
            List of validation results
        """
        tokens = self.extract_computation_tokens(text)
        validations = []
        
        for token in tokens:
            if token["type"] == "computation":
                # Validate ByteLogic syntax
                is_valid, error = self.tokenizer.validate_bytelogic_syntax(token["code"])
                validations.append({
                    "token": token,
                    "valid": is_valid,
                    "error": error,
                    "token_count": len(self.tokenizer.tokenize_bytelogic(token["code"]))
                })
            elif token["type"] == "computed":
                # Basic WAT validation (just check for balanced parentheses)
                code = token["code"]
                paren_count = code.count('(') - code.count(')')
                validations.append({
                    "token": token,
                    "valid": paren_count == 0,
                    "error": "Unbalanced parentheses" if paren_count != 0 else None,
                    "token_count": len(code.split())
                })
        
        return validations
    
    def preview_execution(self, text: str, context: Optional[Dict] = None) -> Dict:
        """
        Preview what would happen if computation tokens were executed.
        
        Returns:
            Preview information including validation and estimated results
        """
        tokens = self.extract_computation_tokens(text)
        validations = self.validate_computation_tokens(text)
        
        preview = {
            "token_count": len(tokens),
            "valid_tokens": sum(1 for v in validations if v["valid"]),
            "invalid_tokens": sum(1 for v in validations if not v["valid"]),
            "tokens": tokens,
            "validations": validations,
            "estimated_execution_time": self._estimate_execution_time(tokens),
            "would_execute": len([t for t in tokens if t["type"] == "computation"]),
            "would_fallback_legacy": len([t for t in tokens if t["type"] == "computed"])
        }
        
        return preview
    
    def _estimate_execution_time(self, tokens: List[Dict]) -> float:
        """Estimate execution time for tokens in milliseconds."""
        total_time = 0.0
        
        for token in tokens:
            if token["type"] == "computation":
                # Estimate based on code complexity
                code_lines = len(token["code"].split('\n'))
                if "CALC" in token["code"]:
                    total_time += 50 + code_lines * 5  # Mathematical calculations
                elif "FOR" in token["code"]:
                    total_time += 100 + code_lines * 10  # Loop processing
                else:
                    total_time += 20 + code_lines * 2   # Simple logic
            
            elif token["type"] == "computed":
                total_time += 30  # Legacy WAT execution
        
        return total_time
    
    def get_statistics(self) -> Dict:
        """Get processing statistics."""
        total_processed = (self.stats["computation_tokens_processed"] + 
                          self.stats["computed_tokens_processed"])
        
        if total_processed > 0:
            success_rate = self.stats["successful_executions"] / total_processed * 100
        else:
            success_rate = 0.0
        
        return {
            **self.stats,
            "total_tokens_processed": total_processed,
            "success_rate_percent": success_rate,
            "cache_stats": self.bytelogic_executor.get_cache_stats() if self.bytelogic_executor else {}
        }
    
    def clear_cache(self):
        """Clear all execution caches."""
        if self.bytelogic_executor:
            self.bytelogic_executor.clear_cache()
        logger.info("Computation processor caches cleared")
    
    def enable_debug_mode(self, enabled: bool = True):
        """Enable debug mode for detailed execution logging."""
        if enabled:
            logging.getLogger('src.execution').setLevel(logging.DEBUG)
        else:
            logging.getLogger('src.execution').setLevel(logging.INFO)


class StreamingComputationProcessor:
    """Processes computation tokens in streaming LLM output."""
    
    def __init__(self, processor: ComputationTokenProcessor):
        self.processor = processor
        self.buffer = ""
        self.in_computation = False
        self.computation_content = ""
    
    def process_chunk(self, chunk: str) -> Tuple[str, bool]:
        """
        Process a streaming chunk.
        
        Returns:
            (processed_chunk, has_pending_computation)
        """
        self.buffer += chunk
        
        # Check for computation token start
        if "<computation>" in self.buffer and not self.in_computation:
            parts = self.buffer.split("<computation>", 1)
            if len(parts) == 2:
                self.in_computation = True
                self.computation_content = ""
                return parts[0], True  # Return text before token, flag pending
        
        # Check for computation token end
        if self.in_computation and "</computation>" in self.buffer:
            parts = self.buffer.split("</computation>", 1)
            if len(parts) == 2:
                self.computation_content += parts[0]
                
                # Execute the computation
                full_token = f"<computation>{self.computation_content}</computation>"
                result = self.processor.process_text(full_token)
                
                # Reset state
                self.in_computation = False
                self.buffer = parts[1]
                self.computation_content = ""
                
                return result + parts[1], False
        
        # If in computation, accumulate content
        if self.in_computation:
            self.computation_content += chunk
            return "", True
        
        # Normal text
        processed = self.buffer
        self.buffer = ""
        return processed, False