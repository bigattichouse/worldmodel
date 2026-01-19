# Computation Token Specification

**Version:** 2.0  
**Date:** 2026-01-19  
**Purpose:** Define the new `<computation>` token format for ByteLogic-based reasoning

---

## Overview

This specification defines the transition from WAT-based `<computed>` tokens to ByteLogic-based `<computation>` tokens in the worldmodel LLM. The new format enables structured logical reasoning while maintaining deterministic execution within WASM sandboxes.

### Token Evolution

| Version | Token Format | Content | Purpose |
|---------|--------------|---------|---------|
| 1.0 | `<computed>result</computed>` | Numeric result only | Basic arithmetic |
| 2.0 | `<computation>bytelog_code</computation>` | ByteLogic program | Logical reasoning + math |

---

## Token Format Specification

### 2.1 Basic Structure

```
<computation>
[ByteLogic Program]
</computation>
```

**Components:**
- **Opening tag:** `<computation>` (exactly, no attributes)
- **Content:** Valid ByteLogic program code
- **Closing tag:** `</computation>` (exactly)
- **Whitespace:** Preserved within content, normalized around tags

### 2.2 Content Requirements

**Valid ByteLogic Program:**
- Must contain at least one statement (REL, FACT, RULE, SOLVE, QUERY, or CALC)
- Must be syntactically correct according to ByteLogic grammar
- Should terminate with SOLVE and QUERY for datalog programs
- Should terminate with RESULT for calculation programs

**Example - Logic Program:**
```
<computation>
REL parent
REL grandparent
FACT parent alice bob
FACT parent bob charlie
RULE grandparent: SCAN parent MATCH $0, JOIN parent $0, EMIT grandparent $1 $2
SOLVE
QUERY grandparent alice ?
</computation>
```

**Example - Calculation Program:**
```
<computation>
CALC factorial
  INPUT $0
  IF $0 <= 1 THEN
    RESULT 1
  ELSE
    LET $1 = CALC factorial($0 - 1)
    RESULT $0 * $1
  END
END
RESULT CALC factorial(5)
</computation>
```

### 2.3 Result Integration

**Execution Pipeline:**
1. LLM generates text containing `<computation>` token
2. ByteLogicExecutor extracts and compiles ByteLogic code
3. Compiled WASM executes in sandbox
4. Results are parsed and formatted
5. Token is replaced with human-readable result

**Result Format:**
```
Original: "Who are Alice's grandchildren? <computation>...</computation>"
After execution: "Who are Alice's grandchildren? → charlie, david"
```

---

## 3. Content Categories

### 3.1 Logic Programming

**Family Relationships:**
```
<computation>
REL parent
REL sibling
FACT parent alice bob
FACT parent alice charlie
RULE sibling: SCAN parent MATCH $0, JOIN parent $0, EMIT sibling $1 $2
SOLVE
QUERY sibling bob ?
</computation>
```

**Graph Traversal:**
```
<computation>
REL edge
REL reachable
FACT edge a b
FACT edge b c
FACT edge c d
RULE reachable: SCAN edge, EMIT reachable $0 $1
RULE reachable: SCAN edge, JOIN reachable $1, EMIT reachable $0 $2
SOLVE
QUERY reachable a d
</computation>
```

**Classification:**
```
<computation>
REL isa
REL has_property
FACT isa tweety bird
FACT isa bird animal
FACT has_property bird flies
RULE isa: SCAN isa, JOIN isa $1, EMIT isa $0 $2
RULE has_property: SCAN isa, JOIN has_property $1, EMIT has_property $0 $2
SOLVE
QUERY has_property tweety ?
</computation>
```

### 3.2 Mathematical Computation

**Arithmetic Functions:**
```
<computation>
CALC compound_interest
  INPUT $principal $rate $years
  LET $amount = $principal * POW(1 + $rate, $years)
  RESULT $amount
END
RESULT CALC compound_interest(1000, 0.05, 3)
</computation>
```

**Iterative Calculations:**
```
<computation>
CALC fibonacci
  INPUT $n
  LET $a = 0
  LET $b = 1
  FOR $i IN RANGE(0, $n)
    LET $temp = $a + $b
    LET $a = $b
    LET $b = $temp
  END
  RESULT $a
END
RESULT CALC fibonacci(10)
</computation>
```

### 3.3 Hybrid Reasoning

**Logic + Math Integration:**
```
<computation>
REL employee
REL salary
REL department
REL avg_salary

FACT employee alice engineering
FACT employee bob marketing
FACT salary alice 95000
FACT salary bob 78000

CALC department_average
  INPUT $dept
  LET $total = 0
  LET $count = 0
  FOR emp IN (QUERY employee ? $dept)
    FOR sal IN (QUERY salary emp.a ?)
      LET $total = $total + sal.b
      LET $count = $count + 1
    END
  END
  IF $count > 0 THEN
    RESULT $total / $count
  ELSE
    RESULT 0
  END
END

LET $eng_avg = CALC department_average("engineering")
FACT avg_salary engineering $eng_avg
SOLVE
QUERY avg_salary engineering ?
</computation>
```

---

## 4. Parsing Specification

### 4.1 Token Detection

**Regex Pattern:**
```python
import re

COMPUTATION_PATTERN = re.compile(
    r'<computation>\s*(.*?)\s*</computation>',
    re.DOTALL | re.IGNORECASE
)
```

**Extraction Logic:**
```python
def extract_computation_tokens(text: str) -> List[Tuple[str, str]]:
    """
    Extract computation tokens from text.
    
    Returns:
        List of (full_match, bytelog_code) tuples
    """
    matches = []
    for match in COMPUTATION_PATTERN.finditer(text):
        full_match = match.group(0)
        bytelog_code = match.group(1).strip()
        matches.append((full_match, bytelog_code))
    return matches
```

### 4.2 Content Validation

**Syntax Checking:**
```python
def validate_bytelog_syntax(code: str) -> Tuple[bool, Optional[str]]:
    """
    Validate ByteLogic syntax.
    
    Returns:
        (is_valid, error_message)
    """
    try:
        # Use ByteLogic parser to check syntax
        result = subprocess.run([
            './bytelogic/build/bytelogic', 
            '--syntax-check', 
            '-'
        ], input=code, text=True, capture_output=True)
        
        if result.returncode == 0:
            return True, None
        else:
            return False, result.stderr
    except Exception as e:
        return False, str(e)
```

### 4.3 Error Handling

**Invalid Syntax:**
```
Input: "<computation>INVALID SYNTAX</computation>"
Behavior: Replace with "<error>syntax_error</error>"
```

**Compilation Failure:**
```
Input: "<computation>REL undefined\nFACT unknown x y\nSOLVE</computation>"
Behavior: Replace with "<error>compilation_failed</error>"
```

**Execution Timeout:**
```
Input: "<computation>[infinite loop program]</computation>"
Behavior: Replace with "<error>execution_timeout</error>"
```

---

## 5. Implementation Guidelines

### 5.1 Token Processing Pipeline

```python
class ComputationTokenProcessor:
    def __init__(self, bytelogic_executor: ByteLogicExecutor):
        self.executor = bytelogic_executor
        
    def process_text(self, text: str) -> str:
        """Process all computation tokens in text."""
        def replace_computation(match):
            bytelog_code = match.group(1).strip()
            result = self.executor.execute_bytelogic(bytelog_code)
            
            if result["success"]:
                return self._format_result(result)
            else:
                return f"<error>{result['error']}</error>"
        
        return COMPUTATION_PATTERN.sub(replace_computation, text)
    
    def _format_result(self, result: Dict) -> str:
        """Format execution result for display."""
        if result["query_results"]:
            # Format query results
            items = [str(item) for item in result["query_results"]]
            return f"→ {', '.join(items)}"
        elif result["calculation_result"] is not None:
            # Format calculation result
            return f"→ {result['calculation_result']}"
        else:
            return "→ (completed)"
```

### 5.2 Model Integration

**Generation Hook:**
```python
def post_process_generation(self, text: str) -> str:
    """Post-process generated text to execute computation tokens."""
    if "<computation>" in text:
        text = self.computation_processor.process_text(text)
    return text
```

**Streaming Support:**
```python
def stream_with_computation(self, prompt: str):
    """Stream generation with real-time computation execution."""
    for chunk in self.model.stream(prompt):
        # Check if chunk completes a computation token
        if chunk.endswith("</computation>"):
            # Extract and execute the completed computation
            # Replace in stream
        yield chunk
```

### 5.3 Caching Strategy

**Compilation Caching:**
```python
class ComputationCache:
    def __init__(self, max_size: int = 1000):
        self.wat_cache = {}  # bytelog_code -> wat_code
        self.wasm_cache = {}  # wat_code -> compiled_wasm
        self.result_cache = {}  # (wasm, inputs) -> results
        
    def get_or_compile(self, bytelog_code: str) -> bytes:
        """Get compiled WASM from cache or compile."""
        if bytelog_code in self.wat_cache:
            wat_code = self.wat_cache[bytelog_code]
        else:
            wat_code = self.compile_to_wat(bytelog_code)
            self.wat_cache[bytelog_code] = wat_code
            
        if wat_code in self.wasm_cache:
            return self.wasm_cache[wat_code]
        else:
            wasm_bytes = self.compile_to_wasm(wat_code)
            self.wasm_cache[wat_code] = wasm_bytes
            return wasm_bytes
```

---

## 6. Security Considerations

### 6.1 Sandboxing

**WASM Isolation:**
- All ByteLogic programs execute in WASM sandbox
- No access to host filesystem or network
- Memory and CPU limits enforced
- Deterministic execution guaranteed

**Resource Limits:**
```python
EXECUTION_LIMITS = {
    "max_memory_mb": 64,
    "max_execution_time_ms": 1000,
    "max_program_size_kb": 100,
    "max_facts": 10000,
    "max_rules": 1000
}
```

### 6.2 Input Validation

**Content Filtering:**
```python
FORBIDDEN_PATTERNS = [
    r'import\s+',  # No imports
    r'export\s+',  # No exports  
    r'memory\s+',  # No direct memory access
    r'table\s+',   # No function tables
]

def validate_content(bytelog_code: str) -> bool:
    """Check for forbidden patterns."""
    for pattern in FORBIDDEN_PATTERNS:
        if re.search(pattern, bytelog_code, re.IGNORECASE):
            return False
    return True
```

### 6.3 Output Sanitization

**Result Validation:**
```python
def sanitize_result(result: Any) -> str:
    """Sanitize execution results for output."""
    if isinstance(result, (int, float)):
        # Numeric results are safe
        return str(result)
    elif isinstance(result, str):
        # Escape special characters
        return html.escape(result)
    elif isinstance(result, list):
        # Sanitize list elements
        return [sanitize_result(item) for item in result]
    else:
        return "<error>invalid_result_type</error>"
```

---

## 7. Performance Specification

### 7.1 Execution Targets

| Metric | Target | Maximum |
|--------|---------|---------|
| **Compilation Time** | <100ms | <500ms |
| **Execution Time** | <50ms | <200ms |
| **Memory Usage** | <16MB | <64MB |
| **Cache Hit Rate** | >80% | N/A |

### 7.2 Scalability Limits

**Program Size:**
- Maximum 100KB ByteLogic source
- Maximum 10,000 facts per program
- Maximum 1,000 rules per program
- Maximum 50 relations per program

**Query Complexity:**
- Maximum 10 inference steps
- Maximum 1,000,000 intermediate facts
- Timeout after 1 second execution

### 7.3 Optimization Strategies

**Compilation Optimization:**
- Cache compiled WAT and WASM bytecode
- Reuse compilation results for identical programs
- Parallel compilation for batch processing

**Execution Optimization:**
- WASM module pooling and reuse
- Incremental fact loading
- Query result streaming for large datasets

---

## 8. Testing Strategy

### 8.1 Unit Tests

**Token Extraction:**
```python
def test_computation_token_extraction():
    text = "Result: <computation>REL test\nSOLVE</computation> done"
    tokens = extract_computation_tokens(text)
    assert len(tokens) == 1
    assert tokens[0][1] == "REL test\nSOLVE"
```

**Syntax Validation:**
```python
def test_bytelog_syntax_validation():
    valid_code = "REL parent\nFACT parent alice bob\nSOLVE\nQUERY parent alice ?"
    is_valid, error = validate_bytelog_syntax(valid_code)
    assert is_valid
    assert error is None
```

### 8.2 Integration Tests

**End-to-End Execution:**
```python
def test_end_to_end_computation():
    text = "Who are parents? <computation>REL parent\nFACT parent alice bob\nSOLVE\nQUERY parent ? ?</computation>"
    result = processor.process_text(text)
    assert "→ alice bob" in result
```

**Error Handling:**
```python
def test_syntax_error_handling():
    text = "Bad syntax: <computation>INVALID</computation>"
    result = processor.process_text(text)
    assert "<error>syntax_error</error>" in result
```

### 8.3 Performance Tests

**Compilation Benchmarks:**
```python
def test_compilation_performance():
    code = generate_large_bytelog_program()  # 1000 facts, 100 rules
    start_time = time.time()
    result = executor.execute_bytelogic(code)
    execution_time = time.time() - start_time
    assert execution_time < 0.5  # Under 500ms
    assert result["success"]
```

**Memory Usage:**
```python
def test_memory_limits():
    code = generate_memory_intensive_program()
    with memory_profiler():
        result = executor.execute_bytelogic(code)
        peak_memory = get_peak_memory_usage()
        assert peak_memory < 64 * 1024 * 1024  # Under 64MB
```

---

## 9. Migration Timeline

### Phase 1: Infrastructure (Weeks 1-2)
- [ ] Implement ComputationTokenProcessor
- [ ] Add ByteLogic syntax validation
- [ ] Create compilation cache system
- [ ] Unit tests for token processing

### Phase 2: Integration (Weeks 3-4)  
- [ ] Integrate with model generation pipeline
- [ ] Add streaming computation support
- [ ] Implement error handling and fallbacks
- [ ] Integration tests

### Phase 3: Optimization (Weeks 5-6)
- [ ] Performance tuning and caching
- [ ] Security hardening
- [ ] Comprehensive testing
- [ ] Documentation updates

### Phase 4: Deployment (Weeks 7-8)
- [ ] Gradual rollout with A/B testing
- [ ] Monitor performance and accuracy
- [ ] Training data migration
- [ ] Full production deployment

---

## 10. Backward Compatibility

### 10.1 Legacy Token Support

**Dual Processing:**
```python
def process_all_tokens(self, text: str) -> str:
    """Process both old and new token formats."""
    # Process legacy <computed> tokens
    text = self.legacy_processor.process_computed_tokens(text)
    
    # Process new <computation> tokens
    text = self.computation_processor.process_text(text)
    
    return text
```

### 10.2 Gradual Migration

**Week 1-4:** Support both `<computed>` and `<computation>` tokens
**Week 5-6:** Prefer `<computation>` for new generation, maintain `<computed>` parsing
**Week 7-8:** Full migration to `<computation>`, deprecate `<computed>`

### 10.3 Fallback Mechanisms

**Compilation Failure:**
```python
def execute_with_fallback(self, bytelog_code: str) -> Dict:
    """Try ByteLogic execution, fallback to simulation."""
    try:
        return self.execute_bytelogic(bytelog_code)
    except CompilationError:
        return self.simulate_execution(bytelog_code)
    except ExecutionTimeout:
        return {"success": False, "error": "timeout"}
```

---

This specification provides comprehensive guidance for implementing the new `<computation>` token format while maintaining system reliability and performance.