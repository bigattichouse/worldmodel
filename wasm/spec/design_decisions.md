# WASM WorldModel Design Decisions

## Core Architecture Choice: WASM-Only Approach

### Decision
Focus exclusively on WebAssembly (WASM) for internal computation, using tool calling for external capabilities.

### Rationale

**WASM serves a fundamentally different purpose than external code execution:**

- **WASM**: Computation *as part of reasoning* - executed during model forward pass
- **Python/Tool Calling**: External capabilities - separate from reasoning process

### Tool Execution Model

The `<model><requires>` pattern represents tool calling. WASM extends this by making some tools internal:

```
Internal Tools (WASM execution):
- Mathematical computation (17 × 23)
- Algorithmic processing (sorting, searching)  
- Logical operations (boolean algebra)
- Numerical analysis (equation solving)
→ Executed during model reasoning

External Tools (API execution):
- Date/time queries (datetime.now())
- File system operations (os.listdir())
- Platform information (platform.system())
- System integration
→ Executed via sandbox or direct host calls
```

**Key insight**: Both are tool calls, but WASM tools execute as part of reasoning while API tools execute externally.

### Advantages of WASM-Only

1. **Architectural Clarity**: Single execution model within reasoning
2. **Research Focus**: Clear evaluation of multimodal computation effectiveness  
3. **Performance**: Deterministic execution integrated with attention mechanism
4. **Complexity Management**: One execution path to optimize and debug

### Future Extensions

- Tool calling can be added later for external capabilities
- WASM can be extended with API bindings for specific library access
- Clear separation maintains architectural integrity

### Implementation Status

- ✅ WASM modal architecture implemented
- ✅ Cross-modal attention (text ↔ WASM)
- ✅ Internal execution during forward pass
- ✅ Training pipeline ready
- 🔄 Tool calling integration deferred

This approach enables novel research into computation-as-reasoning while maintaining clear architectural boundaries.