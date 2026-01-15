# WorldModel WASM Modal Architecture

## Vision

Create a language model that processes WebAssembly (WASM) as an internal representation alongside natural language. Instead of generating textual code that gets compiled externally, the model maintains WASM bytecode as part of its internal state and reasoning process.

The model builds computational representations (mathematics, logic, algorithms) as executable WASM code within its attention mechanism. When reasoning about quantitative problems, the model constructs and executes precise computations rather than relying solely on pattern matching.

## Current State vs. Proposed Architecture

### Current WorldModel Architecture
```
Text Input → LLM → Text Code → External Compiler → WASM → Execution → Results
```

### Proposed WASM Modal Architecture  
```
Text Input → LLM with WASM Modal → Internal WASM Representation → Direct Execution → Results
```

## Core Architecture: Hybrid WASM-Attention Integration

### Concept: World Models as Executable Code
The model builds deterministic computational models of the world (physics, mathematics, logic) as WASM bytecode embedded directly in its attention mechanism. When asked about physics, the model doesn't just describe - it constructs and executes precise simulations.

### Hybrid Architecture (A + B)
- **Dedicated WASM embedding dimensions**: Direct bytecode representation in latent space
- **WASM attention heads**: Specialized attention that operates on executable code
- **Integrated computational graph**: WASM operations become part of the transformer's forward pass

```
Text Input → Transformer → {
    Text Tokens (traditional)
    + WASM Embeddings (executable world models)  
    + Cross-modal Attention (text ↔ code)
} → Output with Executable Models
```

## Design Questions

### Training Data Structure: Pre-Compiled WASM Approach

Building on the current `<think><model>` structure, but with WASM compilation integrated into the training pipeline:

```
Training Example (before compilation):
User: Calculate projectile trajectory for 45° launch at 100 m/s
Assistant: <think>Need to solve kinematic equations with gravity</think>
<model>
def trajectory(v0, angle, g=9.81):
    t_flight = 2 * v0 * sin(angle) / g
    max_height = (v0 * sin(angle))**2 / (2 * g)
    return t_flight, max_height
</model>

Training Example (after WASM compilation):
User: Calculate projectile trajectory for 45° launch at 100 m/s  
Assistant: <think>Need to solve kinematic equations with gravity</think>
<wasm_model>
[WASM bytecode representing the trajectory function]
</wasm_model>
```

**Advantages:**
- Model learns to generate WASM directly, not high-level code
- Deterministic execution baked into the model's output
- Can accept training data in any language (Python, C, Rust) → WASM

### WASM-Attention Head Architecture: Executable Internal State

The key insight: WASM/WAT becomes part of the model's computational process, not just its output.

```
Traditional Attention:
Q, K, V → Attention Weights → Context Vector

WASM-Attention Head:
Q, K, V → Attention Weights → Context Vector + WAT Program → Execute WASM → Updated Internal State
```

**Deep Coupling Approach:**
- **WAT as tokenized input**: Human-readable, debuggable WebAssembly Text format
- **Execution during forward pass**: WASM programs execute as attention heads compute
- **Hybrid embeddings (D)**: Each WAT function becomes a high-dimensional embedding that can be:
  - Composed with other functions
  - Modified through gradient updates  
  - Executed to update the model's internal world state

**Architecture Questions:**
- Should WASM execution happen in parallel with traditional attention heads?
- How do we backpropagate through WASM execution results?
- What happens when the model generates invalid WASM during training?

### WASM Execution Timing: Math Problem Walkthrough

**Problem:** "What's 17 × 23?"

**Traditional LLM Process:**
```
Input: "What's 17 × 23?"
→ Token Embedding → Attention Layers → Pattern Matching → Output: "391" (maybe)
```

**WASM-Enhanced LLM Process:**
```
Layer 1-2: "What's 17 × 23?"
→ Early attention recognizes multiplication pattern
→ Activates multiplication reasoning

Layer 3-4: WASM-Attention Head
→ Constructs: (module (func $mult (param i32 i32) (result i32) 
                local.get 0 local.get 1 i32.mul))
→ Executes: mult(17, 23) = 391
→ Result becomes part of internal state: "EXACT: 391"

Layer 5-6: Late attention layers
→ Access both text tokens AND computational result  
→ High confidence in numerical answer
→ Generate response using proven result
```

**Key Insight:** WASM execution happens in **middle layers**, where the model:
1. Has understood the problem type (early layers)
2. Can construct appropriate computational models
3. Uses exact results to inform final reasoning (late layers)

**This suggests Option A+: Execute WASM after attention, feed results back into subsequent layers**

### Ideal Architecture: Hybrid Result Tokens + Parallel Computation Stream

**Best of both worlds: A + C combined**

```
Main Text Stream:     "What's" "17" "×" "23" "?"
                              ↓
Computation Stream:   [WASM execution] → <computed>391</computed>
                              ↓
Integrated Stream:    "What's" "17" "×" "23" "?" <computed>391</computed> "The answer is 391"
```

**Why this hybrid is ideal:**

1. **Clear Provenance**: `<computed>391</computed>` tokens explicitly mark precise vs. hallucinated results
2. **Scalable**: Parallel stream handles complex multi-step computations without bloating main sequence
3. **Natural Attention**: Model learns to attend between linguistic reasoning and computational facts
4. **Debugging**: Easy to trace which results came from execution vs. generation
5. **Training Efficiency**: Can mask/unmask computed tokens during training to teach reliance on precision

**Architecture Flow:**
```
Layer N: Text attention generates WASM program
Layer N: WASM execution → results stored in computation stream  
Layer N+1: Computation stream results injected as special tokens
Layer N+2+: Model attends to both text and <computed> tokens
```

**This creates a "computational scratchpad" that the model writes to and reads from during reasoning.**

### Error Handling: Learning from Audio/Video Generation

**Audio/Video Model Approach:**
- **Invalid outputs are still "outputs"** - a distorted audio waveform or corrupted video frame
- **Gradients flow through everything** - even bad generations contribute to learning
- **Quality emerges gradually** - models start with noise, slowly learn structure
- **No "error states"** - every tensor is valid in the latent space

**Applied to WASM Generation:**

**Option E: Differentiable WASM Approximation**
```
Generated WAT → Soft Execution → Approximate Results → <computed>~391.2</computed>
```

Instead of binary compile/execute, use:
1. **Soft instruction mapping**: Invalid WASM opcodes map to learned approximations
2. **Differentiable operations**: Replace discrete WASM ops with differentiable alternatives during training
3. **Gradual hardening**: Start soft, gradually enforce strict WASM validity
4. **Continuous results**: Every "execution" produces a result tensor, even if nonsensical

### Proper Training Approach: No Approximations

You're absolutely right - let's do real training, not hacks.

**Clean Training Strategy:**
```
Training Data: Valid WAT programs with expected outputs
Model generates: WAT tokens during forward pass  
Execution: Real WASM compilation and execution
Loss: Based on execution results matching training targets
```

**Handling Invalid WASM During Training:**
- **Simple reality**: Invalid WASM = no execution = no computational result  
- **Training signal**: Model learns that invalid WASM provides no benefit
- **Natural selection**: Valid WASM generations get reinforced, invalid ones don't
- **No special handling needed** - let the model learn what works

**This is cleaner and more honest:**
1. Generate real WAT code
2. Compile and execute it (or fail silently)  
3. Use results if successful, ignore if failed
4. Train on the final output quality

The model will naturally learn to generate valid WASM because only valid WASM contributes to good performance. Just like learning any other structured output format.

### Curriculum Training: Converting Existing Examples

**Excellent insight!** Our existing 1,393 WorldModel examples are perfect for curriculum training.

**Curriculum Progression:**

**Stage 1: Basic Arithmetic (200 examples)**
```
Current: User: Calculate 25% of 80
         Assistant: <think>Need to calculate 25% of 80</think>
         <model>result = 0.25 * 80; print(result)</model>

WASM:    User: Calculate 25% of 80  
         Assistant: <think>Need to calculate 25% of 80</think>
         <wat_model>
         (module
           (func $percent (param f64 f64) (result f64)
             local.get 0 local.get 1 f64.mul))
         </wat_model>
         <computed>20.0</computed>
         25% of 80 equals 20.
```

**Stage 2: System Tasks (300 examples)**
Convert datetime, string processing, file operations to WAT equivalents

**Stage 3: Complex Logic (893 examples)**  
Physics simulations, data analysis, multi-step computations

**Key Advantage:** We already have the problem-solution pairs! Just need to:
1. Convert Python/code blocks to equivalent WAT
2. Add `<computed>` result tokens
3. Maintain the same reasoning quality

### SOTA-Based Architecture Design

**Most Mathematically Rigorous Approach: Flamingo-style Cross-Modal Architecture**

Based on DeepMind's Flamingo and similar SOTA multimodal models:

```
Text Transformer Layers:     L1 → L2 → L3 → ... → L12
                                ↕     ↕     ↕
WASM Processing Layers:      W1 → W2 → W3 → ... → W12
                                ↓     ↓     ↓
Cross-Modal Attention:       [Text ⟷ WASM] every N layers
```

**Key SOTA Principles:**
1. **Separate modality encoders**: Text and WASM have dedicated processing streams
2. **Periodic fusion**: Cross-attention every few layers (not every layer - computationally expensive)
3. **Modality-specific embeddings**: WAT tokens get specialized positional/semantic embeddings
4. **Late fusion for output**: Final layers combine both streams for generation

**This mirrors vision-language models** where image and text streams process separately then fuse strategically.

### Proof of Concept: Auto-Conversion Pipeline

**Perfect for validation!** Auto-conversion lets us:
1. Rapidly generate training data from existing 1,393 examples
2. Test the core architecture without manual WAT crafting
3. Validate that models can learn WASM generation patterns
4. Scale to better examples once concept is proven

**Simple Python→WAT converter for arithmetic operations, then expand.**

## Detailed Architecture Specification

### Cross-Modal Fusion Strategy: Flamingo-Style (SOTA)

**Fusion Frequency:** Every 3-4 layers (following DeepMind Flamingo methodology)

```
Layer 1-3:   [Text Processing]     [WASM Processing]
Layer 4:     [Cross-Modal Attention: Text ⟷ WASM]
Layer 5-7:   [Text Processing]     [WASM Processing]  
Layer 8:     [Cross-Modal Attention: Text ⟷ WASM]
Layer 9-12:  [Text Processing]     [WASM Processing]
Output:      [Unified Generation with <computed> tokens]
```

**Advantages of 3-4 Layer Intervals:**
- Computationally efficient (proven at scale)
- Allows modality-specific processing between fusions
- Sufficient interaction for cross-modal learning
- Balances specialization with integration

### Implementation Components

#### 1. WASM Stream Architecture
- **WAT Tokenizer**: Custom tokenizer for WebAssembly Text format
- **WASM Embeddings**: Specialized embeddings for opcodes, functions, parameters
- **WASM Positional Encoding**: Code structure awareness (function scope, control flow)
- **WASM Attention**: Self-attention within WASM code sequences

#### 2. Cross-Modal Attention Mechanism  
```python
class CrossModalAttention(nn.Module):
    def forward(self, text_hidden, wasm_hidden):
        # Text queries attend to WASM keys/values
        text_to_wasm = attention(text_hidden, wasm_hidden, wasm_hidden)
        # WASM queries attend to text keys/values  
        wasm_to_text = attention(wasm_hidden, text_hidden, text_hidden)
        return text_to_wasm, wasm_to_text
```

#### 3. Execution Integration
- **Compilation Layer**: WAT → WASM bytecode compilation
- **Execution Engine**: WASM runtime integrated into forward pass
- **Result Injection**: `<computed>` tokens inserted into text stream

### Training Pipeline

#### Phase 1: Curriculum Learning
1. **Stage 1**: Basic arithmetic (200 examples) - simple WAT functions
2. **Stage 2**: System operations (300 examples) - more complex WAT
3. **Stage 3**: Complex simulations (893 examples) - full WASM programs

#### Phase 2: Auto-Conversion System
- Parse existing `<model>` Python code blocks
- Convert to equivalent WAT using automated transpiler
- Generate `<computed>` result tokens
- Validate execution matches expected outputs

#### Phase 3: End-to-End Training
- Train on text + WAT sequences
- Loss function combines text generation + execution accuracy
- Model learns to generate valid WAT that produces correct results

## Next Steps

1. Implement WAT tokenizer and WASM embeddings
2. Build auto-conversion pipeline for existing training data
3. Create cross-modal transformer architecture
4. Integrate WASM execution engine
5. Train and evaluate proof of concept