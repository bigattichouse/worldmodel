# WorldModel WASM Modal Architecture

Experimental implementation of multimodal transformer with native WebAssembly processing capabilities.

## Project Structure

```
wasm/
├── spec/               # Architecture specifications and design documents
├── src/                # Core implementation  
├── data/               # Training data and converters
├── models/             # Model implementations and checkpoints
├── tests/              # Unit and integration tests
├── tools/              # Development and debugging tools
└── README.md           # This file
```

## Overview

This project implements a Flamingo-style cross-modal transformer that processes both natural language text and WebAssembly (WASM) code as native modalities. The model can:

1. Generate WASM code during its reasoning process
2. Execute WASM programs as part of attention computation  
3. Use computational results to inform text generation
4. Maintain deterministic world models as executable code

## Key Features

- **Cross-Modal Architecture**: Separate text and WASM processing streams with periodic fusion
- **Executable Internal State**: WASM programs execute during model forward pass
- **Computational Provenance**: `<computed>` tokens mark precise vs. hallucinated results
- **Curriculum Training**: Progressive complexity from arithmetic to complex simulations

## Getting Started

See `spec/architecture.md` for detailed design documentation.

**Note**: This is experimental research code. The main WorldModel system remains in the parent directory and continues to work independently.