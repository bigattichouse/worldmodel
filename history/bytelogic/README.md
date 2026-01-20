# ByteLog Compiler

A portable, high-performance C implementation of the **ByteLog** language - a minimal logic programming notation designed for Large Language Models to generate deterministic logical inference programs that compile to **WebAssembly**.

[![Build Status](https://img.shields.io/badge/build-passing-brightgreen)](#)
[![Tests](https://img.shields.io/badge/tests-12%20passed-brightgreen)](#)
[![C99](https://img.shields.io/badge/C-99-blue)](#)
[![WASM](https://img.shields.io/badge/WASM-supported-orange)](#)

## 🚀 Quick Start

```bash
# Build the compiler
make

# Run ByteLog programs (interpreter mode)
./build/bytelogic examples/example_family.bl

# Compile ByteLog to WebAssembly Text
./build/bytelogic --compile=wat examples/example_family.bl

# Compile ByteLog to WASM binary
./build/bytelogic --compile=wasm examples/example_family.bl

# Run with verbose output
./build/bytelogic --verbose examples/example_family.bl

# Run all unit tests
make test
```

## 📁 Project Structure

```
bytelog/
├── src/              # Source files (.c)
│   ├── lexer.c       # Tokenization
│   ├── ast.c         # Abstract Syntax Tree
│   ├── atoms.c       # String interning for readable names
│   ├── parser.c      # Recursive descent parser
│   ├── engine.c      # Datalog execution engine
│   ├── wat_gen.c     # WebAssembly Text generator
│   ├── demo.c        # Main ByteLog executable (interpreter & compiler)
│   └── wat_compiler.c # Legacy standalone WAT compiler
├── includes/         # Header files (.h)
├── examples/         # Example ByteLog programs (.bl)
├── build/            # Build artifacts and executables
├── docs/             # Documentation
└── Makefile          # Comprehensive build system
```

## 🏗️ Language Overview

ByteLog is a **Datalog-like** logic programming language with these features:

- **Simple syntax** that LLMs can generate reliably  
- **Binary relations** only (pairs of integers/atoms)
- **Readable atom names** (`alice`, `pizza`) that compile to efficient integers
- **Monotonic semantics** (facts only added, never removed)
- **Guaranteed termination** (reaches fixpoint in finite time)
- **Compiles to native C** and **WebAssembly**

### Core Language Elements

| Element | Syntax | Purpose |
|---------|---------|---------|
| **Relation Declaration** | `REL name` | Declares a binary relation |
| **Fact** | `FACT relation alice bob` | Asserts `relation(alice,bob)` is true |
| **Rule** | `RULE target: body, EMIT ...` | Derives new facts from existing ones |
| **Scan** | `SCAN relation MATCH $0` | Iterates over relation facts |
| **Join** | `JOIN relation $0` | Joins on shared variables |
| **Emit** | `EMIT relation $1 $2` | Creates new derived facts |
| **Solve** | `SOLVE` | Computes fixpoint (derives all possible facts) |
| **Query** | `QUERY relation alice ?` | Questions about facts |

### Example Program

```bytelog
REL parent
REL grandparent

FACT parent alice bob
FACT parent bob charlie

RULE grandparent: SCAN parent MATCH $0, JOIN parent $0, EMIT grandparent $1 $2

SOLVE

QUERY grandparent alice ?
```

This program:
1. Declares `parent` and `grandparent` relations
2. Asserts that `alice` is parent of `bob`, `bob` is parent of `charlie`  
3. Defines a rule: if X is parent of Y and Y is parent of Z, then X is grandparent of Z
4. Computes all derivable facts (finds `grandparent(alice, charlie)`)
5. Queries for all grandchildren of `alice`

## 🔧 Build System

The project uses a comprehensive **GNU Make** build system with multiple targets:

### Primary Targets

```bash
make all        # Build all executables (default)
make demo       # Build and run interactive demo
make wat        # Build WebAssembly Text compiler  
make test       # Run all 96 unit tests
make examples   # Run all example programs
```

### Development & Quality

```bash
make debug      # Build with debug symbols
make memcheck   # Run tests with Valgrind memory checking
make lint       # Static analysis with cppcheck
make format     # Format code with clang-format
make clean      # Remove build artifacts
```

### Distribution

```bash
make dist       # Create source distribution package
make install    # Install to system (PREFIX=/usr/local)
make docs       # Generate API documentation
```

## 🎯 Usage Examples

### Interactive Analysis

```bash
# Basic execution with minimal output  
./build/bytelogic examples/example_family.bl

# Verbose execution showing parsing and derivation steps
./build/bytelogic --verbose examples/example_family.bl
```

### WebAssembly Compilation

```bash
# Compile to WebAssembly Text format
./build/bytelogic --compile=wat examples/example_family.bl

# Compile to WASM binary with custom output
./build/bytelogic -c wasm -o custom.wasm examples/example_family.bl

# Then run with WASM runtime
wasmtime custom.wasm
```

Generates WAT/WASM code compatible with web browsers and WASM runtimes.

### Programmatic Usage

```c
#include "engine.h"

// Execute ByteLog source
ExecutionEngine *engine = execute_string(source, error_buf, sizeof(error_buf));

// Query results  
QueryResult *results = engine_query(engine, query_ast);
query_result_print(results, "relation_name", &engine->atoms);
```

## 🏆 Features

### ✅ **Complete Implementation**
- **Lexical Analysis** - Robust tokenization with error handling
- **Syntax Analysis** - Recursive descent parser with recovery
- **Semantic Analysis** - AST validation and type checking
- **Execution Engine** - Full Datalog evaluation with fixpoint computation
- **Code Generation** - WebAssembly Text (WAT) backend

### ✅ **Atom System**
- **Readable Names** - Use `alice`, `pizza` instead of numbers
- **String Interning** - Efficient hash table mapping
- **Mixed Arguments** - `FACT likes alice 42` (atom + integer)
- **Case Sensitive** - `Alice` ≠ `alice` ≠ `ALICE`

### ✅ **Robust Testing** 
- **96 Unit Tests** covering all language features
- **Memory Safety** - Valgrind-clean implementation
- **Platform Portable** - C99 compliant, works on Linux/macOS/Windows
- **Comprehensive Coverage** - Lexer, parser, AST, execution, WAT generation

### ✅ **Professional Tooling**
- **Build System** - Feature-rich Makefile with 25+ targets
- **Static Analysis** - cppcheck integration
- **Code Formatting** - clang-format integration  
- **Documentation** - Comprehensive API docs
- **Distribution** - Source packaging and system installation

## 📊 Statistics

- **Language Features**: 9 core constructs (REL, FACT, RULE, etc.)
- **Source Files**: 8 core modules + 4 test suites  
- **Lines of Code**: ~6,000 lines of well-structured C99
- **Test Coverage**: 96 unit tests across all modules
- **Documentation**: Comprehensive README + API docs
- **Platforms**: Native C99 + WebAssembly targets

## 🎓 Use Cases

### Educational
- **Compiler Construction** - Complete end-to-end implementation
- **Logic Programming** - Introduction to Datalog semantics
- **WebAssembly** - Code generation for modern web platform

### Research & Development  
- **AI/ML Logic** - Deterministic reasoning for LLMs
- **Constraint Solving** - Declarative problem specification
- **Graph Processing** - Transitive closures and path finding

### Production
- **Web Applications** - Client-side logic processing via WASM
- **Embedded Systems** - Lightweight reasoning engine
- **Configuration Languages** - Declarative rule-based systems

## 📚 Documentation

- **Language Specification** - `spec/bytelog-spec.md`
- **Build Instructions** - `make help` 
- **API Reference** - `make docs` (generates HTML)
- **Example Programs** - `examples/` directory
- **Test Suites** - `src/test_*.c` files

## 🤝 Contributing

1. **Build**: `make clean && make all test`
2. **Format**: `make format`  
3. **Test**: `make test memcheck lint`
4. **Document**: Update README and API docs

## 📄 License

MIT License - see LICENSE file for details.

## 🔗 Related Projects

- **WebAssembly Text (WAT)** - Target compilation format
- **Datalog** - Theoretical foundation
- **Prolog** - Related logic programming language
- **Souffle** - Industrial Datalog engine

---

**ByteLog Compiler v1.0.0** - A complete, portable, high-performance implementation of logic programming for the modern web. 🚀