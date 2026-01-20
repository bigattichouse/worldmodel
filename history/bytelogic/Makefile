# ═══════════════════════════════════════════════════════════════════════════
# ByteLog Compiler Makefile
# ═══════════════════════════════════════════════════════════════════════════
#
# Portable build system for the ByteLog compiler and tools.
# Supports development, testing, WebAssembly compilation, and packaging.
#
# Quick start:
#   make           - Build all targets
#   make test      - Run all unit tests  
#   make demo      - Run ByteLog interpreter
#   make clean     - Remove build artifacts
#   make help      - Show all available targets
#
# ═══════════════════════════════════════════════════════════════════════════

# ─────────────────────────────────────────────────────────────────────────
# Project Configuration
# ───────────────────────────────────────────────────────────────────────── 

PROJECT_NAME = ByteLog Compiler
VERSION = 1.0.0

# Compiler settings
CC = gcc
CFLAGS = -std=c99 -Wall -Wextra -Wpedantic -O2 -D_GNU_SOURCE
DEBUG_CFLAGS = -std=c99 -Wall -Wextra -Wpedantic -g -DDEBUG -O0 -D_GNU_SOURCE
TEST_CFLAGS = -std=c99 -Wall -Wextra -Wpedantic -g -O0 -D_GNU_SOURCE

# ─────────────────────────────────────────────────────────────────────────
# Directory Structure
# ───────────────────────────────────────────────────────────────────────── 

SRC_DIR = src
INCLUDE_DIR = includes
BUILD_DIR = build
TEST_DIR = $(SRC_DIR)
EXAMPLE_DIR = examples
DOC_DIR = docs

# Include path
INCLUDES = -I$(INCLUDE_DIR)

# ─────────────────────────────────────────────────────────────────────────
# Source Files and Targets
# ───────────────────────────────────────────────────────────────────────── 

# Core library sources (order matters for dependencies)
CORE_SOURCES = lexer.c ast.c atoms.c parser.c engine.c wat_gen.c
CORE_OBJECTS = $(addprefix $(BUILD_DIR)/, $(CORE_SOURCES:.c=.o))

# Executable sources  
BYTELOGIC_SOURCE = demo.c
WAT_COMPILER_SOURCE = wat_compiler.c

# Test sources
TEST_SOURCES = test_lexer.c test_parser.c test_ast.c test_atoms.c

# Output executables
BYTELOGIC = $(BUILD_DIR)/bytelogic
WAT_COMPILER = $(BUILD_DIR)/wat_compiler
TEST_EXECUTABLES = $(addprefix $(BUILD_DIR)/, $(TEST_SOURCES:.c=))

# ─────────────────────────────────────────────────────────────────────────
# Default Target
# ───────────────────────────────────────────────────────────────────────── 

.PHONY: all
all: $(BUILD_DIR) $(CORE_OBJECTS) $(BYTELOGIC) $(WAT_COMPILER)
	@echo ""
	@echo "✅ $(PROJECT_NAME) v$(VERSION) built successfully!"
	@echo ""
	@echo "Available executables:"
	@echo "  $(BYTELOGIC)    - ByteLog interpreter and analyzer"
	@echo "  $(WAT_COMPILER) - WebAssembly Text compiler"
	@echo ""
	@echo "Run 'make help' for all available commands."

# ─────────────────────────────────────────────────────────────────────────
# Build Directory
# ───────────────────────────────────────────────────────────────────────── 

$(BUILD_DIR):
	@echo "📁 Creating build directory..."
	@mkdir -p $(BUILD_DIR)

# ─────────────────────────────────────────────────────────────────────────
# Core Library Compilation
# ───────────────────────────────────────────────────────────────────────── 

# Generic rule for core library objects
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.c $(INCLUDE_DIR)/%.h | $(BUILD_DIR)
	@echo "🔨 Compiling $<..."
	@$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@

# Special dependencies (files that include multiple headers)
$(BUILD_DIR)/parser.o: $(SRC_DIR)/parser.c $(INCLUDE_DIR)/parser.h \
                       $(INCLUDE_DIR)/lexer.h $(INCLUDE_DIR)/ast.h \
                       $(INCLUDE_DIR)/atoms.h | $(BUILD_DIR)
	@echo "🔨 Compiling parser.c..."
	@$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@

$(BUILD_DIR)/engine.o: $(SRC_DIR)/engine.c $(INCLUDE_DIR)/engine.h \
                       $(INCLUDE_DIR)/ast.h $(INCLUDE_DIR)/atoms.h \
                       $(INCLUDE_DIR)/parser.h | $(BUILD_DIR)
	@echo "🔨 Compiling engine.c..."
	@$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@

$(BUILD_DIR)/wat_gen.o: $(SRC_DIR)/wat_gen.c $(INCLUDE_DIR)/wat_gen.h \
                        $(INCLUDE_DIR)/ast.h $(INCLUDE_DIR)/atoms.h \
                        $(INCLUDE_DIR)/parser.h | $(BUILD_DIR)
	@echo "🔨 Compiling wat_gen.c..."
	@$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@

# ─────────────────────────────────────────────────────────────────────────
# Executable Targets
# ───────────────────────────────────────────────────────────────────────── 

$(BYTELOGIC): $(SRC_DIR)/$(BYTELOGIC_SOURCE) $(CORE_OBJECTS) | $(BUILD_DIR)
	@echo "🔧 Building ByteLog interpreter..."
	@$(CC) $(CFLAGS) $(INCLUDES) $< $(CORE_OBJECTS) -o $@

$(WAT_COMPILER): $(SRC_DIR)/$(WAT_COMPILER_SOURCE) $(CORE_OBJECTS) | $(BUILD_DIR)
	@echo "🔧 Building WAT compiler..."
	@$(CC) $(CFLAGS) $(INCLUDES) $< $(CORE_OBJECTS) -o $@

# ─────────────────────────────────────────────────────────────────────────
# Test Executables
# ───────────────────────────────────────────────────────────────────────── 

$(BUILD_DIR)/test_lexer: $(SRC_DIR)/test_lexer.c $(BUILD_DIR)/lexer.o | $(BUILD_DIR)
	@echo "🧪 Building lexer tests..."
	@$(CC) $(TEST_CFLAGS) $(INCLUDES) $< $(BUILD_DIR)/lexer.o -o $@

$(BUILD_DIR)/test_ast: $(SRC_DIR)/test_ast.c $(BUILD_DIR)/ast.o | $(BUILD_DIR)
	@echo "🧪 Building AST tests..."
	@$(CC) $(TEST_CFLAGS) $(INCLUDES) $< $(BUILD_DIR)/ast.o -o $@

$(BUILD_DIR)/test_parser: $(SRC_DIR)/test_parser.c $(CORE_OBJECTS) | $(BUILD_DIR)
	@echo "🧪 Building parser tests..."
	@$(CC) $(TEST_CFLAGS) $(INCLUDES) $< $(CORE_OBJECTS) -o $@

$(BUILD_DIR)/test_atoms: $(SRC_DIR)/test_atoms.c $(CORE_OBJECTS) | $(BUILD_DIR)
	@echo "🧪 Building atom tests..."
	@$(CC) $(TEST_CFLAGS) $(INCLUDES) $< $(CORE_OBJECTS) -o $@

# ─────────────────────────────────────────────────────────────────────────
# Test Targets
# ───────────────────────────────────────────────────────────────────────── 

.PHONY: test test-lexer test-ast test-parser test-atoms
test: test-lexer test-ast test-parser test-atoms
	@echo ""
	@echo "🎉 All tests completed successfully!"

test-lexer: $(BUILD_DIR)/test_lexer
	@echo "🧪 Running lexer tests..."
	@$(BUILD_DIR)/test_lexer

test-ast: $(BUILD_DIR)/test_ast
	@echo "🧪 Running AST tests..."
	@$(BUILD_DIR)/test_ast

test-parser: $(BUILD_DIR)/test_parser
	@echo "🧪 Running parser tests..."
	@$(BUILD_DIR)/test_parser

test-atoms: $(BUILD_DIR)/test_atoms
	@echo "🧪 Running atom tests..."
	@$(BUILD_DIR)/test_atoms

# ─────────────────────────────────────────────────────────────────────────
# Development and Demo Targets
# ───────────────────────────────────────────────────────────────────────── 

.PHONY: debug
debug: CFLAGS = $(DEBUG_CFLAGS)
debug: clean all
	@echo "🐛 Debug build completed with symbols and debugging enabled."

.PHONY: demo
demo: $(BYTELOGIC)
	@echo "🚀 Running ByteLog interpreter..."
	@$(BYTELOGIC) $(EXAMPLE_DIR)/example_family.bl

.PHONY: wat
wat: $(WAT_COMPILER)
	@echo "📦 ByteLog to WAT Compiler ready!"
	@echo ""
	@echo "Usage:"
	@echo "  $(WAT_COMPILER) input.bl [output.wat]"
	@echo ""
	@echo "Example:"
	@echo "  $(WAT_COMPILER) $(EXAMPLE_DIR)/example_family.bl"

.PHONY: check
check: test
	@echo "✅ All checks passed!"

# ─────────────────────────────────────────────────────────────────────────
# Quality Assurance
# ───────────────────────────────────────────────────────────────────────── 

.PHONY: memcheck
memcheck: $(TEST_EXECUTABLES)
	@echo "🔍 Running memory checks..."
	@command -v valgrind >/dev/null 2>&1 || { echo "❌ Valgrind not found, skipping memory checks"; exit 0; }
	@for test in $(TEST_EXECUTABLES); do \
		echo "  Checking $$test..."; \
		valgrind --leak-check=full --error-exitcode=1 $$test > /dev/null || exit 1; \
	done
	@echo "✅ All memory checks passed!"

.PHONY: lint
lint:
	@echo "🔍 Running static analysis..."
	@command -v cppcheck >/dev/null 2>&1 || { echo "❌ cppcheck not found, skipping static analysis"; exit 0; }
	@cppcheck --enable=all --std=c99 --suppress=unusedFunction \
		$(SRC_DIR)/*.c --include=$(INCLUDE_DIR)
	@echo "✅ Static analysis completed!"

.PHONY: format
format:
	@echo "🎨 Formatting code..."
	@command -v clang-format >/dev/null 2>&1 || { echo "❌ clang-format not found, skipping formatting"; exit 0; }
	@clang-format -i $(SRC_DIR)/*.c $(INCLUDE_DIR)/*.h
	@echo "✅ Code formatting completed!"

# ─────────────────────────────────────────────────────────────────────────
# Examples and Documentation
# ───────────────────────────────────────────────────────────────────────── 

.PHONY: examples
examples: $(BYTELOGIC) $(WAT_COMPILER)
	@echo "🎯 Running all examples..."
	@echo ""
	@echo "═══ Family Relations Example ═══"
	@$(BYTELOGIC) $(EXAMPLE_DIR)/example_family.bl
	@echo ""
	@echo "═══ Atom Usage Example ═══"
	@$(BYTELOGIC) $(EXAMPLE_DIR)/example_atoms.bl
	@echo ""
	@echo "🔧 Compiling examples to WebAssembly..."
	@$(WAT_COMPILER) $(EXAMPLE_DIR)/example_family.bl $(BUILD_DIR)/example_family.wat
	@$(WAT_COMPILER) $(EXAMPLE_DIR)/example_atoms.bl $(BUILD_DIR)/example_atoms.wat
	@echo "✅ WebAssembly files generated in $(BUILD_DIR)/"

.PHONY: docs
docs:
	@echo "📚 Generating documentation..."
	@command -v doxygen >/dev/null 2>&1 || { echo "❌ doxygen not found, skipping documentation"; exit 0; }
	@doxygen Doxyfile 2>/dev/null
	@echo "✅ Documentation generated in $(DOC_DIR)/"

# ─────────────────────────────────────────────────────────────────────────
# Packaging and Distribution
# ───────────────────────────────────────────────────────────────────────── 

.PHONY: dist
dist: clean
	@echo "📦 Creating distribution package..."
	@VERSION=$$(date +%Y%m%d); \
	PACKAGE="bytelog-compiler-$$VERSION"; \
	mkdir -p "$$PACKAGE"; \
	cp -r $(SRC_DIR) $(INCLUDE_DIR) $(EXAMPLE_DIR) Makefile README.md "$$PACKAGE/"; \
	tar czf "$$PACKAGE.tar.gz" "$$PACKAGE"; \
	rm -rf "$$PACKAGE"; \
	echo "✅ Created $$PACKAGE.tar.gz"

.PHONY: install
install: all
	@echo "🚀 Installing ByteLog Compiler..."
	@PREFIX=${PREFIX:-/usr/local}; \
	mkdir -p "$$PREFIX/bin"; \
	cp $(BYTELOGIC) "$$PREFIX/bin/bytelogic"; \
	cp $(WAT_COMPILER) "$$PREFIX/bin/bytelog-wat"; \
	echo "✅ Installed to $$PREFIX/bin/"

# ─────────────────────────────────────────────────────────────────────────
# Cleaning Targets
# ───────────────────────────────────────────────────────────────────────── 

.PHONY: clean
clean:
	@echo "🧹 Cleaning build artifacts..."
	@rm -rf $(BUILD_DIR)
	@rm -f *.wat core vgcore.*

.PHONY: distclean
distclean: clean
	@echo "🧹 Deep cleaning..."
	@rm -rf $(DOC_DIR)/html $(DOC_DIR)/latex
	@rm -f *.tar.gz

# ─────────────────────────────────────────────────────────────────────────
# Help and Information
# ───────────────────────────────────────────────────────────────────────── 

.PHONY: help
help:
	@echo "$(PROJECT_NAME) v$(VERSION) Build System"
	@echo "═══════════════════════════════════════════════════════════"
	@echo ""
	@echo "🚀 Primary Targets:"
	@echo "  all        - Build all executables (default)"
	@echo "  demo       - Build and run ByteLog interpreter"
	@echo "  wat        - Build WebAssembly Text compiler"
	@echo "  test       - Run all unit tests (96 tests)"
	@echo "  examples   - Run all example programs"
	@echo ""
	@echo "🧪 Testing & Quality:"
	@echo "  test-*     - Run specific test suite (lexer, ast, parser, atoms)"
	@echo "  memcheck   - Run tests with Valgrind memory checking"
	@echo "  lint       - Static analysis with cppcheck"
	@echo "  format     - Format code with clang-format"
	@echo "  check      - Run all tests and checks"
	@echo ""
	@echo "🛠️  Development:"
	@echo "  debug      - Build with debug symbols and assertions"
	@echo "  clean      - Remove build artifacts"
	@echo "  distclean  - Remove all generated files"
	@echo ""
	@echo "📦 Distribution:"
	@echo "  dist       - Create source distribution package"
	@echo "  install    - Install to system (PREFIX=/usr/local)"
	@echo "  docs       - Generate API documentation"
	@echo ""
	@echo "🏗️  Build Configuration:"
	@echo "  CC         = $(CC)"
	@echo "  CFLAGS     = $(CFLAGS)"
	@echo "  INCLUDES   = $(INCLUDES)"
	@echo "  BUILD_DIR  = $(BUILD_DIR)"
	@echo ""
	@echo "📁 Directory Structure:"
	@echo "  $(SRC_DIR)/         - Source files (.c)"
	@echo "  $(INCLUDE_DIR)/     - Header files (.h)"  
	@echo "  $(EXAMPLE_DIR)/     - Example ByteLog programs (.bl)"
	@echo "  $(BUILD_DIR)/       - Build artifacts and executables"
	@echo ""
	@echo "🎯 Quick Examples:"
	@echo "  make && make demo                              # Build and run interpreter"
	@echo "  make wat && ./$(WAT_COMPILER) examples/family.bl   # Compile to WASM"
	@echo "  make test                                      # Run all tests"
	@echo "  make memcheck                                  # Memory leak detection"

.PHONY: info
info:
	@echo "$(PROJECT_NAME) v$(VERSION) Project Information"
	@echo "═══════════════════════════════════════════════════════════"
	@echo ""
	@echo "🏗️  Architecture:"
	@echo "  • Lexical Analysis    - Tokenization of ByteLog source"
	@echo "  • Syntax Analysis     - Recursive descent parser"
	@echo "  • Abstract Syntax Tree - Program representation"
	@echo "  • Atom System         - String interning for readable names"
	@echo "  • Execution Engine    - Datalog evaluation with fixpoint computation"
	@echo "  • WebAssembly Backend - Code generation for WASM deployment"
	@echo ""
	@echo "📊 Statistics:"
	@echo "  • Source files: $(words $(CORE_SOURCES)) core + $(words $(TEST_SOURCES)) test"
	@echo "  • Unit tests: 96 tests across all modules"
	@echo "  • Language features: REL, FACT, RULE, SCAN, JOIN, EMIT, SOLVE, QUERY"
	@echo "  • Target platforms: Native C99, WebAssembly"
	@echo ""
	@echo "🎯 Use Cases:"
	@echo "  • Logic programming and constraint solving"
	@echo "  • Datalog query processing"
	@echo "  • Web-based logic applications (via WASM)"
	@echo "  • Educational compiler implementation"

# ─────────────────────────────────────────────────────────────────────────
# Platform-specific Configuration
# ───────────────────────────────────────────────────────────────────────── 

# Windows compatibility
ifeq ($(OS),Windows_NT)
    RM = del /Q
    MKDIR = if not exist $(BUILD_DIR) mkdir $(BUILD_DIR)
    EXE_SUFFIX = .exe
else
    RM = rm -f
    MKDIR = mkdir -p $(BUILD_DIR)
    EXE_SUFFIX = 
endif

# Compiler detection and optimization
ifeq ($(CC),clang)
    CFLAGS += -Weverything -Wno-padded -Wno-switch-enum
endif

# ─────────────────────────────────────────────────────────────────────────
# Special Targets for CI/CD
# ───────────────────────────────────────────────────────────────────────── 

.PHONY: ci
ci: all test lint
	@echo "🤖 CI pipeline completed successfully!"

.PHONY: test-compilers
test-compilers: clean
	@echo "🔧 Testing with different compilers..."
	@for compiler in gcc clang; do \
		if command -v $$compiler >/dev/null 2>&1; then \
			echo "Testing with $$compiler..."; \
			$(MAKE) CC=$$compiler all test || exit 1; \
			$(MAKE) clean; \
		fi; \
	done
	@echo "✅ All compiler tests passed!"

# ─────────────────────────────────────────────────────────────────────────
# Dependency Tracking (Advanced)
# ───────────────────────────────────────────────────────────────────────── 

# Automatic header dependency generation
-include $(CORE_OBJECTS:.o=.d)

$(BUILD_DIR)/%.d: $(SRC_DIR)/%.c | $(BUILD_DIR)
	@$(CC) -MM -MT $(@:.d=.o) $(INCLUDES) $< > $@

.PHONY: deps
deps: $(CORE_OBJECTS:.o=.d)
	@echo "🔗 Dependency files generated"

# ═══════════════════════════════════════════════════════════════════════════
# End of Makefile
# ═══════════════════════════════════════════════════════════════════════════