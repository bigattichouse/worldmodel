# Tests Directory

This directory contains all test scripts for the ByteLogic WorldModel system.

## Test Scripts

### `run_all_tests.py` 
**Main test runner** - Run this first
```bash
python3 run_all_tests.py
```

### Individual Tests

#### `test_bytelogic_simple.py`
- Tests ByteLogic compiler functionality
- Basic tokenization and syntax validation
- WAT compilation pipeline
- Example program execution

#### `test_bytelogic_integration.py`
- Comprehensive integration tests
- Tests all ByteLogic components
- Training data generation
- Model adapter integration

#### `test_training_pipeline.py`
- Tests the training pipeline
- Dataset loading and processing
- Tokenizer integration
- Curriculum learning

#### `validate_bytelogic_training_data.py`
- Validates training dataset quality
- Syntax checking with ByteLogic compiler
- Token format verification
- Quality statistics

#### `check_token_migration.py`
- Verifies WAT → ByteLogic migration
- Checks for remaining WAT tokens
- Confirms computation token usage

## Quick Start

1. **Run all tests:**
   ```bash
   cd tests
   python3 run_all_tests.py
   ```

2. **Run individual test:**
   ```bash
   python3 test_bytelogic_simple.py
   ```

3. **Validate specific dataset:**
   ```bash
   python3 validate_bytelogic_training_data.py ../training/datasets/corrected_bytelogic_dataset.json
   ```

## Expected Output

All tests should pass:
```
🎉 All tests passed! System ready for training.
```

If tests fail, check:
- ByteLogic compiler is built (`cd ../bytelogic && make`)
- Training datasets exist in `../training/datasets/`
- Python dependencies are installed