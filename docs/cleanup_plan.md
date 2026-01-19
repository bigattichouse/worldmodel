# Directory Cleanup and Organization Plan

## Current Issues
- Scripts scattered in root directory
- Test files mixed with main code
- Multiple dataset generators
- Documentation spread across multiple locations
- Legacy files mixed with current code

## Proposed Structure

```
worldmodel/
├── README.md
├── QUICKSTART.md
├── requirements.txt
├── setup.sh
├── bytelogic/                    # Submodule (keep as-is)
├── src/                          # Core source code (organized)
├── tests/                        # All test scripts
├── tools/                        # Dataset generators and utilities
├── docs/                         # All documentation
├── training/                     # Training data and configs
├── scripts/                      # Utility and setup scripts
├── examples/                     # Usage examples
└── legacy/                       # Archive old files
```

## Files to Organize

### Move to tests/
- test_bytelogic_integration.py
- test_bytelogic_simple.py  
- test_training_pipeline.py
- validate_bytelogic_training_data.py
- check_token_migration.py

### Move to tools/
- generate_bytelogic_dataset.py
- comprehensive_bytelogic_generator.py
- corrected_bytelogic_generator.py
- simple_dataset_generator.py

### Move to docs/
- BYTELOGIC_TRAINING_READY.md
- TOKEN_MIGRATION_SUMMARY.md
- CRASH_FIX.md
- doc/* (merge into single docs/)
- spec/* (merge into docs/specs/)

### Move to scripts/
- run_inference.py
- setup.sh (keep copy in root)

### Move to legacy/
- train_worldmodel.py (old WAT-based training)
- data/wasm_*.txt (old WAT training data)

### Keep in Root
- README.md (main project readme)
- QUICKSTART.md 
- requirements.txt
- train_bytelogic_worldmodel.py (main training script)
- bytelogic/ (submodule)
- src/ (source code)
- training/ (training data)