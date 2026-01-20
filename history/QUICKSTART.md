# ByteLogic WorldModel - Quick Start Guide

Get up and running with ByteLogic-powered structured reasoning in 5 minutes.

## 🚀 Quick Setup

### 1. Build ByteLogic Compiler
```bash
cd bytelogic
make
cd ..
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Tests
```bash
cd tests
python3 run_all_tests.py
cd ..
```

### 4. Start Training
```bash
python3 train_bytelogic_worldmodel.py \
  --dataset training/datasets/corrected_bytelogic_dataset.json \
  --model ../model/Qwen3-0.6B \
  --epochs 3
```

## 📁 Project Structure

```
worldmodel/
├── train_bytelogic_worldmodel.py    # Main training script
├── bytelogic/                       # ByteLogic compiler (submodule)
├── src/                             # Core source code
├── tests/                           # All test scripts
├── tools/                           # Dataset generators
├── training/datasets/               # Training data
├── docs/                           # Documentation
├── scripts/                        # Utility scripts
└── legacy/                         # Archived files
```

## 🧪 Testing

**Run all tests:**
```bash
cd tests && python3 run_all_tests.py
```

**Individual tests:**
```bash
cd tests
python3 test_bytelogic_simple.py              # Basic functionality
python3 test_training_pipeline.py             # Training pipeline
python3 validate_bytelogic_training_data.py   # Data validation
```

## 🛠️ Tools

**Generate training data:**
```bash
cd tools
python3 corrected_bytelogic_generator.py
```

**Validate migration:**
```bash
cd tests  
python3 check_token_migration.py
```

## 📚 Key Files

- **Training**: `train_bytelogic_worldmodel.py`
- **Dataset**: `training/datasets/corrected_bytelogic_dataset.json` 
- **Tests**: `tests/run_all_tests.py`
- **Documentation**: `docs/BYTELOGIC_TRAINING_READY.md`

## ✅ Success Criteria

1. ✅ All tests pass (`tests/run_all_tests.py`)
2. ✅ Training data validates (1,000 examples, 100% syntax correct)
3. ✅ No WAT tokens remain (full ByteLogic migration)
4. ✅ Training completes without errors

Ready to train! 🎉