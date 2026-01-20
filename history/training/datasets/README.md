# Training Datasets

This directory contains ByteLogic training datasets organized for clarity.

## 🏆 **Main Dataset (Use This)**

### `complete_bytelogic_dataset.json` ⭐ **RECOMMENDED**
- **1,100 examples** total (ByteLogic language + error handling)
- **Format**: JSON with train/validation/test splits (880/110/110)
- **Features**: Core ByteLogic 2.0 syntax + error detection/recovery
- **Quality**: 100% syntax validated + comprehensive error handling
- **Best for**: Production training with robust error handling

## 📁 **Component Datasets (parts/ subdirectory)**

Individual datasets are available in `parts/` for reference:

### Core Language Components
- `parts/corrected_bytelogic_dataset.json` - 1,000 core ByteLogic examples
- `parts/comprehensive_bytelogic_dataset.json` - 1,650 experimental examples (ByteLogic 3.0)

### Error Handling Component  
- `parts/bytelogic_error_handling_dataset.json` - 100 error handling examples

### JSONL Exports
- `parts/bytelogic_train_*.jsonl` - Training splits
- `parts/bytelogic_validation_*.jsonl` - Validation splits
- `parts/bytelogic_test_*.jsonl` - Test splits

## Usage Recommendations

### For Production Training (Recommended)
```bash
python3 train_bytelogic_worldmodel.py \
  --dataset training/datasets/complete_bytelogic_dataset.json
```

### For Core Language Only
```bash
python3 train_bytelogic_worldmodel.py \
  --dataset training/datasets/parts/corrected_bytelogic_dataset.json
```

### For Experimental Training (Advanced Features)
```bash
python3 train_bytelogic_worldmodel.py \
  --dataset training/datasets/parts/comprehensive_bytelogic_dataset.json
```

## Dataset Organization

**Main dataset**: `complete_bytelogic_dataset.json` combines all training data for production use.

**Component datasets** in `parts/`:
- Core ByteLogic examples (1,000)
- Error handling examples (100) 
- Experimental advanced features (1,650)
- JSONL exports for streaming

**File Sizes**:
- `complete_bytelogic_dataset.json`: 1.2MB (1,100 examples) ⭐ **USE THIS**
- `parts/corrected_bytelogic_dataset.json`: 1.1MB (1,000 examples)
- `parts/comprehensive_bytelogic_dataset.json`: 2.1MB (1,650 examples)
- `parts/bytelogic_error_handling_dataset.json`: 0.1MB (100 examples)

**Recommendation**: Use `complete_bytelogic_dataset.json` for production training with comprehensive error handling.