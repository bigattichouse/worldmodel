# Training Datasets

This directory contains ByteLogic training datasets in different formats and versions.

## Primary Datasets (Use These)

### `corrected_bytelogic_dataset.json` ⭐ **RECOMMENDED**
- **1,000 examples** (100% syntax validated)
- **ByteLogic 2.0 compatible** (works with current compiler)
- **Perfect quality**: 0 syntax errors
- **Categories**: Family relationships, graph algorithms, logic programming
- **Format**: JSON with train/validation/test splits

### `comprehensive_bytelogic_dataset.json`
- **1,650 examples** with advanced features
- **ByteLogic 3.0 syntax** (requires compiler updates)
- **Advanced features**: Loops, calculations, string processing
- **Status**: Some syntax errors due to unsupported features

## JSONL Exports (Auto-Generated)

These are automatically generated from the main JSON files:

### Corrected Dataset (JSONL)
- `bytelogic_train_corrected.jsonl` - 800 training examples
- `bytelogic_validation_corrected.jsonl` - 100 validation examples  
- `bytelogic_test_corrected.jsonl` - 100 test examples

### Comprehensive Dataset (JSONL)
- `bytelogic_train_comprehensive.jsonl` - 1,320 training examples
- `bytelogic_validation_comprehensive.jsonl` - 165 validation examples
- `bytelogic_test_comprehensive.jsonl` - 165 test examples

## Usage Recommendations

### For Production Training
```bash
python3 train_bytelogic_worldmodel.py \
  --dataset training/datasets/corrected_bytelogic_dataset.json
```

### For Experimental Training (Advanced Features)
```bash
python3 train_bytelogic_worldmodel.py \
  --dataset training/datasets/comprehensive_bytelogic_dataset.json
```

### For Streaming Training (Large Datasets)
```bash
python3 train_bytelogic_worldmodel.py \
  --dataset training/datasets/bytelogic_train_corrected.jsonl
```

## Dataset Consolidation

**All datasets are consolidated into the main JSON files.** The JSONL files are exports for convenience:

- `corrected_bytelogic_dataset.json` = All corrected examples
- `comprehensive_bytelogic_dataset.json` = All comprehensive examples  
- Individual JSONL files = Splits for streaming/memory efficiency

You can use either:
1. **JSON files** (recommended) - Load entire dataset with splits
2. **JSONL files** - Stream individual splits for large-scale training

## File Sizes

- **corrected_bytelogic_dataset.json**: 1.1MB (1,000 examples)
- **comprehensive_bytelogic_dataset.json**: 2.1MB (1,650 examples)
- **Total JSONL files**: ~3MB (all splits combined)

**Recommendation**: Use `corrected_bytelogic_dataset.json` for reliable training with 100% validated syntax.