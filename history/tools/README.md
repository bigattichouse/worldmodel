# Tools Directory

This directory contains utilities for dataset generation and development.

## Dataset Generators

### `corrected_bytelogic_generator.py`
**Recommended for training**
- Generates ByteLogic 2.0 compatible examples
- 1,000 high-quality examples
- 100% syntax validation
- Categories: family relationships, graph algorithms, logic programming

```bash
python3 corrected_bytelogic_generator.py
```

### `comprehensive_bytelogic_generator.py`
- Generates examples using ByteLogic 3.0 features (may need compiler updates)
- 1,650+ examples with advanced features
- Includes loops, calculations, string processing

### `simple_dataset_generator.py`
- Basic dataset generator without complex imports
- Useful for quick testing and prototyping

### `generate_bytelogic_dataset.py`
- Original dataset generator
- Uses the full training infrastructure

## Usage

1. **Generate recommended training data:**
   ```bash
   cd tools
   python3 corrected_bytelogic_generator.py
   ```

2. **Output location:**
   ```
   ../training/datasets/corrected_bytelogic_dataset.json
   ```

3. **Validate generated data:**
   ```bash
   cd ../tests
   python3 validate_bytelogic_training_data.py ../training/datasets/corrected_bytelogic_dataset.json
   ```