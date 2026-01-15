# WASM WorldModel Quickstart

## When Your Other Training is Done

### 1. Check System Status
```bash
cd wasm/
python system_status.py
```

### 2. Start WASM Training

**Basic Training (10 epochs):**
```bash
python train_wasm_worldmodel.py
```

**Long Training (30 epochs):**
```bash
python train_wasm_worldmodel.py --epochs 30
```

**Fast Development (no sandbox):**
```bash
python train_wasm_worldmodel.py --no-sandbox --epochs 5
```

### 3. Monitor Training
- Automatic checkpointing every ~10 times per epoch
- Resume automatically if interrupted
- Emergency saves on Ctrl+C or errors
- Performance metrics every 50 iterations

### 4. After Training Completes

**Interactive Inference:**
```bash
python run_wasm_inference.py --model ./wasm_worldmodel_output/final_model
```

**Single Query:**
```bash
python run_wasm_inference.py --model ./wasm_worldmodel_output/final_model \
    --query "Calculate 17 times 23"
```

**Benchmark Mode:**
```bash
python run_wasm_inference.py --model ./wasm_worldmodel_output/final_model --benchmark
```

### 5. Test WASM Execution

**Test live execution during forward pass:**
```bash
python test_live_execution.py
```

**Demo arithmetic calculations:**
```bash
python demo_wasm_execution.py
```

## Expected Results

After training, the model should:
- Generate proper WASM tokens (not UNK tokens)
- Execute math during reasoning: `17×23 → <computed>391</computed>`
- Give precise answers with computational provenance
- Show `<computed>` tokens in responses

## Key Files

- `train_wasm_worldmodel.py` - Main training script
- `run_wasm_inference.py` - Inference script  
- `system_status.py` - Check system health
- `wasm_worldmodel_output/final_model/` - Trained model location

## Training Data

- **1,071 examples** across 3 curriculum stages
- Basic arithmetic → System operations → Complex logic
- Text + WASM + execution examples

## Architecture

- **Cross-modal transformer** (text ↔ WASM streams)
- **Live execution** at attention layers [3, 7, 11]
- **Computed token injection** into generation
- **QEMU sandbox** for external API calls (default enabled)

## What Makes This Special

🧠 **WASM executes WHILE the model thinks**
🔢 **Real calculations during forward pass**  
🏷️ **Computed provenance in tokens**
🎯 **Deterministic reasoning, not hallucination**

## Status: Ready to Train!

The system executes WASM code during token generation. Training will teach it to generate useful WASM programs instead of random tokens.

**Live execution demo:** `17×23 → 391.0` ✅ Working!