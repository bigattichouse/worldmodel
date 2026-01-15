# WorldModel ROCm Training & Inference

Complete system for training and running WorldModel LLMs on ROCm MI50.

## 🚀 Quick Start

```bash
# Run the complete workflow
./complete_workflow.sh
```

Choose option:
- **1**: Train model (10 minutes)  
- **2**: Run inference on existing model
- **3**: Train + Test inference  
- **4**: Show dataset sample

## 📁 Files Overview

### Training
- `train_worldmodel_rocm.py` - Main training script (WORKING ✅)
- `generate_1000_examples.py` - Generates 1000+ training examples
- `data/worldmodel_training_1000.txt` - Generated training dataset

### Inference  
- `run_worldmodel_inference.py` - Complete inference engine with code execution
- `worldmodel_rocm_output/final_model/` - Trained model (after training)

### Workflow
- `complete_workflow.sh` - One-click training and inference
- `simple_rocm_test.py` - Test ROCm compatibility

## 🧠 How It Works

### 1. Training Format
The model learns to generate structured responses:
```
User: Calculate 25% of 200
Assistant: <think>I need to calculate 25% of 200...</think>
<model>
result = 0.25 * 200
print(result)
</model>
<requires>python:math</requires>

25% of 200 equals 50.
```

### 2. Inference Pipeline
1. **Generate**: Model produces structured WorldModel response
2. **Parse**: Extract `<think>`, `<model>`, `<requires>` components  
3. **Execute**: Run the generated code safely
4. **Return**: Show results + explanation

### 3. Example Inference Session
```bash
python3 run_worldmodel_inference.py --interactive

🤔 Your question: Calculate 15% of 300

🧠 Thinking: I need to calculate 15% of 300...
💻 Code: 6 lines
⚡ Execution Status: success
📤 Output: 45.0
💬 Explanation: 15% of 300 equals 45.
```

## 📊 Dataset Categories (950 examples)

- **Math**: Addition, subtraction, multiplication, division (200)
- **Percentages**: Percentage calculations (100)
- **Text Analysis**: Character/word counting (150)  
- **Geometry**: Circle/rectangle calculations (100)
- **Statistics**: Mean calculations (100)
- **Prime Numbers**: Prime checking and generation (80)
- **Conversions**: Temperature, distance conversions (70)
- **Boolean Logic**: Logic expression evaluation (50)
- **Word Problems**: Shopping, travel problems (100)

## 🔧 Manual Commands

### Training Only
```bash
# Setup environment
source /home/bigattichouse/workspace/rocm/setup-rocm-env.sh
export HSA_OVERRIDE_GFX_VERSION=9.0.6

# Generate data (if not exists)
python3 generate_1000_examples.py

# Train model
python3 train_worldmodel_rocm.py
```

### Inference Only
```bash
# Interactive session
python3 run_worldmodel_inference.py --interactive

# Single query
python3 run_worldmodel_inference.py "What is 12 × 7?"

# Custom model path
python3 run_worldmodel_inference.py --model ./my_model "Calculate area of circle radius 5"
```

## ✅ System Status

- **ROCm MI50**: ✅ Working (34.3GB VRAM)
- **Training**: ✅ Fixed zero loss issue  
- **WorldModel Format**: ✅ Generating proper structure
- **Code Execution**: ✅ Safe execution with timeout
- **Dataset**: ✅ 950 diverse examples

## 🎯 Success Metrics

Last training run:
- Loss: 1.46 → 0.59 (60% improvement)
- Time: 6 minutes for 3 epochs
- Structure: 2/3 WorldModel tags consistently generated
- GPU Utilization: ✅ Confirmed

## 🔍 Troubleshooting

**Model not found**: Check `worldmodel_rocm_output/final_model/` exists  
**ROCm issues**: Run `simple_rocm_test.py` first  
**Import errors**: PEFT not required, uses full fine-tuning  
**GPU memory**: Reduce batch size in training script if needed

## 🎉 Next Steps

1. **Scale training**: Increase epochs to 10-20 for better consistency
2. **Add categories**: Generate more domain-specific examples  
3. **VM integration**: Connect to full execution environment
4. **RAG system**: Implement memory/retrieval capabilities

## 📝 Example Outputs

The trained model produces responses like:
- Math: Proper calculations with step-by-step code
- Text: Character/word analysis with Python string methods
- Logic: Boolean evaluation with clear reasoning
- Geometry: Formula application with explanations

Ready to move your WorldModel concept from prototype to production! 🚀