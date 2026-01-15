# WorldModel Inference Guide

## After Training Completes

When your ROCm training finishes, you'll have a fine-tuned model ready for inference. Here's how to use it:

## Quick Start

### 1. Single Question
```bash
source venv/bin/activate
source /home/bigattichouse/workspace/rocm/setup-rocm-env.sh
export HSA_OVERRIDE_GFX_VERSION=9.0.6

python run_inference.py -m ./qwen3_0.6b_rocm_conservative -p "Count the R's in strawberry"
```

### 2. Interactive Chat
```bash
python run_inference.py -m ./qwen3_0.6b_rocm_conservative -i
```

## Model Types

The training creates two types of models:

### LoRA Adapter (Recommended)
- **Location**: `./qwen3_0.6b_rocm_conservative/` (adapter files)
- **Size**: ~50MB (small)
- **Usage**: Requires base model
- **Command**: 
```bash
python run_inference.py -m ./qwen3_0.6b_rocm_conservative -b ../model/Qwen3-0.6B -i
```

### Merged Model (Standalone)
- **Location**: `./qwen3_0.6b_rocm_conservative/merged_model/` (if created)
- **Size**: ~1.2GB (full model)
- **Usage**: Standalone
- **Command**:
```bash
python run_inference.py -m ./qwen3_0.6b_rocm_conservative/merged_model -i
```

## Command Line Options

```bash
python run_inference.py [options]

Required:
  -m, --model PATH        Path to fine-tuned model

Optional:
  -b, --base-model PATH   Path to base model (for LoRA adapters)
  -p, --prompt TEXT       Single prompt to process  
  -i, --interactive       Interactive chat mode
  --max-tokens INT        Maximum tokens to generate (default: 512)
  --temperature FLOAT     Sampling temperature (default: 0.7)
  --top-p FLOAT          Top-p sampling (default: 0.9)
```

## Example Sessions

### WorldModel Questions
```bash
👤 You: Count the R's in strawberry
🤖 WorldModel: I need to count each letter 'R' in the word "strawberry".

s-t-r-a-w-b-e-r-r-y

Looking at each letter:
- s: not an R
- t: not an R  
- r: this is an R (1st R)
- a: not an R
- w: not an R
- b: not an R
- e: not an R
- r: this is an R (2nd R) 
- r: this is an R (3rd R)
- y: not an R

Therefore, there are 3 R's in the word "strawberry".

👤 You: What's 25 + 37?
🤖 WorldModel: I need to add 25 + 37.

25 + 37 = 62

Therefore, 25 + 37 equals 62.
```

## Performance Tips

### GPU Memory Usage
- **Inference**: Uses ~3-5GB VRAM (much less than training)
- **Multiple sessions**: Can run multiple inference processes
- **Batch inference**: Process multiple prompts efficiently

### Speed Optimization
```bash
# For faster inference, reduce precision
python run_inference.py -m ./model --temperature 0.1 --max-tokens 256
```

### Memory Optimization
```bash
# For memory-limited systems, use CPU
python run_inference.py -m ./model -i  # Auto-detects available hardware
```

## Troubleshooting

### Model Not Found
```bash
# Check if training completed and model was saved
ls -la ./qwen3_0.6b_rocm_conservative/
```

### ROCm Issues
```bash
# Ensure ROCm environment is set up
source /home/bigattichouse/workspace/rocm/setup-rocm-env.sh
export HSA_OVERRIDE_GFX_VERSION=9.0.6
```

### Memory Issues During Inference
- Reduce `--max-tokens` to 256 or 128
- Use CPU inference (automatically falls back)
- Close other GPU applications

## Integration with Other Tools

### Using with llama.cpp (for deployment)
1. Convert the fine-tuned model to GGUF format:
```bash
# (Future enhancement - model conversion script)
python convert_to_gguf.py ./qwen3_0.6b_rocm_conservative
```

2. Use with llama.cpp for optimized inference:
```bash
./llama-cli -m model.gguf -p "Your prompt here" --gpu-layers 32
```

### API Server (Future)
```bash
# Run as REST API server
python api_server.py -m ./qwen3_0.6b_rocm_conservative --port 8000
```

## Model Quality Assessment

Test your fine-tuned model with WorldModel-style questions:
- Counting tasks: "Count the R's in strawberry"  
- Step-by-step reasoning: "What's 127 + 89?"
- Logic problems: "If I have 3 apples and eat 1, how many remain?"
- Explanation tasks: "Why is the sky blue?"

Compare responses to the base model to see improvement from fine-tuning.

## Next Steps

1. **Test thoroughly**: Try various WorldModel prompts
2. **Adjust parameters**: Temperature, top-p for different use cases  
3. **Deploy**: Integrate into your applications
4. **Iterate**: Fine-tune further if needed

Your WorldModel is now ready for production use on ROCm! 🎉