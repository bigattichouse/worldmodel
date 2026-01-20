#!/usr/bin/env python3
import os
import sys
sys.path.append("src")

# Setup ROCm
os.environ["HSA_OVERRIDE_GFX_VERSION"] = "9.0.6"
os.environ["PYTORCH_ROCM_ARCH"] = "gfx906"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from transformers import AutoTokenizer, AutoModelForCausalLM

def quick_test():
    print("🔄 Loading fine-tuned model...")
    
    # Load model and tokenizer directly
    tokenizer = AutoTokenizer.from_pretrained("./qwen3_0.6b_rocm_conservative/final_model")
    model = AutoModelForCausalLM.from_pretrained(
        "./qwen3_0.6b_rocm_conservative/final_model",
        device_map="auto",
        torch_dtype="auto"
    )
    
    print("✅ Model loaded! Testing WorldModel prompts...")
    
    test_prompts = [
        "Count the R's in strawberry",
        "What is 15% tip on $67.50?",
        "Calculate 127 + 89"
    ]
    
    for prompt in test_prompts:
        print(f"\n📝 Prompt: {prompt}")
        print("🤖 Response:")
        
        # Tokenize input
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        # Generate with conservative settings
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                max_new_tokens=128,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        # Decode response
        response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        print(response.strip())
        print("-" * 50)

if __name__ == "__main__":
    import torch
    quick_test()