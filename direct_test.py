#!/usr/bin/env python3
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Setup ROCm
os.environ["HSA_OVERRIDE_GFX_VERSION"] = "9.0.6"
os.environ["PYTORCH_ROCM_ARCH"] = "gfx906" 
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def direct_test():
    print("🔄 Testing direct generation...")
    
    # Load directly
    tokenizer = AutoTokenizer.from_pretrained("./qwen3_0.6b_rocm_conservative/final_model")
    model = AutoModelForCausalLM.from_pretrained(
        "./qwen3_0.6b_rocm_conservative/final_model",
        device_map="auto", 
        torch_dtype=torch.float16
    )
    
    prompt = "Count the S's in Mississippi"
    print(f"📝 Prompt: {prompt}")
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    print(f"Input tokens: {inputs.input_ids.shape[1]}")
    
    # Force longer generation
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_new_tokens=256,
            min_new_tokens=50,  # Force minimum length
            temperature=0.8,
            do_sample=True,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=None,  # Disable EOS stopping 
            early_stopping=False
        )
    
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    print(f"🤖 Raw response ({len(response)} chars):")
    print(repr(response))
    print("\n🤖 Formatted response:")
    print(response)

if __name__ == "__main__":
    direct_test()