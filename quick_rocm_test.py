#!/usr/bin/env python3
"""
Quick ROCm Test for WorldModel Training
======================================

Tests key components before full training:
1. ROCm GPU detection and basic operations
2. Model loading with correct settings
3. Data processing pipeline
4. Small training loop to verify no zero loss
"""

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType
import time
import os

def test_rocm_setup():
    """Test ROCm GPU setup."""
    print("=== ROCm Setup Test ===")
    
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Device count: {torch.cuda.device_count()}")
        print(f"Current device: {torch.cuda.current_device()}")
        print(f"Device name: {torch.cuda.get_device_name()}")
        
        # Test basic operations
        x = torch.randn(100, 100, device='cuda')
        y = torch.matmul(x, x)
        print(f"✅ Basic GPU operations working")
        
        # Test memory
        total_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"Total GPU memory: {total_mem:.1f}GB")
        
        torch.cuda.empty_cache()
        return True
    else:
        print("❌ No CUDA/ROCm device detected")
        return False

def test_model_loading():
    """Test model loading with ROCm-compatible settings."""
    print("\n=== Model Loading Test ===")
    
    # Use smallest available model
    model_path = "/home/bigattichouse/workspace/model/Qwen3-0.6B"
    
    print(f"Loading model: {model_path}")
    
    try:
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load model with ROCm settings
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.float32,  # ROCm compatible
            device_map={"": 0} if torch.cuda.is_available() else None,
            low_cpu_mem_usage=True,
            use_cache=False,
            attn_implementation="eager"
        )
        
        print(f"✅ Model loaded successfully")
        print(f"   Vocab size: {len(tokenizer)}")
        print(f"   Model vocab size: {model.config.vocab_size}")
        print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        return model, tokenizer
        
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        return None, None

def test_data_processing():
    """Test data processing pipeline."""
    print("\n=== Data Processing Test ===")
    
    # Sample WorldModel data
    sample_data = [
        {
            "user": "Calculate 15% of 200",
            "assistant": "<think>I need to calculate 15% of 200. 15% = 0.15, so 0.15 × 200 = 30</think>\n<model>\nresult = 0.15 * 200\nprint(f\"15% of 200 = {result}\")\n</model>\n<requires>python:math</requires>\n\n15% of 200 equals 30."
        },
        {
            "user": "What is 7 × 8?",
            "assistant": "<think>I need to multiply 7 × 8 = 56</think>\n<model>\nresult = 7 * 8\nprint(f\"7 × 8 = {result}\")\n</model>\n<requires>python:math</requires>\n\n7 × 8 equals 56."
        }
    ]
    
    try:
        model_path = "/home/bigattichouse/workspace/model/Qwen3-0.6B"
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        print("Testing data formatting...")
        
        for i, sample in enumerate(sample_data):
            print(f"\n--- Sample {i+1} ---")
            
            # Format as conversation
            messages = [
                {"role": "user", "content": sample["user"]},
                {"role": "assistant", "content": sample["assistant"]}
            ]
            
            # Use chat template
            if hasattr(tokenizer, 'apply_chat_template'):
                formatted = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=False
                )
            else:
                formatted = f"<|im_start|>user\n{sample['user']}<|im_end|>\n<|im_start|>assistant\n{sample['assistant']}<|im_end|>"
            
            print(f"Formatted: {formatted[:200]}...")
            
            # Tokenize
            encoding = tokenizer(
                formatted,
                truncation=True,
                max_length=512,
                padding="max_length",
                return_tensors="pt"
            )
            
            input_ids = encoding['input_ids'].squeeze()
            labels = input_ids.clone()
            
            # Count non-padding tokens
            attention_mask = encoding['attention_mask'].squeeze()
            non_pad_tokens = attention_mask.sum().item()
            
            print(f"   Tokens: {non_pad_tokens} (total: {len(input_ids)})")
            print(f"   Contains WorldModel tags: {any(tag in formatted for tag in ['<think>', '<model>', '<requires>'])}")
        
        print("✅ Data processing working")
        return True
        
    except Exception as e:
        print(f"❌ Data processing failed: {e}")
        return False

def test_small_training_loop(model, tokenizer):
    """Test a small training loop to verify loss calculation."""
    print("\n=== Small Training Loop Test ===")
    
    if model is None or tokenizer is None:
        print("❌ No model available for training test")
        return False
    
    try:
        # Setup LoRA
        lora_config = LoraConfig(
            r=4,  # Very small for testing
            lora_alpha=8,
            lora_dropout=0.1,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        
        model = get_peft_model(model, lora_config)
        model.train()
        
        # Simple optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        
        # Test data
        test_text = "<|im_start|>user\nWhat is 2+2?<|im_end|>\n<|im_start|>assistant\n<think>I need to calculate 2+2 = 4</think>\n<model>\nresult = 2 + 2\nprint(result)\n</model>\n<requires>python:math</requires>\n\n2 + 2 equals 4.<|im_end|>"
        
        # Tokenize
        encoding = tokenizer(
            test_text,
            truncation=True,
            max_length=256,
            return_tensors="pt"
        )
        
        input_ids = encoding['input_ids']
        attention_mask = encoding['attention_mask']
        
        if torch.cuda.is_available():
            input_ids = input_ids.cuda()
            attention_mask = attention_mask.cuda()
        
        # Create labels (mask user part)
        labels = input_ids.clone()
        user_part = "<|im_start|>user\nWhat is 2+2?<|im_end|>\n<|im_start|>assistant\n"
        user_tokens = tokenizer.encode(user_part, add_special_tokens=False)
        if len(user_tokens) < labels.shape[1]:
            labels[0, :len(user_tokens)] = -100
        
        print(f"Input shape: {input_ids.shape}")
        print(f"User tokens to mask: {len(user_tokens)}")
        print(f"Total tokens: {input_ids.shape[1]}")
        
        # Training loop
        losses = []
        for step in range(5):
            optimizer.zero_grad()
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            losses.append(loss.item())
            
            print(f"Step {step}: Loss = {loss.item():.6f}")
            
            loss.backward()
            optimizer.step()
        
        # Check if loss is decreasing
        if losses[0] > 0 and losses[-1] < losses[0]:
            print("✅ Training loop working - loss is decreasing")
            print(f"   Initial loss: {losses[0]:.6f}")
            print(f"   Final loss: {losses[-1]:.6f}")
            print(f"   Improvement: {((losses[0] - losses[-1]) / losses[0] * 100):.1f}%")
            return True
        elif all(l == 0 for l in losses):
            print("❌ Zero loss detected - training not working")
            return False
        else:
            print(f"⚠️  Loss not decreasing properly")
            print(f"   Losses: {losses}")
            return False
    
    except Exception as e:
        print(f"❌ Training loop failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("🧪 ROCm WorldModel Quick Test Suite")
    print("=" * 50)
    
    # Test 1: ROCm setup
    rocm_ok = test_rocm_setup()
    
    # Test 2: Model loading  
    model, tokenizer = test_model_loading()
    model_ok = model is not None
    
    # Test 3: Data processing
    data_ok = test_data_processing()
    
    # Test 4: Training loop
    training_ok = test_small_training_loop(model, tokenizer)
    
    # Summary
    print("\n" + "=" * 50)
    print("🎯 Test Results Summary")
    print("=" * 50)
    print(f"ROCm Setup: {'✅' if rocm_ok else '❌'}")
    print(f"Model Loading: {'✅' if model_ok else '❌'}")
    print(f"Data Processing: {'✅' if data_ok else '❌'}")
    print(f"Training Loop: {'✅' if training_ok else '❌'}")
    
    if all([rocm_ok, model_ok, data_ok, training_ok]):
        print("\n🎉 ALL TESTS PASSED - Ready for full training!")
        print("Run: python train_rocm_fixed.py")
    else:
        print("\n⚠️  Some tests failed - check issues above")
    
    print("=" * 50)

if __name__ == "__main__":
    main()