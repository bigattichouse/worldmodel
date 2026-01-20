#!/usr/bin/env python3
"""
Simple ROCm Test without PEFT dependencies
==========================================

Tests core functionality for WorldModel training:
1. ROCm GPU detection
2. Model loading 
3. Data processing
4. Basic forward pass and loss calculation
"""

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
import time

def test_rocm_setup():
    """Test ROCm GPU setup."""
    print("=== ROCm Setup Test ===")
    
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"Device count: {torch.cuda.device_count()}")
        print(f"Current device: {torch.cuda.current_device()}")
        try:
            device_name = torch.cuda.get_device_name()
            print(f"Device name: {device_name}")
            
            # Test basic operations
            x = torch.randn(100, 100, device='cuda')
            y = torch.matmul(x, x)
            print(f"✅ Basic GPU operations working")
            
            # Test memory
            total_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"Total GPU memory: {total_mem:.1f}GB")
            
            torch.cuda.empty_cache()
            return True
        except Exception as e:
            print(f"⚠️  GPU issues: {e}")
            return False
    else:
        print("❌ No CUDA/ROCm device detected")
        return False

def test_model_loading():
    """Test model loading with ROCm-compatible settings."""
    print("\n=== Model Loading Test ===")
    
    # Try different models in order of preference
    model_candidates = [
        "/home/bigattichouse/workspace/model/Qwen3-0.6B",
        "/home/bigattichouse/workspace/model/gemma-3-270M-it", 
        "/home/bigattichouse/workspace/model/Qwen2.5-3B-Instruct"
    ]
    
    for model_path in model_candidates:
        print(f"Trying model: {model_path}")
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
            
            print(f"✅ Model loaded successfully: {model_path}")
            print(f"   Vocab size: {len(tokenizer)}")
            print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
            
            return model, tokenizer, model_path
            
        except Exception as e:
            print(f"❌ Failed to load {model_path}: {e}")
            continue
    
    print("❌ No models could be loaded")
    return None, None, None

def test_data_processing(tokenizer, model_path):
    """Test data processing pipeline."""
    print("\n=== Data Processing Test ===")
    
    # Sample WorldModel data
    sample_data = [
        {
            "user": "Calculate 15% of 200",
            "assistant": "<think>I need to calculate 15% of 200. 15% = 0.15, so 0.15 × 200 = 30</think>\n<model>\nresult = 0.15 * 200\nprint(f\"15% of 200 = {result}\")\n</model>\n<requires>python:math</requires>\n\n15% of 200 equals 30."
        }
    ]
    
    try:
        print("Testing data formatting...")
        
        sample = sample_data[0]
        
        # Format as conversation - detect model type
        if "qwen" in model_path.lower():
            # Qwen format
            formatted = f"<|im_start|>user\n{sample['user']}<|im_end|>\n<|im_start|>assistant\n{sample['assistant']}<|im_end|>"
        elif "gemma" in model_path.lower():
            # Gemma format
            formatted = f"<start_of_turn>user\n{sample['user']}<end_of_turn>\n<start_of_turn>model\n{sample['assistant']}<end_of_turn>"
        else:
            # Generic format
            formatted = f"### Human: {sample['user']}\n\n### Assistant: {sample['assistant']}"
        
        print(f"Formatted text (first 200 chars): {formatted[:200]}...")
        
        # Tokenize
        encoding = tokenizer(
            formatted,
            truncation=True,
            max_length=512,
            padding="max_length",
            return_tensors="pt"
        )
        
        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()
        
        # Count non-padding tokens
        non_pad_tokens = attention_mask.sum().item()
        
        print(f"✅ Data processing working")
        print(f"   Total tokens: {len(input_ids)}")
        print(f"   Non-padding tokens: {non_pad_tokens}")
        print(f"   Contains WorldModel tags: {any(tag in formatted for tag in ['<think>', '<model>', '<requires>'])}")
        
        return encoding, formatted
        
    except Exception as e:
        print(f"❌ Data processing failed: {e}")
        return None, None

def test_simple_training_step(model, tokenizer, encoding, formatted_text, model_path):
    """Test a simple training step to verify loss calculation."""
    print("\n=== Simple Training Step Test ===")
    
    if model is None or tokenizer is None or encoding is None:
        print("❌ No model/data available for training test")
        return False
    
    try:
        model.train()
        
        # Simple optimizer for just this test
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        
        input_ids = encoding['input_ids']
        attention_mask = encoding['attention_mask']
        
        # Move to GPU if available
        if torch.cuda.is_available():
            input_ids = input_ids.cuda()
            attention_mask = attention_mask.cuda()
        
        # Create labels (mask user part)
        labels = input_ids.clone()
        
        # Find where assistant response starts
        if "qwen" in model_path.lower():
            user_end_marker = "<|im_start|>assistant\n"
        elif "gemma" in model_path.lower():
            user_end_marker = "<start_of_turn>model\n"
        else:
            user_end_marker = "### Assistant:"
        
        # Find marker in the formatted text and tokenize up to that point
        marker_pos = formatted_text.find(user_end_marker)
        if marker_pos > 0:
            user_part = formatted_text[:marker_pos + len(user_end_marker)]
            user_tokens = tokenizer.encode(user_part, add_special_tokens=False, truncation=True, max_length=512)
            user_len = min(len(user_tokens), labels.shape[1])
            if user_len > 0:
                labels[0, :user_len] = -100
        
        print(f"Input shape: {input_ids.shape}")
        print(f"Labels shape: {labels.shape}")
        print(f"User tokens masked: {user_len if 'user_len' in locals() else 'unknown'}")
        
        # Training steps
        losses = []
        for step in range(3):
            optimizer.zero_grad()
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            loss_value = loss.item()
            losses.append(loss_value)
            
            print(f"Step {step}: Loss = {loss_value:.6f}")
            
            if loss_value > 0:  # Only backprop if loss is meaningful
                loss.backward()
                optimizer.step()
            
        # Check results
        if all(l == 0 for l in losses):
            print("❌ Zero loss detected - labels might be all -100")
            print("   This suggests masking issue or data format problem")
            return False
        elif losses[0] > 0:
            print("✅ Training step working - loss is non-zero")
            if len(losses) > 1 and losses[-1] < losses[0]:
                improvement = ((losses[0] - losses[-1]) / losses[0] * 100)
                print(f"   Loss improved by {improvement:.1f}%")
            return True
        else:
            print(f"⚠️  Unusual loss pattern: {losses}")
            return False
    
    except Exception as e:
        print(f"❌ Training step failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("🧪 Simple ROCm WorldModel Test Suite")
    print("=" * 50)
    
    # Test 1: ROCm setup
    rocm_ok = test_rocm_setup()
    
    # Test 2: Model loading  
    model, tokenizer, model_path = test_model_loading()
    model_ok = model is not None
    
    # Test 3: Data processing
    encoding, formatted_text = test_data_processing(tokenizer, model_path) if tokenizer else (None, None)
    data_ok = encoding is not None
    
    # Test 4: Simple training step
    training_ok = test_simple_training_step(model, tokenizer, encoding, formatted_text, model_path)
    
    # Summary
    print("\n" + "=" * 50)
    print("🎯 Test Results Summary")
    print("=" * 50)
    print(f"ROCm Setup: {'✅' if rocm_ok else '❌'}")
    print(f"Model Loading: {'✅' if model_ok else '❌'}")
    print(f"Data Processing: {'✅' if data_ok else '❌'}")  
    print(f"Training Step: {'✅' if training_ok else '❌'}")
    
    if all([rocm_ok, model_ok, data_ok, training_ok]):
        print("\n🎉 ALL TESTS PASSED!")
        print("✨ Core training functionality is working")
        print("📝 Ready to implement full training pipeline")
    else:
        print("\n⚠️  Some tests failed - check issues above")
        if not rocm_ok:
            print("   • ROCm/GPU not working properly")
        if not model_ok:
            print("   • Model loading failed")
        if not data_ok:
            print("   • Data processing issues")
        if not training_ok:
            print("   • Training step problems (likely zero loss)")
    
    print("=" * 50)

if __name__ == "__main__":
    main()