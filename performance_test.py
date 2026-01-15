#!/usr/bin/env python3
"""
Quick Performance Test for ROCm Optimizations
============================================

Test script to verify optimizations before full training.
"""

import torch
import time
from transformers import AutoTokenizer, AutoModelForCausalLM

def test_precision():
    """Test 1: Verify FP16 vs FP32 performance."""
    print("🧪 Test 1: Precision Performance")
    print("-" * 40)
    
    model_path = "/home/bigattichouse/workspace/model/Qwen3-0.6B"
    
    # Test FP32
    print("Testing FP32...")
    model_fp32 = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.float32,
        device_map={"": 0}
    )
    
    inputs_fp32 = torch.randint(0, 1000, (4, 512)).cuda()
    
    start_time = time.time()
    for _ in range(10):
        with torch.no_grad():
            outputs = model_fp32(inputs_fp32)
    torch.cuda.synchronize()
    fp32_time = time.time() - start_time
    
    print(f"FP32: {fp32_time:.3f}s for 10 forward passes")
    print(f"Model dtype: {next(model_fp32.parameters()).dtype}")
    
    del model_fp32
    torch.cuda.empty_cache()
    
    # Test FP16
    print("Testing FP16...")
    model_fp16 = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map={"": 0}
    )
    
    inputs_fp16 = torch.randint(0, 1000, (4, 512)).cuda()
    
    start_time = time.time()
    for _ in range(10):
        with torch.no_grad():
            outputs = model_fp16(inputs_fp16)
    torch.cuda.synchronize()
    fp16_time = time.time() - start_time
    
    print(f"FP16: {fp16_time:.3f}s for 10 forward passes")
    print(f"Model dtype: {next(model_fp16.parameters()).dtype}")
    
    speedup = fp32_time / fp16_time
    print(f"🚀 FP16 Speedup: {speedup:.2f}x")
    
    del model_fp16
    torch.cuda.empty_cache()

def test_attention_backends():
    """Test 2: Compare attention implementations."""
    print("\n🧪 Test 2: Attention Backend Performance")
    print("-" * 40)
    
    model_path = "/home/bigattichouse/workspace/model/Qwen3-0.6B"
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map={"": 0}
    )
    
    inputs = torch.randint(0, 1000, (4, 512)).cuda()
    
    # Test with FlashAttention disabled (recommended for ROCm)
    print("Testing with FlashAttention disabled (ROCm-safe)...")
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    
    start_time = time.time()
    for _ in range(10):
        with torch.no_grad():
            outputs = model(inputs)
    torch.cuda.synchronize()
    safe_time = time.time() - start_time
    
    print(f"ROCm-safe attention: {safe_time:.3f}s for 10 forward passes")
    
    del model
    torch.cuda.empty_cache()

def test_batch_sizes():
    """Test 3: Optimal batch size for 32GB VRAM."""
    print("\n🧪 Test 3: Batch Size Performance")
    print("-" * 40)
    
    model_path = "/home/bigattichouse/workspace/model/Qwen3-0.6B"
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map={"": 0}
    )
    
    batch_sizes = [1, 2, 4, 6, 8]
    seq_length = 400  # Optimized length
    
    for batch_size in batch_sizes:
        try:
            torch.cuda.empty_cache()
            
            inputs = torch.randint(0, 1000, (batch_size, seq_length)).cuda()
            
            start_time = time.time()
            with torch.no_grad():
                outputs = model(inputs)
            torch.cuda.synchronize()
            forward_time = time.time() - start_time
            
            memory_gb = torch.cuda.max_memory_allocated() / 1e9
            tokens_per_sec = (batch_size * seq_length) / forward_time
            
            print(f"Batch {batch_size}: {forward_time:.3f}s, "
                  f"{tokens_per_sec:.0f} tok/s, "
                  f"{memory_gb:.1f}GB memory")
            
        except RuntimeError as e:
            print(f"Batch {batch_size}: OOM - {str(e)[:50]}...")
            break
    
    del model
    torch.cuda.empty_cache()

def test_memory_usage():
    """Test 4: Current memory availability."""
    print("\n🧪 Test 4: Memory Analysis")
    print("-" * 40)
    
    if torch.cuda.is_available():
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        allocated_memory = torch.cuda.memory_allocated() / 1e9
        reserved_memory = torch.cuda.memory_reserved() / 1e9
        available_memory = total_memory - reserved_memory
        
        print(f"Total VRAM: {total_memory:.1f}GB")
        print(f"Currently allocated: {allocated_memory:.1f}GB")
        print(f"Currently reserved: {reserved_memory:.1f}GB")
        print(f"Available: {available_memory:.1f}GB")
        
        print(f"\n💡 Recommendations:")
        if available_memory > 20:
            print(f"✅ Plenty of memory - can use batch_size=6-8")
        elif available_memory > 15:
            print(f"✅ Good memory - can use batch_size=4-6") 
        else:
            print(f"⚠️  Limited memory - stick with batch_size=2-4")

def main():
    """Run all performance tests."""
    print("🚀 WorldModel ROCm Performance Tests")
    print("=" * 50)
    
    # Set ROCm-safe attention backends
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    
    test_memory_usage()
    test_precision()
    test_attention_backends() 
    test_batch_sizes()
    
    print("\n✅ All tests completed!")
    print("\nTo train with optimizations:")
    print("python3 train_worldmodel_rocm_fast.py")

if __name__ == "__main__":
    main()