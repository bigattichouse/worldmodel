#!/usr/bin/env python3
"""
Test ROCm PyTorch functionality based on research findings
from ../rocm documentation about gfx906 compatibility issues
"""
import sys
sys.path.insert(0, 'src')

def test_basic_rocm_pytorch():
    """Test basic ROCm PyTorch functionality before attempting training"""
    print("🧪 Testing ROCm + PyTorch Compatibility")
    print("="*60)
    
    try:
        import torch
        print(f"✅ PyTorch imported: {torch.__version__}")
        
        # Test 1: Basic CUDA availability
        print(f"\n🔍 GPU Detection:")
        print(f"   CUDA Available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"   Device Count: {torch.cuda.device_count()}")
            print(f"   Device Name: {torch.cuda.get_device_name(0)}")
            print(f"   Capability: {torch.cuda.get_device_capability(0)}")
            print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
            
            # Test 2: Simple tensor operations (these should work per research)
            print(f"\n🧮 Basic Tensor Operations:")
            try:
                device = 'cuda'
                x = torch.randn(1000, 1000, device=device)
                y = torch.randn(1000, 1000, device=device)
                z = torch.matmul(x, y)
                print(f"   ✅ Matrix multiplication: {z.shape} on {z.device}")
                
                # Test memory cleanup
                del x, y, z
                torch.cuda.empty_cache()
                print(f"   ✅ Memory cleanup successful")
                
            except Exception as e:
                print(f"   ❌ Basic operations failed: {e}")
                return False
                
            # Test 3: Simple model loading (this might fail per research)
            print(f"\n🤖 Model Loading Test:")
            try:
                # Very simple test - just load tokenizer first
                from transformers import AutoTokenizer
                tokenizer = AutoTokenizer.from_pretrained('../model/Qwen2.5-3B-Instruct')
                print(f"   ✅ Tokenizer loaded successfully")
                
                # Now try basic model loading
                from transformers import AutoModelForCausalLM
                print(f"   📥 Loading Qwen2.5-3B model...")
                model = AutoModelForCausalLM.from_pretrained(
                    '../model/Qwen2.5-3B-Instruct',
                    torch_dtype=torch.float16,  # Start with float16, not bfloat16
                    device_map="cpu"  # Load to CPU first
                )
                print(f"   ✅ Model loaded to CPU successfully")
                
                # Try moving to GPU (this is where segfaults might happen)
                print(f"   🚀 Moving model to GPU...")
                model = model.to('cuda')
                print(f"   ✅ Model moved to GPU successfully")
                
                # Simple inference test
                inputs = tokenizer("Hello world", return_tensors='pt').to('cuda')
                with torch.no_grad():
                    outputs = model.generate(**inputs, max_new_tokens=5)
                response = tokenizer.decode(outputs[0], skip_special_tokens=True)
                print(f"   ✅ Simple inference successful: '{response}'")
                
                # Cleanup
                del model, inputs, outputs
                torch.cuda.empty_cache()
                
            except Exception as e:
                print(f"   ❌ Model/inference test failed: {e}")
                print(f"   🔍 This matches documented gfx906 issues")
                return False
                
            return True
        else:
            print(f"   ❌ No GPU detected")
            return False
            
    except Exception as e:
        print(f"❌ Failed to import PyTorch: {e}")
        return False

if __name__ == "__main__":
    # Set ROCm environment
    import os
    os.environ['HSA_OVERRIDE_GFX_VERSION'] = '10.3.0'
    os.environ['PYTORCH_ROCM_ARCH'] = 'gfx906'
    
    success = test_basic_rocm_pytorch()
    
    if success:
        print(f"\n🎉 ROCm PyTorch test passed! Training should work.")
    else:
        print(f"\n⚠️ ROCm PyTorch issues detected.")
        print(f"📋 Per research findings:")
        print(f"   - PyTorch + transformers + gfx906 has known segfault issues")
        print(f"   - CPU training is recommended fallback")
        print(f"   - Consider ROCm 5.7.3 downgrade for better compatibility")