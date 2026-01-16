#!/usr/bin/env python3
"""
WASM WorldModel Inference
========================

Load and run inference with the trained WASM WorldModel.
Tests the model's ability to perform real computation during reasoning.
"""

import sys
import os
sys.path.append('src')

import torch
from src.models.qwen_wasm_adapter import QwenWASMAdapter
from src.tokenization.wat_tokenizer import WATTokenizer
import pickle

def load_trained_model(checkpoint_path: str):
    """Load the trained WASM WorldModel."""
    print("🔄 Loading trained WASM WorldModel...")
    
    # Skip training args due to PyTorch security restrictions
    # We know the base model path from our training setup
    
    # Initialize the WASM adapter (same config as training)
    base_model_path = "/home/bigattichouse/workspace/model/Qwen3-0.6B"
    wasm_adapter = QwenWASMAdapter(
        model_path=base_model_path,
        cross_modal_layers=[3, 7, 11],
        freeze_text_layers=False,
        use_sandbox=False  # Direct execution for inference
    )
    
    # Load the trained weights with security bypass for our trusted model
    model_state_path = os.path.join(checkpoint_path, "pytorch_model.bin")
    try:
        state_dict = torch.load(model_state_path, map_location="cpu", weights_only=True)
    except Exception as e:
        print(f"Warning: weights_only failed ({e}), using trusted load...")
        state_dict = torch.load(model_state_path, map_location="cpu", weights_only=False)
    
    # Load state dict with missing key handling
    missing_keys, unexpected_keys = wasm_adapter.load_state_dict(state_dict, strict=False)
    
    if missing_keys:
        print(f"   Missing keys (will use random init): {len(missing_keys)}")
    if unexpected_keys:
        print(f"   Unexpected keys (ignored): {len(unexpected_keys)}")
    
    # Load WASM tokenizer
    wasm_tokenizer = WATTokenizer(vocab_size=8000)
    wasm_adapter.set_wasm_tokenizer(wasm_tokenizer)
    
    # Set to eval mode
    wasm_adapter.eval()
    
    print("✅ Model loaded successfully!")
    return wasm_adapter

def test_computation_reasoning(model, test_cases):
    """Test the model's computation-as-reasoning capabilities."""
    print("\n🧪 Testing Computation-as-Reasoning")
    print("=" * 60)
    
    for i, test_case in enumerate(test_cases, 1):
        question = test_case["question"]
        expected = test_case["expected"]
        
        print(f"\n{i}. Question: {question}")
        print(f"   Expected: {expected}")
        
        try:
            # Run inference
            result = model.generate_with_wasm(
                input_text=f"User: {question}\nAssistant:",
                max_length=50,
                temperature=0.1,  # Low temperature for consistent results
                execute_wasm=True
            )
            
            output = result["output"]
            wasm_executed = result["wasm_executed"]
            execution_results = result["execution_results"]
            
            print(f"   Generated: {output}")
            print(f"   WASM executed: {wasm_executed}")
            
            if execution_results:
                for j, exec_result in enumerate(execution_results):
                    if exec_result["success"]:
                        computed = exec_result["result"]
                        print(f"   Computed result {j+1}: {computed}")
                    else:
                        print(f"   Execution {j+1} failed: {exec_result.get('error', 'unknown')}")
            
            print("   " + "✅ SUCCESS" if wasm_executed else "❌ NO WASM EXECUTION")
            
        except Exception as e:
            print(f"   ❌ Error: {e}")

def main():
    """Run WASM WorldModel inference demo."""
    print("🚀 WASM WorldModel Inference Demo")
    print("=" * 70)
    
    # Load the trained model
    checkpoint_path = "./wasm_worldmodel_final_trained"
    
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        print("   Make sure you have the trained model checkpoint")
        return
    
    try:
        model = load_trained_model(checkpoint_path)
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return
    
    # Test cases for computation reasoning
    test_cases = [
        {"question": "Calculate 15 + 27", "expected": 42},
        {"question": "What is 8 × 9?", "expected": 72},
        {"question": "Calculate 144 ÷ 12", "expected": 12},
        {"question": "What is 25 - 7?", "expected": 18},
        {"question": "Calculate 13 + 19", "expected": 32},
    ]
    
    # Run inference tests
    test_computation_reasoning(model, test_cases)
    
    # Interactive mode
    print(f"\n" + "=" * 70)
    print("🎮 Interactive Mode (Ctrl+C to exit)")
    print("   Ask mathematical questions to test WASM computation!")
    print("=" * 70)
    
    try:
        while True:
            question = input("\nYour question: ").strip()
            if not question:
                continue
                
            try:
                result = model.generate_with_wasm(
                    input_text=f"User: {question}\nAssistant:",
                    max_length=100,
                    temperature=0.2,
                    execute_wasm=True
                )
                
                print(f"Model: {result['output']}")
                
                if result["wasm_executed"] and result["execution_results"]:
                    for exec_result in result["execution_results"]:
                        if exec_result["success"]:
                            print(f"🔧 WASM computed: {exec_result['result']}")
                        else:
                            print(f"🔧 WASM failed: {exec_result.get('error', 'unknown')}")
                else:
                    print("🔧 No WASM execution")
                    
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error: {e}")
    
    except KeyboardInterrupt:
        pass
    
    print("\n👋 Inference demo completed!")

if __name__ == "__main__":
    main()