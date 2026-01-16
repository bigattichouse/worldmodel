#!/usr/bin/env python3
"""Simple WASM WorldModel inference test."""

import sys
sys.path.append('src')

import torch
from src.models.qwen_wasm_adapter import QwenWASMAdapter
from src.tokenization.wat_tokenizer import WATTokenizer

def test_simple_inference():
    """Test simple single-step inference."""
    print("🚀 Simple WASM WorldModel Inference Test")
    print("=" * 60)
    
    # Load model
    print("Loading model...")
    wasm_adapter = QwenWASMAdapter(
        model_path="/home/bigattichouse/workspace/model/Qwen3-0.6B",
        cross_modal_layers=[3, 7, 11],
        use_sandbox=False
    )
    
    # Load weights
    state_dict = torch.load("./wasm_worldmodel_final_trained/pytorch_model.bin", 
                           map_location="cpu", weights_only=False)
    wasm_adapter.load_state_dict(state_dict, strict=False)
    wasm_adapter.eval()
    
    # Set WASM tokenizer
    wasm_tokenizer = WATTokenizer(vocab_size=8000)
    wasm_adapter.set_wasm_tokenizer(wasm_tokenizer)
    
    print("✅ Model loaded successfully!")
    
    # Test cases
    test_cases = [
        "Calculate 15 + 27",
        "What is 8 × 9?",
        "Calculate 144 ÷ 12"
    ]
    
    for test_case in test_cases:
        print(f"\n📝 Input: {test_case}")
        
        # Prepare inputs
        inputs = wasm_adapter.text_tokenizer(
            f"User: {test_case}\nAssistant:",
            return_tensors="pt",
            max_length=50,
            truncation=True
        )
        
        # Forward pass with WASM execution
        with torch.no_grad():
            try:
                outputs = wasm_adapter(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    execute_wasm=True
                )
                
                print("✅ Forward pass successful!")
                print(f"   Output shape: {outputs['logits'].shape}")
                
                # Check if WASM was executed
                if 'execution_results' in outputs and outputs['execution_results']:
                    print(f"   WASM executions: {len(outputs['execution_results'])}")
                    for i, result in enumerate(outputs['execution_results']):
                        if result and result.get('success'):
                            computed = result.get('result', 'N/A')
                            print(f"   ✨ WASM result {i+1}: {computed}")
                        else:
                            error = result.get('error', 'Unknown') if result else 'No result'
                            print(f"   ❌ WASM execution {i+1} failed: {error}")
                else:
                    print("   🔧 No WASM execution results found")
                
            except Exception as e:
                print(f"❌ Forward pass failed: {e}")
                import traceback
                traceback.print_exc()

if __name__ == "__main__":
    test_simple_inference()