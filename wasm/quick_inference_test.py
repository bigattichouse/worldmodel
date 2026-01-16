#!/usr/bin/env python3
"""Quick test to see generated WAT code."""

import sys
sys.path.append('src')

import torch
from src.models.qwen_wasm_adapter import QwenWASMAdapter

def test_wat_generation():
    """Test a single forward pass to see WAT generation."""
    print("🔍 Testing WAT generation...")
    
    # Initialize model
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
    
    # Simple test
    text = "Calculate 15 + 27"
    inputs = wasm_adapter.text_tokenizer(text, return_tensors="pt")
    
    print(f"Input: {text}")
    
    # Single forward pass
    with torch.no_grad():
        outputs = wasm_adapter(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            execute_wasm=True  # Enable WASM execution
        )
    
    print("Forward pass completed!")

if __name__ == "__main__":
    test_wat_generation()