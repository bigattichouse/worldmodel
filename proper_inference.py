#!/usr/bin/env python3
"""
Proper Inference Script for ByteLogic WorldModel After Training
===============================================================

Correctly loads the trained model for inference after training is complete.
"""

import torch
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from models.qwen_wasm_adapter import QwenWASMAdapter
from execution.bytelogic_executor import ByteLogicExecutor
from execution.computation_processor import ComputationTokenProcessor
import json

def load_fully_trained_model():
    """Load the model with trained weights applied properly."""
    print("Loading fully trained ByteLogic WorldModel...")
    
    # Configuration should match the training configuration
    computation_layers = [3, 7, 11, 15, 19, 23, 27]
    
    # First, create a fresh model with the same architecture as used during training
    base_model_path = "~/workspace/model/Qwen3-0.6B"
    base_model_path = base_model_path.replace("~", str(Path.home()))
    
    print(f"Loading base model from: {base_model_path}")
    
    try:
        # Create the adapter with the same parameters used during training
        model = QwenWASMAdapter(
            model_path=base_model_path,
            cross_modal_layers=computation_layers,
            freeze_text_layers=False  # During inference, we may want to keep this flexible
        )
        
        print(f"Base model initialized with {sum(p.numel() for p in model.parameters()):,} parameters")
        
        # Load the checkpoint file
        checkpoint_path = "integrated_worldmodel_output/final_integrated_model.pt"
        print(f"Loading trained weights from: {checkpoint_path}")
        
        # Load the checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Determine if it's a full checkpoint or just state dict
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            # This is a training checkpoint with metadata
            state_dict = checkpoint['model_state_dict']
            print("Loaded training checkpoint with metadata")
            print(f"Epoch: {checkpoint.get('epoch', 'N/A')}")
            print(f"Best validation loss: {checkpoint.get('metrics', {}).get('val_loss', 'N/A')}")
        else:
            # This is a direct state dict (which seems to be the case based on training)
            state_dict = checkpoint
            print("Loaded direct state dict checkpoint")
        
        # Move model to device first to make sure all components are on the same device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

        # Load the trained weights
        model_state_dict = model.state_dict()

        # Check for key mismatches
        checkpoint_keys = set(state_dict.keys())
        model_keys = set(model_state_dict.keys())

        missing_in_checkpoint = model_keys - checkpoint_keys
        unexpected_in_checkpoint = checkpoint_keys - model_keys

        print(f"Keys in model: {len(model_keys)}")
        print(f"Keys in checkpoint: {len(checkpoint_keys)}")
        print(f"Missing in checkpoint: {len(missing_in_checkpoint)}")
        print(f"Unexpected in checkpoint: {len(unexpected_in_checkpoint)}")

        if missing_in_checkpoint:
            print("  Missing keys:", list(missing_in_checkpoint)[:10])  # Show first 10
        if unexpected_in_checkpoint:
            print("  Unexpected keys:", list(unexpected_in_checkpoint)[:10])  # Show first 10

        # Load with strict=False to handle potential mismatches
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

        if missing_keys:
            print(f"Missing keys after load: {len(missing_keys)}")
            for key in missing_keys[:5]:  # Show first 5
                print(f"  - {key}")
        if unexpected_keys:
            print(f"Unexpected keys after load: {len(unexpected_keys)}")
            for key in unexpected_keys[:5]:  # Show first 5
                print(f"  - {key}")

        model.eval()  # Set to evaluation mode
        
        print(f"✅ Model loaded successfully and moved to {device}")
        print(f"   Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"   Text parameters: {sum(p.numel() for p in model.text_model.parameters()):,}")
        print(f"   WASM parameters: {sum(p.numel() for p in model.wasm_layers.parameters()):,}")
        
        return model, device
        
    except Exception as e:
        print(f"❌ Error loading fully trained model: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def test_computation_generation(model, device, test_input):
    """Test if the model can properly generate computation code now."""
    print(f"\\nTesting computation generation: '{test_input}'")
    print("-" * 60)
    
    try:
        # Tokenize input
        inputs = model.text_tokenizer(test_input, return_tensors="pt", padding=True, truncation=True)
        
        # Move to device
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)
        
        print(f"Input shape: {input_ids.shape}")
        
        # Forward pass with computation enabled
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                execute_wasm=True,
                return_dict=True
            )
        
        # Analyze the results
        execution_results = outputs.get('execution_results', [])
        print(f"Number of execution candidates generated: {len(execution_results)}")
        
        for i, result in enumerate(execution_results):
            layer = result.get('layer', 'N/A')
            executed = result.get('executed', False)
            success = result.get('success', None)
            code_key = 'bytelogic_code' if 'bytelogic_code' in result else 'wat_code'
            code = result.get(code_key, '')
            error = result.get('error', '')
            score = result.get('score', 'N/A')
            
            print(f"\\nCandidate {i+1} (Layer {layer}):")
            print(f"  - Executed: {executed}")
            print(f"  - Success: {success}")
            print(f"  - Score: {score}")
            if code:
                print(f"  - Code type: {code_key}")
                print(f"  - Code preview: {code[:150]}...")
            if error:
                print(f"  - Error: {error}")
        
        # Try to generate some tokens
        print("\\nGenerating output sequence...")
        result = model.generate_with_wasm(
            input_text=test_input,
            max_length=150,
            temperature=0.1,
            execute_wasm=True
        )
        
        print(f"Input: {result['input']}")
        print(f"Generated: {result['output']}")
        print(f"WASM executed: {result['wasm_executed']}")
        print(f"Computation tokens processed: {result['computation_tokens_processed']}")
        
        return result
        
    except Exception as e:
        print(f"❌ Error in computation generation test: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    print("Proper Inference Script for Trained ByteLogic WorldModel")
    print("="*80)
    print("Loading the model with properly trained weights...")
    
    # Load the trained model
    model, device = load_fully_trained_model()
    
    if model is None:
        print("\\n❌ Failed to load model. Exiting.")
        return
    
    print(f"\\nModel loaded successfully on {device}")
    
    # Test cases that should ideally trigger computation generation
    test_cases = [
        "What is 12 * 15?",
        "Calculate 24 divided by 6", 
        "If Alice is parent of Bob, who is the child?",
        "What is 5 + 7?",
        "How many children does Alice have if Alice is parent of Bob and Charlie?"
    ]
    
    print(f"\\nRunning {len(test_cases)} test cases...")
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\\n{'='*20} TEST CASE {i} {'='*20}")
        test_computation_generation(model, device, test_case)
    
    print("\\n" + "="*80)
    print("INFERENCE TEST COMPLETE")
    print("="*80)
    print("If the model is working correctly after our fixes:")
    print("- It should generate ByteLogic code instead of WASM code")
    print("- The execution results should show higher success rates") 
    print("- Computation tokens should appear in the output")
    print("- The ByteLogic executor should be able to process the generated code")


if __name__ == "__main__":
    main()