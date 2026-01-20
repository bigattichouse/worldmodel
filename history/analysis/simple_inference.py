#!/usr/bin/env python3
"""
Simple Inference Script for ByteLogic WorldModel
===============================================

Test the trained model to see how function calls are performing.
"""

import torch
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from models.qwen_wasm_adapter import QwenWASMAdapter
import logging
import json

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_model_inference():
    """Test the trained model with sample inputs."""
    
    # Load the trained model
    model_path = "integrated_worldmodel_output/final_integrated_model.pt"
    
    if not Path(model_path).exists():
        print(f"Model not found at {model_path}")
        print("Looking for available models in integrated_worldmodel_output/")
        
        output_dir = Path("integrated_worldmodel_output/")
        if output_dir.exists():
            model_files = list(output_dir.glob("*.pt"))
            print(f"Found model files: {model_files}")
            
            # Look for the most recent checkpoint
            checkpoint_files = list(output_dir.glob("checkpoint_*.pt"))
            if checkpoint_files:
                print(f"Available checkpoints: {checkpoint_files}")
                model_path = str(sorted(checkpoint_files)[-1])  # Use the most recent
                print(f"Using checkpoint: {model_path}")
            else:
                print("No checkpoints found!")
                return
        else:
            print("Output directory not found!")
            return
    
    print(f"Loading model from: {model_path}")
    
    # Initialize model
    try:
        # Note: We need to initialize differently since we saved the state dict directly
        base_model_path = "~/workspace/model/Qwen3-0.6B"  # You might need to adjust this path
        
        model = QwenWASMAdapter(
            model_path=base_model_path.replace("~", str(Path.home())),
            cross_modal_layers=[3, 7, 11, 15, 19, 23, 27],
            freeze_text_layers=False
        )
        
        # Load the trained weights
        trained_state = torch.load(model_path, map_location='cpu')
        model.load_state_dict(trained_state, strict=False)  # Use strict=False to allow partial loading
        
        model.eval()
        print("✅ Model loaded successfully")
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("Trying alternative loading method...")
        return

    # Test cases focusing on computation/function calling
    test_cases = [
        "What is 12 * 15?",
        "Calculate 24 divided by 6",
        "What is 7 + 8?",
        "What is 100 - 25?",
        "What is 5 squared?",
        "What is 2 to the power of 8?",
        "Who is the parent of Bob in this family tree: Alice is parent of Bob, Bob is parent of Charlie?",
        "If John loves Mary and Mary loves Tom, who does John love?",
        "What are the ancestors of Charlie if Alice is parent of Bob and Bob is parent of Charlie?"
    ]
    
    print("\n" + "="*60)
    print("TESTING MODEL INFERENCE")
    print("="*60)
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\nTest {i}: {test_case}")
        print("-" * 40)
        
        try:
            # Use the model's generate_with_wasm method
            result = model.generate_with_wasm(
                input_text=test_case,
                max_length=200,
                temperature=0.1,  # Low temperature for more consistent results
                execute_wasm=True
            )
            
            print(f"Input: {result['input']}")
            print(f"Output: {result['output']}")
            print(f"Raw Output: {result['raw_output']}")
            print(f"WASM Executed: {result['wasm_executed']}")
            print(f"Computation Tokens Processed: {result['computation_tokens_processed']}")
            
            # Print execution results if any
            execution_results = result['execution_results']
            if execution_results:
                print(f"Execution Results ({len(execution_results)}):")
                for j, exec_res in enumerate(execution_results):
                    print(f"  Result {j+1}: Layer {exec_res.get('layer', 'N/A')}")
                    print(f"    Success: {exec_res.get('success', 'N/A')}")
                    print(f"    Executed: {exec_res.get('executed', 'N/A')}")
                    if exec_res.get('wat_code'):
                        print(f"    WAT Code: {exec_res['wat_code'][:100]}...")
                    if exec_res.get('error'):
                        print(f"    Error: {exec_res['error']}")
                    print()
            
        except Exception as e:
            print(f"Error processing test case {i}: {e}")
            import traceback
            traceback.print_exc()


def manual_model_load_and_test():
    """Manual approach to load and test the model if the automatic way doesn't work."""
    print("\nAttempting manual model loading...")
    
    # Try to load the state dict to inspect its contents
    try:
        checkpoint_path = "integrated_worldmodel_output/final_integrated_model.pt"
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        print(f"Checkpoint keys: {list(checkpoint.keys()) if isinstance(checkpoint, dict) else 'State dict'}")
        
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            # This is a training checkpoint format
            state_dict = checkpoint['model_state_dict']
            print(f"Model state dict keys: {len(list(state_dict.keys()))} keys")
        else:
            # This is a direct state dict
            state_dict = checkpoint
            print(f"Direct state dict keys: {len(list(state_dict.keys()))} keys")
        
        # Show some key names to understand the structure
        keys = list(state_dict.keys())
        print(f"Sample keys: {keys[:10]}")
        
    except Exception as e:
        print(f"Could not load checkpoint: {e}")
        return False
    
    print("Manual loading approach completed.")
    return True


def create_simple_test_interface():
    """Create a simple interactive test interface."""
    print("\n" + "="*60)
    print("SIMPLE TEST INTERFACE")
    print("="*60)
    print("This is a template for testing the model. Due to loading complexities,")
    print("we'll first try to see what model files are available.\n")
    
    # List available models
    output_dir = Path("integrated_worldmodel_output/")
    if output_dir.exists():
        print("Available model files:")
        for file in output_dir.iterdir():
            if file.suffix == '.pt':
                size_mb = file.stat().st_size / (1024 * 1024)
                print(f"  {file.name} ({size_mb:.2f} MB)")
    
    print("\nTo use this properly, you would need to:")
    print("1. Identify the correct base model path for loading")
    print("2. Load the trained weights correctly")
    print("3. Use the inference methods to test computation tokens")


if __name__ == "__main__":
    print("Simple Inference Script for ByteLogic WorldModel")
    print("="*60)
    
    # First, check what model files are available
    print("Checking available model files...")
    create_simple_test_interface()
    
    print("\n" + "-"*60)
    
    # Try the inference tests
    test_model_inference()
    
    print("\n" + "-"*60)
    
    # Try manual loading
    manual_model_load_and_test()
    
    print("\nInference script completed.")