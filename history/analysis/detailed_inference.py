#!/usr/bin/env python3
"""
Detailed Inference Script for ByteLogic WorldModel
==================================================

Test the trained model to see how function calls are performing with detailed token output.
"""

import torch
import sys
from pathlib import Path
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from models.qwen_wasm_adapter import QwenWASMAdapter
from tokenization.bytelogic_tokenizer import ByteLogicTokenizer
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def analyze_model_components():
    """Analyze the model checkpoint to understand its structure."""
    checkpoint_path = "integrated_worldmodel_output/final_integrated_model.pt"
    
    print(f"Analyzing checkpoint: {checkpoint_path}")
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        print(f"Checkpoint type: {type(checkpoint)}")
        
        if isinstance(checkpoint, dict):
            print(f"Checkpoint keys: {list(checkpoint.keys())}")
            
            # If it's a training checkpoint with metadata
            if 'model_state_dict' in checkpoint:
                print("This appears to be a training checkpoint with metadata")
                print(f"Epoch: {checkpoint.get('epoch', 'N/A')}")
                print(f"Metrics: {checkpoint.get('metrics', 'N/A')}")
                
                state_dict = checkpoint['model_state_dict']
            else:
                print("This appears to be a direct state dict")  
                state_dict = checkpoint
                
            print(f"State dict keys count: {len(state_dict.keys())}")
            
            # Show some sample keys to understand the structure
            keys = list(state_dict.keys())
            print(f"Sample keys (first 20):")
            for i, key in enumerate(keys[:20]):
                param = state_dict[key]
                print(f"  {i+1:2d}. {key}: {list(param.shape) if hasattr(param, 'shape') else 'scalar'}")
                
        else:
            # Direct state dict
            state_dict = checkpoint
            print(f"Direct state dict keys count: {len(state_dict.keys())}")
            
            keys = list(state_dict.keys())
            print(f"Sample keys (first 20):")
            for i, key in enumerate(keys[:20]):
                param = state_dict[key]
                print(f"  {i+1:2d}. {key}: {list(param.shape) if hasattr(param, 'shape') else 'scalar'}")
                
        return state_dict if isinstance(state_dict, dict) else checkpoint
            
    except Exception as e:
        print(f"Error analyzing checkpoint: {e}")
        import traceback
        traceback.print_exc()
        return None


def create_loaded_model_interface(state_dict):
    """Create a working model interface using the loaded state dict."""
    print("\\nCreating model interface...")
    
    # Initialize a fresh model with the same architecture as used during training
    base_model_path = "~/workspace/model/Qwen3-0.6B"
    base_model_path = base_model_path.replace("~", str(Path.home()))
    
    print(f"Loading base model: {base_model_path}")
    
    try:
        # Initialize the model with the same parameters used during training
        model = QwenWASMAdapter(
            model_path=base_model_path,
            cross_modal_layers=[3, 7, 11, 15, 19, 23, 27],  # Same as training
            freeze_text_layers=False  # Same as training
        )
        
        print(f"Base model initialized with {sum(p.numel() for p in model.parameters()):,} parameters")
        
        # Load the trained weights
        print("Loading trained weights...")
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        
        if missing_keys:
            print(f"Missing keys: {len(missing_keys)} - {missing_keys[:10]}...")  # Show first 10
        if unexpected_keys:
            print(f"Unexpected keys: {len(unexpected_keys)} - {unexpected_keys[:10]}...")  # Show first 10
            
        print("✅ Model loaded with trained weights")
        model.eval()  # Set to evaluation mode
        
        return model
        
    except Exception as e:
        print(f"Error creating model interface: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_computation_detailed(model, test_cases):
    """Test the model with detailed computation token analysis."""
    print("\\n" + "="*80)
    print("DETAILED COMPUTATION TOKEN ANALYSIS")
    print("="*80)
    
    # Detailed test cases for computation
    detailed_test_cases = [
        ("Simple math: 5 + 3", "What is 5 + 3?"),
        ("Multiplication: 6 * 7", "What is 6 multiplied by 7?"),
        ("Division: 15 / 3", "What is 15 divided by 3?"),
        ("Logic query: parent relationship", "If Alice is parent of Bob, who is the child of Alice?"),
        ("Complex math: 2^3 * 4", "What is 2 to the power of 3, then multiplied by 4?")
    ]
    
    for test_name, test_case in detailed_test_cases:
        print(f"\\n{test_name}: '{test_case}'")
        print("-" * 60)
        
        try:
            inputs = model.text_tokenizer(test_case, return_tensors="pt", padding=True, truncation=True)
            
            with torch.no_grad():
                # Forward pass with computation execution
                outputs = model(
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    execute_wasm=True,
                    return_dict=True
                )
            
            print(f"Input shape: {inputs.input_ids.shape}")
            print(f"Logits shape: {outputs['logits'].shape}")
            
            # Decode the input
            input_text = model.text_tokenizer.decode(inputs.input_ids[0], skip_special_tokens=True)
            print(f"Decoded input: {input_text}")
            
            # Look for computation tokens in output
            generated_ids = inputs.input_ids.clone()
            
            # Simple greedy decoding for demonstration
            next_input = generated_ids
            generated_tokens = []
            
            for step in range(50):  # Generate up to 50 tokens
                with torch.no_grad():
                    step_outputs = model(
                        input_ids=next_input,
                        attention_mask=torch.ones_like(next_input),
                        execute_wasm=True
                    )
                
                logits = step_outputs["logits"]
                next_token_id = torch.argmax(logits[0, -1, :], dim=-1)
                
                # Stop if eos token
                if next_token_id.item() == model.text_tokenizer.eos_token_id:
                    break
                
                generated_tokens.append(next_token_id.item())
                next_input = torch.cat([next_input, next_token_id.unsqueeze(0).unsqueeze(0)], dim=1)
                
                # Check if we're generating computation-related tokens
                token_text = model.text_tokenizer.decode([next_token_id.item()])
                if '<computation>' in token_text or '</computation>' in token_text or '<result>' in token_text:
                    print(f"  Generated computation token: {token_text} (ID: {next_token_id.item()})")
                
                # Stop early if we hit a computation tag
                if len(generated_tokens) > 0 and any('<computation>' in model.text_tokenizer.decode([tok]) for tok in generated_tokens[-5:]):
                    break
            
            # Decode the full generated sequence
            full_sequence = torch.cat([inputs.input_ids[0], torch.tensor(generated_tokens)], dim=0)
            generated_text = model.text_tokenizer.decode(full_sequence, skip_special_tokens=True)
            
            print(f"Full generated text: {generated_text}")
            
            # Analyze execution results from the forward pass
            execution_results = outputs.get('execution_results', [])
            print(f"\\nExecution results: {len(execution_results)} operations analyzed during forward pass")
            
            for i, result in enumerate(execution_results):
                print(f"  Operation {i+1}:")
                print(f"    Layer: {result.get('layer', 'N/A')}")
                print(f"    Was executed: {result.get('executed', 'N/A')}")
                print(f"    Success: {result.get('success', 'N/A')}")
                wat_code = result.get('wat_code', '')
                if wat_code:
                    print(f"    WAT code: {wat_code[:100]}{'...' if len(wat_code) > 100 else ''}")
                error = result.get('error', '')
                if error:
                    print(f"    Error: {error}")
                print()
                
        except Exception as e:
            print(f"Error in detailed test: {e}")
            import traceback
            traceback.print_exc()


def run_simple_generation(model, test_cases):
    """Run simple generation tests with more detailed output."""
    print("\\n" + "="*80)
    print("SIMPLE GENERATION TESTS")
    print("="*80)
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\\nTest {i}: '{test_case}'")
        print("-" * 40)
        
        try:
            # Use the model's generation method
            result = model.generate_with_wasm(
                input_text=test_case,
                max_length=150,
                temperature=0.1,
                execute_wasm=True
            )
            
            print(f"Input: {result['input']}")
            print(f"Processed Output: {result['output']}")
            print(f"Raw Output: {result['raw_output']}")
            print(f"WASM Executed: {result['wasm_executed']}")
            print(f"Computation tokens processed: {result['computation_tokens_processed']}")
            
            # Detailed execution analysis
            execution_results = result['execution_results']
            print(f"\\nForward-pass execution analysis ({len(execution_results)} operations):")
            
            for j, exec_res in enumerate(execution_results):
                layer = exec_res.get('layer', 'N/A')
                executed = exec_res.get('executed', False)
                success = exec_res.get('success', None)
                wat_code = exec_res.get('wat_code', '')
                error = exec_res.get('error', '')
                score = exec_res.get('score', 'N/A')
                
                print(f"  Op {j+1} @ layer {layer}:")
                print(f"    - Executed: {executed}")
                print(f"    - Success: {success}")
                print(f"    - Score: {score}")
                if wat_code:
                    print(f"    - WAT preview: {wat_code[:80]}...")
                if error:
                    print(f"    - Error: {error}")
                    
            print()
            
        except Exception as e:
            print(f"Error in generation test {i}: {e}")
            import traceback
            traceback.print_exc()


def main():
    print("Detailed Inference Script for ByteLogic WorldModel")
    print("="*80)
    
    # Analyze the checkpoint structure
    state_dict = analyze_model_components()
    
    if state_dict is None:
        print("Failed to analyze model checkpoint. Cannot proceed.")
        return
    
    # Create the loaded model interface
    model = create_loaded_model_interface(state_dict)
    
    if model is None:
        print("Failed to create model interface. Cannot proceed.")
        return
    
    # Test cases focusing on computation/function calling
    test_cases = [
        "What is 12 * 15?",
        "Calculate 24 divided by 6",
        "What is 7 + 8?",
        "If Alice is parent of Bob, who is the child of Alice?",
        "What is 2 to the power of 8?"
    ]
    
    # Run detailed computation analysis
    test_computation_detailed(model, test_cases)
    
    # Run simple generation tests
    run_simple_generation(model, test_cases)
    
    # Summary statistics
    if hasattr(model, 'computation_processor'):
        stats = model.get_computation_stats()
        print("\\n" + "="*80)
        print("COMPUTATION PROCESSOR STATISTICS")
        print("="*80)
        for key, value in stats.items():
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()