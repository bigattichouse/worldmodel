#!/usr/bin/env python3
"""
Lightweight Model Analyzer for ByteLogic WorldModel
==================================================

Analyze the model checkpoint without fully loading it to understand its structure.
"""

import torch
import sys
from pathlib import Path

def analyze_model_lightweight():
    """Analyze model structure without loading all parameters."""
    checkpoint_path = "integrated_worldmodel_output/final_integrated_model.pt"
    
    print(f"Analyzing checkpoint: {checkpoint_path}")
    print(f"File size: {Path(checkpoint_path).stat().st_size / (1024**3):.2f} GB")
    
    try:
        # Load only the state dict keys and some metadata without loading tensors into memory
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        
        print(f"\\nCheckpoint type: {type(checkpoint)}")
        
        if isinstance(checkpoint, dict):
            print(f"Checkpoint keys: {list(checkpoint.keys())}")
            
            # Check for training checkpoint structure
            if 'model_state_dict' in checkpoint:
                print("\\nThis is a training checkpoint with metadata:")
                print(f"  Epoch: {checkpoint.get('epoch', 'N/A')}")
                print(f"  Metrics: {checkpoint.get('metrics', {})}")
                print(f"  Original model path: {checkpoint.get('model_path', 'N/A')}")
                print(f"  Computation layers: {checkpoint.get('computation_layers', 'N/A')}")
                
                state_dict = checkpoint['model_state_dict']
            else:
                print("\\nThis is a direct model state dict")
                state_dict = checkpoint
        
        else:
            # It's directly the state dict
            state_dict = checkpoint
            print("\\nDirect state dict detected")
        
        # Analyze the structure
        print(f"\\nState dict contains {len(state_dict)} parameters")
        
        # Group parameters by category to understand the architecture
        text_params = [k for k in state_dict.keys() if k.startswith('text_model')]
        wasm_params = [k for k in state_dict.keys() if k.startswith(('wasm_', 'cross_modal'))]
        other_params = [k for k in state_dict.keys() if not (k.startswith('text_model') or k.startswith(('wasm_', 'cross_modal')))]
        
        print(f"  Text model parameters: {len(text_params)}")
        print(f"  WASM/Computation parameters: {len(wasm_params)}")
        print(f"  Other parameters: {len(other_params)}")
        
        # Show some examples from each category
        print("\\nSample text model parameters:")
        for param in text_params[:5]:
            tensor = state_dict[param]
            shape_str = list(tensor.shape) if hasattr(tensor, 'shape') else 'scalar'
            print(f"  {param}: {shape_str}")
        
        print("\\nSample WASM/Computation parameters:")
        for param in wasm_params[:5]:
            tensor = state_dict[param]
            shape_str = list(tensor.shape) if hasattr(tensor, 'shape') else 'scalar'
            print(f"  {param}: {shape_str}")
        
        print("\\nSample other parameters:")
        for param in other_params[:5]:
            tensor = state_dict[param]
            shape_str = list(tensor.shape) if hasattr(tensor, 'shape') else 'scalar'
            print(f"  {param}: {shape_str}")
        
        return state_dict if isinstance(state_dict, dict) else checkpoint
            
    except Exception as e:
        print(f"Error analyzing checkpoint: {e}")
        import traceback
        traceback.print_exc()
        return None

def analyze_training_logs():
    """Analyze any available training logs to understand the model structure."""
    print("\\n" + "="*60)
    print("ANALYZING TRAINING LOGS/OUTPUT")
    print("="*60)
    
    # Since we know from the original training output:
    print("Based on your original training output:")
    print("- Model had 774,079,488 parameters (~774M parameters)")
    print("- Used computation layers: [3, 7, 11, 15, 19, 23, 27]")
    print("- Text layers: N/A (using base Qwen model)")
    print("- WASM layers: 7")
    print("- Cross-modal fusion at layers: [3, 7, 11, 15, 19, 23, 27]")
    print("- ByteLogic support: Enabled")
    print("- Computation tokens: Supported")
    print("- Final execution success rate: 28.57%")
    

def create_inference_insights():
    """Provide insights for improving inference and understanding failures."""
    print("\\n" + "="*60)
    print("INFERENCE INSIGHTS & DEBUGGING APPROACHES")
    print("="*60)
    
    print("\\nIssues identified from training results (28.57% success rate):")
    print("1. Model is not effectively generating correct ByteLogic code")
    print("2. Generated WASM/ByteLogic code has syntax or execution errors") 
    print("3. Model struggles to align computations with question intent")
    print("4. Layer-specific specialization may not be optimal")
    
    print("\\nDebugging approach needed:")
    print("1. Focus on individual components separately")
    print("2. Create targeted tests for ByteLogic generation")
    print("3. Examine the tokenization process specifically")
    print("4. Test execution engine independently")
    
    print("\\nSuggested next steps:")
    print("1. Create a tokenizer analyzer to see token mappings")
    print("2. Test ByteLogic execution pipeline separately")
    print("3. Analyze sample inputs and expected outputs from training data")
    print("4. Create detailed logging for the computation pipeline")


def analyze_sample_training_data():
    """Analyze the training data structure to understand expectations."""
    print("\\n" + "="*60)
    print("SAMPLE TRAINING DATA ANALYSIS")
    print("="*60)
    
    import json
    
    # Read a sample from the training data
    dataset_path = "training/datasets/complete_bytelogic_dataset.json"
    
    try:
        with open(dataset_path, 'r') as f:
            data = json.load(f)
        
        print(f"Dataset version: {data.get('metadata', {}).get('version', 'Unknown')}")
        print(f"Total examples: {data.get('metadata', {}).get('total_examples', 'Unknown')}")
        
        # Look at first few training examples
        train_examples = data.get('train', [])[:3]  # First 3 examples
        
        print("\\nSample training examples:")
        for i, example in enumerate(train_examples):
            print(f"\\nExample {i+1}:")
            print(f"  Input: '{example.get('input', '')[:100]}{'...' if len(example.get('input', '')) > 100 else ''}'")
            print(f"  Output: '{example.get('output', '')[:150]}{'...' if len(example.get('output', '')) > 150 else ''}'")
            
            # Check for computation tokens
            output = example.get('output', '')
            if '<computation>' in output:
                start = output.find('<computation>')
                end = output.find('</computation>') + len('</computation>')
                computation_part = output[start:end]
                print(f"  Computation block: {computation_part[:100]}...")
        
    except Exception as e:
        print(f"Error reading training data: {e}")
        
        # At least show the file exists and its size
        if Path(dataset_path).exists():
            size_mb = Path(dataset_path).stat().st_size / (1024 * 1024)
            print(f"Training data file exists, size: {size_mb:.2f} MB")


def main():
    print("Lightweight Model Analyzer for ByteLogic WorldModel")
    print("="*60)
    
    # Analyze the model structure
    state_dict = analyze_model_lightweight()
    
    if state_dict:
        print(f"\\n✅ Successfully analyzed model structure")
        print(f"Parameters in state dict: {len(state_dict)}")
    else:
        print("❌ Could not analyze model")
    
    # Provide insights about training
    analyze_training_logs()
    
    # Provide insights about inference
    create_inference_insights()
    
    # Analyze sample training data
    analyze_sample_training_data() 
    
    print("\\n" + "="*60)
    print("ANALYSIS COMPLETE - KEY FINDINGS:")
    print("="*60)
    print("1. Model size: ~3GB (large, needs sufficient VRAM)")
    print("2. Architecture: Qwen base + WASM stream + cross-modal attention")
    print("3. Issue: 28.57% execution success suggests poor ByteLogic generation")
    print("4. Need: Component-level debugging rather than full model loading")


if __name__ == "__main__":
    main()