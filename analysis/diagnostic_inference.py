#!/usr/bin/env python3
"""
Diagnostic Inference Script for ByteLogic WorldModel
==================================================

Comprehensive analysis to understand why execution success is only 28%.
Focuses on token-level diagnostics and execution pipeline analysis.
"""

import torch
import sys
from pathlib import Path
import json
import re

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from models.qwen_wasm_adapter import QwenWASMAdapter
from execution.bytelogic_executor import ByteLogicExecutor
from execution.computation_processor import ComputationTokenProcessor

def create_diagnostic_model():
    """Create a model with detailed diagnostic capabilities."""
    print("Creating diagnostic model...")
    
    # Load base model
    base_model_path = "~/workspace/model/Qwen3-0.6B"
    base_model_path = base_model_path.replace("~", str(Path.home()))
    
    try:
        model = QwenWASMAdapter(
            model_path=base_model_path,
            cross_modal_layers=[3, 7, 11, 15, 19, 23, 27],
            freeze_text_layers=False
        )
        
        # Load the trained weights from checkpoint
        checkpoint_path = "integrated_worldmodel_output/final_integrated_model.pt"
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
            
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        
        print("✅ Diagnostic model loaded successfully")
        return model
        
    except Exception as e:
        print(f"❌ Error creating diagnostic model: {e}")
        return None


def diagnose_token_generation(model, test_input):
    """Diagnose token generation process in detail."""
    print(f"\\nDIAGNOSIS: Token generation for '{test_input}'")
    print("-" * 60)
    
    # Tokenize input
    inputs = model.text_tokenizer(test_input, return_tensors="pt", padding=True, truncation=True)
    input_ids = inputs['input_ids']
    
    print(f"Input tokens: {input_ids.shape[1]} tokens")
    print(f"Input text: {test_input}")
    
    # Decode input tokens to see how they're represented
    decoded_input = model.text_tokenizer.decode(input_ids[0], skip_special_tokens=True)
    print(f"Decoded back: '{decoded_input}'")
    
    # Run forward pass to see what happens internally
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=inputs['attention_mask'],
            execute_wasm=True,  # Enable execution for diagnostics
            return_dict=True
        )
    
    print(f"Logits shape: {outputs['logits'].shape}")
    print(f"Number of execution results analyzed: {len(outputs['execution_results'])}")
    
    # Print execution results
    for i, exec_result in enumerate(outputs['execution_results']):
        layer = exec_result.get('layer', 'N/A')
        executed = exec_result.get('executed', False)
        success = exec_result.get('success', None)
        wat_code = exec_result.get('wat_code', '')
        score = exec_result.get('score', 'N/A')
        error = exec_result.get('error', '')
        
        print(f"\\nExecution attempt {i+1} (Layer {layer}):")
        print(f"  - Executed: {executed}")
        print(f"  - Success: {success}")
        print(f"  - Score: {score}")
        if wat_code:
            print(f"  - Generated code preview: {wat_code[:150]}...")
        if error:
            print(f"  - Error: {error}")
    
    # Now try to generate more tokens step by step
    print("\\nGenerating tokens step-by-step...")
    
    # Start from the original input
    current_tokens = input_ids.clone()
    
    generated_tokens = []
    max_new_tokens = 100
    
    for step in range(max_new_tokens):
        with torch.no_grad():
            step_outputs = model(
                input_ids=current_tokens,
                attention_mask=torch.ones_like(current_tokens),
                execute_wasm=True
            )
        
        logits = step_outputs["logits"]
        next_token_id = torch.argmax(logits[0, -1, :], dim=-1)
        
        # Check if it's an EOS token
        if next_token_id.item() == model.text_tokenizer.eos_token_id:
            print(f"  Step {step+1}: Generated EOS token - stopping")
            break
        
        generated_tokens.append(next_token_id.item())
        current_tokens = torch.cat([current_tokens, next_token_id.unsqueeze(0).unsqueeze(0)], dim=1)
        
        # Decode the current token to see what it is
        token_text = model.text_tokenizer.decode([next_token_id.item()])
        
        # Look for computation-related tokens
        if ('<' in token_text or '>' in token_text or '<computation>' in test_input or '</computation>' in test_input):
            print(f"  Step {step+1}: Generated special/computation token: '{token_text}' (ID: {next_token_id.item()})")
        
        if step < 5:  # Just show first few tokens
            print(f"  Step {step+1}: Token ID {next_token_id.item()}, Text: '{token_text}'")
        
        # Look for computation token patterns in the growing sequence
        full_generated = model.text_tokenizer.decode(current_tokens[0], skip_special_tokens=False)
        if '<computation>' in full_generated:
            print(f"  Found <computation> tag in generated sequence!")
            # Extract the computation part
            comp_start = full_generated.find('<computation>')
            comp_end = full_generated.find('</computation>')
            if comp_end != -1:
                computation_block = full_generated[comp_start:comp_end+len('</computation>')]
                print(f"  Computation block: {computation_block[:200]}...")
        
        # Stop early if we detect computation tokens
        if len(generated_tokens) > 10 and '<computation>' in full_generated:
            print("  Stopping early - found computation tokens")
            break
    
    # Final decode
    final_text = model.text_tokenizer.decode(current_tokens[0], skip_special_tokens=False)
    print(f"\\nFinal generated sequence:")
    print(f"  Length: {len(final_text)} chars")
    print(f"  Content: {final_text[:500]}{'...' if len(final_text) > 500 else ''}")
    
    # Process computation tokens if they exist
    if '<computation>' in final_text:
        print("\\nProcessing computation tokens...")
        processor = ComputationTokenProcessor(
            bytelogic_executor=ByteLogicExecutor(),
            tokenizer=model.bytelogic_tokenizer
        )
        
        try:
            processed_result = processor.process_text(final_text)
            print(f"  Processed result: {processed_result[:300]}...")
        except Exception as e:
            print(f"  Error processing computation tokens: {e}")
    
    return final_text


def test_byte_logic_execution():
    """Test ByteLogic execution pipeline separately."""
    print("\\n" + "="*80)
    print("BYTELOGIC EXECUTION PIPELINE DIAGNOSTICS")
    print("="*80)
    
    # Test the ByteLogic executor directly
    executor = ByteLogicExecutor()
    
    # Test samples of ByteLogic code that might be generated by the model
    test_codes = [
        # Simple arithmetic
        '''REL calculation
FACT calculation a 5
FACT calculation b 3
SOLVE
QUERY calculation ? ?
''',
        
        # More complex logic
        '''REL parent
REL ancestor
FACT parent alice bob
FACT parent bob charlie
RULE ancestor: SCAN parent MATCH $0, JOIN parent $0, EMIT ancestor $1 $2
SOLVE
QUERY ancestor alice ?
''',
        
        # Invalid code to test error handling
        '''INVALID CODE
THIS IS NOT VALID BYTELOGIC
''',
        
        # Correct simple operation
        '''REL math
FACT math operand1 12
FACT math operand2 4
SOLVE
QUERY math ? ?
'''
    ]
    
    for i, code in enumerate(test_codes, 1):
        print(f"\\nTest {i}: {code.split('\\n')[0][:30]}...")
        result = executor.execute_bytelogic(code)
        print(f"  Success: {result['success']}")
        print(f"  Result: {result['result']}")
        print(f"  Query Results: {result['query_results']}")
        print(f"  Error: {result['error']}")
        print(f"  Computation Token: {result['computation_token']}")


def analyze_training_examples():
    """Analyze sample training examples to understand expected patterns."""
    print("\\n" + "="*80)
    print("TRAINING EXAMPLE ANALYSIS")
    print("="*80)
    
    dataset_path = "training/datasets/complete_bytelogic_dataset.json"
    
    try:
        with open(dataset_path, 'r') as f:
            data = json.load(f)
        
        train_examples = data.get('train', [])[:5]  # First 5 examples
        
        print("\\nSample training examples (showing patterns):")
        for i, example in enumerate(train_examples):
            input_text = example.get('input', '')
            output_text = example.get('output', '')
            
            print(f"\\nExample {i+1}:")
            print(f"  Input: '{input_text}'")
            print(f"  Output: '{output_text[:100]}{'...' if len(output_text) > 100 else ''}'")
            
            # Look for computation patterns
            comp_matches = re.findall(r'<computation>(.*?)</computation>', output_text, re.DOTALL)
            if comp_matches:
                print(f"  Found {len(comp_matches)} computation block(s)")
                for j, comp_block in enumerate(comp_matches):
                    print(f"    Block {j+1}: {comp_block[:100]}...")
            else:
                print("  No computation blocks found in this example")
                
    except Exception as e:
        print(f"Error analyzing training examples: {e}")


def diagnose_model_architecture(model):
    """Diagnose the model architecture and trained components."""
    print("\\n" + "="*80)
    print("MODEL ARCHITECTURE DIAGNOSTICS")
    print("="*80)
    
    print(f"Text model parameters: {sum(p.numel() for p in model.text_model.parameters()):,}")
    print(f"WASM stream parameters: {sum(p.numel() for p in model.wasm_layers.parameters()):,}")
    print(f"Cross-modal parameters: {sum(p.numel() for name, p in model.named_parameters() if 'cross_modal' in name):,}")
    
    # Check if WASM components are properly trained
    wasm_params_trained = []
    text_params_trained = []
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            if 'wasm' in name:
                wasm_params_trained.append(name)
            elif 'text_model' in name:
                text_params_trained.append(name)
    
    print(f"\\nWASM parameters with gradients: {len(wasm_params_trained)}")
    print(f"Text parameters with gradients: {len(text_params_trained)}")
    
    # Show some WASM params
    print("\\nSample WASM trainable parameters:")
    for name in wasm_params_trained[:5]:
        param = dict(model.named_parameters())[name]
        print(f"  {name}: {list(param.shape)}")
    
    print("\\nCross-modal fusion layers:")
    for layer_idx in model.cross_modal_indices:
        layer = model.cross_modal_layers[str(layer_idx)]
        params = sum(p.numel() for p in layer.parameters())
        print(f"  Layer {layer_idx}: {params:,} parameters")


def main():
    print("Diagnostic Inference Script for ByteLogic WorldModel")
    print("="*80)
    print("Analyzing the 28.57% execution success issue...")
    
    # Create diagnostic model
    model = create_diagnostic_model()
    
    if model is None:
        print("\\n❌ Cannot proceed without diagnostic model")
        return
    
    # Diagnose the model architecture
    diagnose_model_architecture(model)
    
    # Test different types of inputs
    test_inputs = [
        "What is 5 + 3?",
        "If Alice is parent of Bob, who is the child?",
        "Calculate 12 * 4",
        "What are the relationships?"
    ]
    
    for test_input in test_inputs:
        diagnose_token_generation(model, test_input)
    
    # Test ByteLogic execution pipeline
    test_byte_logic_execution()
    
    # Analyze training examples
    analyze_training_examples()
    
    print("\\n" + "="*80)
    print("DIAGNOSTIC SUMMARY")
    print("="*80)
    print("\\nBased on the analysis, potential causes for 28% success rate:")
    print("1. Model not properly generating syntactically correct ByteLogic code")
    print("2. WASM components not sufficiently trained vs text components")
    print("3. Computation token detection/handling not working properly")
    print("4. Layer-specific specialization not effective for all operations")
    print("5. Scoring mechanism not selecting best computation candidates")
    
    print("\\nRecommended fixes:")
    print("1. Add more error-correction training examples")
    print("2. Improve scoring function for computation candidates")
    print("3. Adjust training balance between text and WASM components")
    print("4. Enhance tokenization of computation-related concepts")


if __name__ == "__main__":
    main()