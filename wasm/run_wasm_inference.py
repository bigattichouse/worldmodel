#!/usr/bin/env python3
"""
WASM WorldModel Inference
========================

Run inference with a trained WASM WorldModel.
"""

import sys
import os
import json
import pickle
import argparse
import torch
from pathlib import Path

sys.path.append('src')

from src.models.qwen_wasm_adapter import QwenWASMAdapter
from src.tokenization.wat_tokenizer import WATTokenizer
from transformers import AutoTokenizer

def load_trained_model(model_path: str):
    """Load a trained WASM model with all components."""
    model_path = Path(model_path)
    
    # Load training metadata
    metadata_path = model_path / "training_metadata.json"
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        print(f"📋 Model Info:")
        print(f"   Base: {metadata['base_model']}")
        print(f"   Trained: {metadata['training_date']}")
        print(f"   Epochs: {metadata['training_args']['epochs']}")
        print(f"   Dataset: {metadata['dataset_info']['train_size']} examples")
        print(f"   Sandbox: {metadata['sandbox_enabled']}")
    else:
        metadata = {}
        print(f"⚠️  No metadata found, using defaults")
    
    # Load text tokenizer
    text_tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
    
    # Load WASM tokenizer
    wasm_tokenizer_path = model_path / "wasm_tokenizer.pkl"
    if wasm_tokenizer_path.exists():
        with open(wasm_tokenizer_path, 'rb') as f:
            wasm_tokenizer = pickle.load(f)
        print(f"   WASM tokenizer: {wasm_tokenizer.vocab_size} vocab")
    else:
        # Fallback to new tokenizer
        wasm_tokenizer = WATTokenizer(vocab_size=8000)
        print(f"   WASM tokenizer: Created new (8000 vocab)")
    
    # Load WASM adapter model
    sandbox_enabled = metadata.get('sandbox_enabled', True)
    sandbox_config = metadata.get('sandbox_config', {
        'vm_name': 'wasm-inference',
        'memory': '512M',
        'timeout': 30
    })
    
    wasm_adapter = QwenWASMAdapter.from_pretrained(
        model_path=str(model_path),
        freeze_text_layers=False,
        use_sandbox=sandbox_enabled,
        sandbox_config=sandbox_config
    )
    
    # Set WASM tokenizer for token-to-WAT conversion during inference
    wasm_adapter.set_wasm_tokenizer(wasm_tokenizer)
    
    print(f"   Cross-modal layers: {metadata.get('cross_modal_layers', [3, 7, 11])}")
    print(f"   WASM execution: Enabled during forward pass")
    print(f"✅ Model loaded successfully")
    
    return wasm_adapter, text_tokenizer, wasm_tokenizer, metadata

def run_inference(model, text_tokenizer, wasm_tokenizer, query: str, max_length: int = 512):
    """Run inference on a single query."""
    print(f"\n🧠 Query: {query}")
    
    # Tokenize input
    inputs = text_tokenizer(
        query, 
        return_tensors="pt", 
        max_length=max_length, 
        truncation=True, 
        padding=True
    )
    
    # Move to GPU if available
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}
        model = model.cuda()
    
    # Generate response
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=256,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=text_tokenizer.eos_token_id,
            execute_wasm=True  # Enable WASM execution
        )
    
    # Decode response
    input_length = inputs["input_ids"].shape[1]
    response_tokens = outputs[0][input_length:]
    response = text_tokenizer.decode(response_tokens, skip_special_tokens=True)
    
    return response

def interactive_mode(model, text_tokenizer, wasm_tokenizer, metadata):
    """Run interactive inference session."""
    print(f"\n🚀 WASM WorldModel Interactive Mode")
    print(f"=" * 60)
    print(f"Type your queries, 'quit' to exit, 'help' for examples")
    print(f"")
    
    example_queries = [
        "Calculate 17 times 23",
        "What's the square root of 144?", 
        "Generate a function to calculate factorial of 5",
        "What's the current date and time?",
        "Create a physics simulation for projectile motion"
    ]
    
    while True:
        try:
            query = input("🔮 Query: ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
                
            if query.lower() == 'help':
                print(f"\n💡 Example queries:")
                for i, example in enumerate(example_queries, 1):
                    print(f"   {i}. {example}")
                print()
                continue
                
            if not query:
                continue
                
            # Run inference
            response = run_inference(model, text_tokenizer, wasm_tokenizer, query)
            
            print(f"🤖 Response:")
            print(f"   {response}")
            print()
            
        except KeyboardInterrupt:
            print(f"\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            continue

def benchmark_mode(model, text_tokenizer, wasm_tokenizer):
    """Run benchmark queries."""
    print(f"\n📊 WASM Benchmark Mode")
    print(f"=" * 60)
    
    test_queries = [
        "Calculate 17 times 23",
        "What's 2 to the power of 8?",
        "Find the factorial of 6",
        "Calculate the area of a circle with radius 5",
        "What's the current timestamp?",
        "Sort the numbers [5, 2, 8, 1, 9]",
        "Calculate compound interest: $1000 at 5% for 3 years"
    ]
    
    results = []
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n🧪 Test {i}/{len(test_queries)}: {query}")
        
        try:
            import time
            start_time = time.time()
            response = run_inference(model, text_tokenizer, wasm_tokenizer, query)
            inference_time = time.time() - start_time
            
            print(f"✅ Response ({inference_time:.2f}s): {response[:100]}...")
            results.append({
                'query': query,
                'response': response,
                'time': inference_time,
                'success': True
            })
            
        except Exception as e:
            print(f"❌ Failed: {e}")
            results.append({
                'query': query,
                'error': str(e),
                'success': False
            })
    
    # Summary
    successful = sum(1 for r in results if r['success'])
    total_time = sum(r.get('time', 0) for r in results if r['success'])
    avg_time = total_time / successful if successful > 0 else 0
    
    print(f"\n📈 Benchmark Results:")
    print(f"   Success rate: {successful}/{len(test_queries)} ({successful/len(test_queries)*100:.1f}%)")
    print(f"   Average time: {avg_time:.2f}s per query")
    print(f"   Total time: {total_time:.2f}s")

def main():
    """Main inference function."""
    parser = argparse.ArgumentParser(description="WASM WorldModel Inference")
    parser.add_argument('--model', required=True,
                       help='Path to trained WASM model directory')
    parser.add_argument('--query', type=str,
                       help='Single query to run (non-interactive)')
    parser.add_argument('--benchmark', action='store_true',
                       help='Run benchmark test suite')
    parser.add_argument('--max-length', type=int, default=512,
                       help='Maximum input length')
    
    args = parser.parse_args()
    
    # Check model path
    if not os.path.exists(args.model):
        print(f"❌ Model path not found: {args.model}")
        return
    
    print(f"🔥 WASM WorldModel Inference")
    print(f"=" * 50)
    
    # Load model
    try:
        model, text_tokenizer, wasm_tokenizer, metadata = load_trained_model(args.model)
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return
    
    # Run inference mode
    if args.query:
        # Single query mode
        response = run_inference(model, text_tokenizer, wasm_tokenizer, args.query, args.max_length)
        print(f"🤖 Response: {response}")
        
    elif args.benchmark:
        # Benchmark mode
        benchmark_mode(model, text_tokenizer, wasm_tokenizer)
        
    else:
        # Interactive mode
        interactive_mode(model, text_tokenizer, wasm_tokenizer, metadata)

if __name__ == "__main__":
    main()