#!/usr/bin/env python3
"""
WASM WorldModel System Status
============================

Shows the current status of the WASM training and inference system.
"""

import os
import sys
from pathlib import Path

def check_system_status():
    """Check the status of all WASM system components."""
    print("🚀 WASM WorldModel System Status")
    print("=" * 60)
    
    # Check core components
    print("\n📁 Core Components:")
    components = {
        "Training script": "train_wasm_worldmodel.py",
        "Inference script": "run_wasm_inference.py", 
        "Model verification": "test_model_saving.py",
        "Sandbox integration test": "test_sandbox_integration.py",
        "Training component test": "test_wasm_training.py"
    }
    
    for name, filename in components.items():
        status = "✅" if os.path.exists(filename) else "❌"
        print(f"   {status} {name}: {filename}")
    
    # Check source modules
    print("\n🔧 Source Modules:")
    src_modules = [
        "src/models/qwen_wasm_adapter.py",
        "src/tokenization/wat_tokenizer.py",
        "src/training/wasm_dataset.py",
        "src/training/wasm_data_collator.py", 
        "src/training/wasm_trainer.py",
        "src/execution/wasm_executor.py",
        "src/execution/wasm_api.py"
    ]
    
    for module in src_modules:
        status = "✅" if os.path.exists(module) else "❌"
        module_name = module.split("/")[-1].replace(".py", "")
        print(f"   {status} {module_name}: {module}")
    
    # Check data
    print("\n📊 Training Data:")
    data_files = [
        "data/converted/basic_arithmetic_training.txt",
        "data/converted/system_operations_training.txt", 
        "data/converted/complex_logic_training.txt"
    ]
    
    total_examples = 0
    for data_file in data_files:
        if os.path.exists(data_file):
            try:
                with open(data_file, 'r') as f:
                    content = f.read()
                    examples = len([x for x in content.split('\n\n') if x.strip()])
                    total_examples += examples
                print(f"   ✅ {data_file.split('/')[-1]}: {examples} examples")
            except:
                print(f"   ⚠️  {data_file.split('/')[-1]}: Error reading")
        else:
            print(f"   ❌ {data_file.split('/')[-1]}: Not found")
    
    print(f"   📈 Total training examples: {total_examples}")
    
    # Check documentation
    print("\n📚 Documentation:")
    docs = [
        "spec/architecture.md",
        "spec/design_decisions.md",
        "README.md"
    ]
    
    for doc in docs:
        status = "✅" if os.path.exists(doc) else "❌"
        print(f"   {status} {doc}")
    
    # Check dependencies
    print("\n🔗 Dependencies:")
    try:
        import torch
        print(f"   ✅ PyTorch: {torch.__version__}")
    except:
        print(f"   ❌ PyTorch: Not installed")
    
    try:
        import transformers
        print(f"   ✅ Transformers: {transformers.__version__}")
    except:
        print(f"   ❌ Transformers: Not installed")
    
    try:
        import wasmtime
        print(f"   ✅ Wasmtime: Available")
    except:
        print(f"   ⚠️  Wasmtime: Not available (WASM execution will be simulated)")
    
    # Check base model
    print("\n🤖 Base Model:")
    model_path = "/home/bigattichouse/workspace/model/Qwen3-0.6B"
    if os.path.exists(model_path):
        print(f"   ✅ Qwen3-0.6B: {model_path}")
        
        # Check model files
        model_files = ["config.json", "tokenizer_config.json"]
        for file in model_files:
            file_path = os.path.join(model_path, file)
            if os.path.exists(file_path):
                print(f"       ✅ {file}")
            else:
                print(f"       ❌ {file}")
    else:
        print(f"   ❌ Qwen3-0.6B: Not found at {model_path}")
    
    # Training readiness check
    print("\n🎯 System Readiness:")
    ready_for_training = all([
        os.path.exists("train_wasm_worldmodel.py"),
        os.path.exists("src/models/qwen_wasm_adapter.py"),
        os.path.exists("src/tokenization/wat_tokenizer.py"),
        os.path.exists(model_path),
        total_examples > 0
    ])
    
    ready_for_inference = all([
        os.path.exists("run_wasm_inference.py"),
        os.path.exists("src/models/qwen_wasm_adapter.py")
    ])
    
    print(f"   {'✅' if ready_for_training else '❌'} Ready for training")
    print(f"   {'✅' if ready_for_inference else '❌'} Ready for inference")
    
    # Usage examples
    print(f"\n📖 Usage Examples:")
    
    if ready_for_training:
        print(f"   🏋️  Training:")
        print(f"      # Basic training (10 epochs, sandbox enabled)")
        print(f"      python train_wasm_worldmodel.py")
        print(f"      ")
        print(f"      # Long training (30 epochs)")
        print(f"      python train_wasm_worldmodel.py --epochs 30")
        print(f"      ")
        print(f"      # Fast development (no sandbox)")
        print(f"      python train_wasm_worldmodel.py --no-sandbox")
    
    if ready_for_inference:
        print(f"   🤖 Inference (after training):")
        print(f"      # Interactive mode")
        print(f"      python run_wasm_inference.py --model ./wasm_worldmodel_output/final_model")
        print(f"      ")
        print(f"      # Single query")
        print(f"      python run_wasm_inference.py --model ./wasm_worldmodel_output/final_model --query \"Calculate 17 times 23\"")
        print(f"      ")
        print(f"      # Benchmark mode")
        print(f"      python run_wasm_inference.py --model ./wasm_worldmodel_output/final_model --benchmark")
    
    print(f"\n🧪 Testing:")
    print(f"   python test_wasm_training.py      # Test training components")
    print(f"   python test_sandbox_integration.py  # Test sandbox")
    print(f"   python test_model_saving.py         # Test model save/load")
    
    return ready_for_training, ready_for_inference

def main():
    """Main status check."""
    ready_for_training, ready_for_inference = check_system_status()
    
    print(f"\n🎉 Summary:")
    if ready_for_training and ready_for_inference:
        print(f"   ✅ WASM WorldModel system is fully operational!")
        print(f"   🚀 Ready to train and run inference")
    elif ready_for_training:
        print(f"   ✅ System ready for training")
        print(f"   ⚠️  Inference requires a trained model")
    else:
        print(f"   ❌ System needs setup - check missing components above")

if __name__ == "__main__":
    main()