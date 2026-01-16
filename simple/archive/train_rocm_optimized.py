#!/usr/bin/env python3
"""
Optimized ROCm training script for WorldModel using improved PyTorch configuration
Based on finetune.md recommendations to avoid TorchTune compatibility issues
"""

import os
import sys
import asyncio
from pathlib import Path
from dataclasses import dataclass
from typing import List

# Add src to Python path
sys.path.append(str(Path(__file__).parent / "src"))

from src.training.sftTrainer import SFTTrainer, SFTConfig
from src.training.dataGenerator import TrainingExample, DataGenerator
from src.utils.config import TrainingConfig


@dataclass
class OptimizedROCmConfig(SFTConfig):
    """Conservative configuration for ROCm fine-tuning - start small and scale up."""
    
    # Model configuration - using Qwen3-0.6B (newest and right size for MI50)
    model_name: str = "../model/Qwen3-0.6B"
    
    # Optimized memory settings for 0.6B model on 32GB (using more VRAM)
    max_sequence_length: int = 4096  # Longer sequences for better training
    batch_size: int = 4              # Higher batch size with VRAM headroom
    gradient_accumulation_steps: int = 2  # Effective batch = 8
    learning_rate: float = 5e-5      # Keep original conservative rate
    
    # Training configuration
    num_epochs: int = 3              # Multiple epochs for better fine-tuning
    warmup_steps: int = 10           # Minimal warmup for testing
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    
    # Monitoring and saving
    save_steps: int = 25             # More frequent saves
    eval_steps: int = 25
    logging_steps: int = 5
    output_dir: str = "./qwen3_0.6b_rocm_optimized"
    
    # LoRA configuration for 0.6B model (optimized for more VRAM usage)
    use_lora: bool = True
    lora_rank: int = 32              # Higher rank for better adaptation quality
    lora_alpha: int = 64             # 2x rank
    lora_dropout: float = 0.1
    
    # ROCm-specific settings (keep these for stability)
    use_4bit: bool = False
    use_8bit: bool = False
    
    def __post_init__(self):
        super().__post_init__()
        # Qwen2.5 specific LoRA targets
        self.lora_target_modules = [
            "q_proj", "v_proj", "k_proj", "o_proj",  # Attention
            "gate_proj", "up_proj", "down_proj"       # MLP
        ]


def setup_rocm_environment():
    """Setup ROCm environment variables for optimal training."""
    env_vars = {
        "HSA_OVERRIDE_GFX_VERSION": "9.0.6",  # Native gfx906
        "PYTORCH_ROCM_ARCH": "gfx906", 
        "TOKENIZERS_PARALLELISM": "false",
        "HIP_VISIBLE_DEVICES": "0",
        "PYTORCH_HIP_ALLOC_CONF": "expandable_segments:True,max_split_size_mb:512",
        "OMP_NUM_THREADS": "1",
        "HSA_DISABLE_CACHE": "1",
        "HSA_FORCE_FINE_GRAIN_PCIE": "1"
    }
    
    for key, value in env_vars.items():
        os.environ[key] = value
        print(f"Set {key}={value}")


def test_gpu_availability():
    """Test GPU availability and memory."""
    import torch
    
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        # Clear any existing GPU memory first
        torch.cuda.empty_cache()
        
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"GPU Memory: {memory_gb:.1f} GB")
        
        # Test tensor creation with proper cleanup
        try:
            print("Testing GPU tensor creation...")
            test_tensor = torch.zeros(1000, 1000, dtype=torch.float32).cuda()
            print("✅ GPU tensor creation successful")
            del test_tensor
            torch.cuda.empty_cache()
            
            # Test larger allocation to see available memory
            print("Testing larger allocation...")
            large_tensor = torch.zeros(2000, 2000, dtype=torch.float32).cuda()
            print("✅ Large tensor creation successful")
            del large_tensor
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"❌ GPU tensor creation failed: {e}")
            torch.cuda.empty_cache()
            return False
        return True
    else:
        print("❌ No GPU detected")
        return False


async def main():
    """Main training function."""
    print("🔥 Starting Optimized ROCm Fine-tuning")
    print("=" * 50)
    
    # Setup environment
    print("🚀 Setting up ROCm environment...")
    setup_rocm_environment()
    
    # Test GPU
    print("\n🧪 Testing GPU availability...")
    if not test_gpu_availability():
        print("❌ GPU test failed. Check ROCm installation.")
        return
    
    # Load training data
    print("\n📊 Loading training data...")
    data_files = [
        "./data/worldmodel_enhanced_training.json",
        "./data/worldmodel_final_training.json", 
        "./data/worldmodel_training.json"
    ]
    
    examples = []
    for data_file in data_files:
        if Path(data_file).exists():
            print(f"Loading: {data_file}")
            generator = DataGenerator(TrainingConfig())
            file_examples = generator.load_dataset(data_file)
            examples.extend(file_examples)
            print(f"Loaded {len(file_examples)} examples from {data_file}")
            break
    
    if not examples:
        print("❌ No training data found. Check data files exist.")
        return
    
    print(f"Total examples: {len(examples)}")
    
    # Create optimized config
    config = OptimizedROCmConfig()
    
    # Test memory with current batch size
    print(f"\n🔧 Testing with batch_size={config.batch_size}")
    print(f"   gradient_accumulation_steps={config.gradient_accumulation_steps}")
    print(f"   Effective batch size: {config.batch_size * config.gradient_accumulation_steps}")
    
    # Initialize trainer
    print("\n🏋️ Initializing SFT trainer...")
    trainer = SFTTrainer(config)
    
    try:
        # Start training with full dataset - 0.6B has plenty of memory
        print("\n🚀 Starting training...")
        results = await trainer.train(
            examples=examples,  # Full dataset with 0.6B model
            eval_split=0.2,
            resume_from_checkpoint=None
        )
        
        print("\n✅ Training completed successfully!")
        print(f"Final metrics: {results}")
        
        # Save metrics
        metrics_file = Path(config.output_dir) / "training_metrics.json"
        trainer.save_metrics(str(metrics_file))
        print(f"📊 Metrics saved to: {metrics_file}")
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        
        # Suggest memory optimization if OOM
        if "out of memory" in str(e).lower() or "oom" in str(e).lower():
            print("\n💡 Memory optimization suggestions:")
            print("1. Reduce batch_size from 4 to 2 or 1")
            print("2. Reduce max_sequence_length from 4096 to 2048")
            print("3. Reduce lora_rank from 16 to 8")
            print("4. Set gradient_checkpointing=True (already enabled)")


if __name__ == "__main__":
    asyncio.run(main())