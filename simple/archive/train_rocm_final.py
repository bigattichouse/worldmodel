#!/usr/bin/env python3
"""
Final optimized ROCm training script for WorldModel using Qwen3-0.6B
Optimized for ~80-85% VRAM usage on MI50 32GB for best performance
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
class FinalOptimizedConfig(SFTConfig):
    """Final optimized configuration for ROCm MI50 - using ~80-85% VRAM."""
    
    # Model configuration
    model_name: str = "../model/Qwen3-0.6B"
    
    # Performance-optimized memory settings (should use ~25-27GB of 32GB)
    max_sequence_length: int = 4096  # Full context for quality training
    batch_size: int = 4              # Higher throughput
    gradient_accumulation_steps: int = 2  # Effective batch = 8
    learning_rate: float = 1e-4      # Slightly higher for faster convergence
    
    # Training configuration
    num_epochs: int = 3              # Multiple epochs for quality fine-tuning
    warmup_steps: int = 50           # Proper warmup
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    
    # Monitoring and saving
    save_steps: int = 25
    eval_steps: int = 25
    logging_steps: int = 5
    output_dir: str = "./qwen3_0.6b_rocm_final"
    
    # High-quality LoRA configuration
    use_lora: bool = True
    lora_rank: int = 32              # High rank for quality adaptation
    lora_alpha: int = 64             # 2x rank
    lora_dropout: float = 0.1
    
    # ROCm-specific settings
    use_4bit: bool = False
    use_8bit: bool = False
    
    def __post_init__(self):
        super().__post_init__()
        # Qwen3 specific LoRA targets
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
        "PYTORCH_HIP_ALLOC_CONF": "max_split_size_mb:256",  # Smaller splits for efficiency
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
        torch.cuda.empty_cache()
        
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"GPU Memory: {memory_gb:.1f} GB")
        
        try:
            test_tensor = torch.zeros(1000, 1000, dtype=torch.float32).cuda()
            print("✅ GPU operations working")
            del test_tensor
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"❌ GPU test failed: {e}")
            return False
        return True
    else:
        print("❌ No GPU detected")
        return False


async def main():
    """Main training function."""
    print("🔥 Final Optimized ROCm Fine-tuning - Qwen3-0.6B")
    print("=" * 60)
    
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
    config = FinalOptimizedConfig()
    
    print(f"\n🔧 Final Optimized Configuration:")
    print(f"   Model: Qwen3-0.6B (latest)")
    print(f"   Sequence Length: {config.max_sequence_length}")
    print(f"   Batch Size: {config.batch_size}")
    print(f"   Gradient Accumulation: {config.gradient_accumulation_steps}")
    print(f"   Effective Batch: {config.batch_size * config.gradient_accumulation_steps}")
    print(f"   LoRA Rank: {config.lora_rank}")
    print(f"   Epochs: {config.num_epochs}")
    print(f"   Learning Rate: {config.learning_rate}")
    
    # Initialize trainer
    print("\n🏋️ Initializing SFT trainer...")
    trainer = SFTTrainer(config)
    
    try:
        # Start training
        print("\n🚀 Starting optimized training...")
        print(f"Expected VRAM usage: ~80-85% of 32GB")
        
        results = await trainer.train(
            examples=examples,  # Full dataset
            eval_split=0.2,
            resume_from_checkpoint=None
        )
        
        print("\n✅ Training completed successfully!")
        print(f"Final metrics: {results}")
        
        # Save metrics
        metrics_file = Path(config.output_dir) / "training_metrics.json"
        trainer.save_metrics(str(metrics_file))
        print(f"📊 Metrics saved to: {metrics_file}")
        
        print(f"\n🎉 SUCCESS: WorldModel fine-tuned on ROCm MI50!")
        print(f"📁 Model saved to: {config.output_dir}")
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())