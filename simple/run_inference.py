#!/usr/bin/env python3
"""
Inference script for fine-tuned WorldModel using ROCm
Supports both LoRA adapters and merged models
"""

import os
import sys
import asyncio
import argparse
from pathlib import Path
from typing import Optional

# Add src to Python path
sys.path.append(str(Path(__file__).parent / "src"))

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig


def setup_rocm_environment():
    """Setup ROCm environment for inference."""
    env_vars = {
        "HSA_OVERRIDE_GFX_VERSION": "9.0.6",
        "PYTORCH_ROCM_ARCH": "gfx906", 
        "TOKENIZERS_PARALLELISM": "false",
        "HIP_VISIBLE_DEVICES": "0",
    }
    
    for key, value in env_vars.items():
        os.environ[key] = value


class WorldModelInference:
    def __init__(self, model_path: str, base_model_path: Optional[str] = None):
        """
        Initialize inference engine.
        
        Args:
            model_path: Path to fine-tuned model (LoRA or merged)
            base_model_path: Path to base model (if using LoRA adapters)
        """
        self.model_path = Path(model_path)
        self.base_model_path = base_model_path
        self.model = None
        self.tokenizer = None
        
    def load_model(self):
        """Load the fine-tuned model."""
        print(f"🔄 Loading model from: {self.model_path}")
        
        # Setup ROCm
        setup_rocm_environment()
        
        # Clear GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Load tokenizer
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                str(self.model_path),
                trust_remote_code=True
            )
        except:
            # Fallback to base model tokenizer
            base_path = self.base_model_path or "../model/Qwen3-0.6B"
            print(f"Using tokenizer from base model: {base_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(
                base_path,
                trust_remote_code=True
            )
        
        # Set padding token if missing
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Check if this is a LoRA adapter or merged model
        if (self.model_path / "adapter_config.json").exists():
            print("📎 Detected LoRA adapter - loading with base model")
            self._load_lora_model()
        else:
            print("🔗 Detected merged model - loading directly")
            self._load_merged_model()
        
        print("✅ Model loaded successfully")
        
    def _load_lora_model(self):
        """Load LoRA adapter with base model."""
        base_path = self.base_model_path or "../model/Qwen3-0.6B"
        
        # Load base model
        base_model = AutoModelForCausalLM.from_pretrained(
            base_path,
            trust_remote_code=True,
            torch_dtype=torch.float32,
            device_map={"": 0} if torch.cuda.is_available() else None,
            low_cpu_mem_usage=True
        )
        
        # Load LoRA adapter
        self.model = PeftModel.from_pretrained(
            base_model,
            str(self.model_path)
        )
        
    def _load_merged_model(self):
        """Load merged model directly."""
        self.model = AutoModelForCausalLM.from_pretrained(
            str(self.model_path),
            trust_remote_code=True,
            torch_dtype=torch.float32,
            device_map={"": 0} if torch.cuda.is_available() else None,
            low_cpu_mem_usage=True
        )
    
    def generate(self, prompt: str, max_new_tokens: int = 512, 
                temperature: float = 0.7, top_p: float = 0.9,
                use_worldmodel_format: bool = True) -> str:
        """
        Generate response for a prompt.
        
        Args:
            prompt: Input prompt
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling
            use_worldmodel_format: Whether to use WorldModel conversation format
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # Format prompt for WorldModel if requested
        if use_worldmodel_format:
            formatted_prompt = f"<|user|>{prompt}<|end|><|assistant|>"
        else:
            formatted_prompt = prompt
        
        # Tokenize
        inputs = self.tokenizer(
            formatted_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048 - max_new_tokens
        )
        
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        # Generate
        self.model.eval()
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.1
            )
        
        # Decode response
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract assistant response if using WorldModel format
        if use_worldmodel_format and "<|assistant|>" in response:
            response = response.split("<|assistant|>")[-1].strip()
        elif not use_worldmodel_format:
            # Remove input prompt from response
            response = response[len(formatted_prompt):].strip()
        
        return response


def main():
    parser = argparse.ArgumentParser(description="WorldModel Inference")
    parser.add_argument("--model", "-m", required=True, 
                       help="Path to fine-tuned model")
    parser.add_argument("--base-model", "-b", 
                       help="Path to base model (for LoRA adapters)")
    parser.add_argument("--prompt", "-p", 
                       help="Single prompt to process")
    parser.add_argument("--interactive", "-i", action="store_true",
                       help="Interactive chat mode")
    parser.add_argument("--max-tokens", type=int, default=512,
                       help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7,
                       help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=0.9,
                       help="Top-p sampling")
    
    args = parser.parse_args()
    
    # Initialize inference engine
    engine = WorldModelInference(args.model, args.base_model)
    engine.load_model()
    
    print(f"🤖 WorldModel ready! Using model: {args.model}")
    print(f"💫 GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    
    if args.prompt:
        # Single prompt mode
        print(f"\n📝 Prompt: {args.prompt}")
        response = engine.generate(
            args.prompt, 
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p
        )
        print(f"🤖 Response: {response}")
        
    elif args.interactive:
        # Interactive mode
        print("\n💬 Interactive mode (type 'quit' to exit)")
        while True:
            try:
                prompt = input("\n👤 You: ").strip()
                if prompt.lower() in ['quit', 'exit', 'q']:
                    break
                if not prompt:
                    continue
                    
                print("🤖 WorldModel: ", end="", flush=True)
                response = engine.generate(
                    prompt,
                    max_new_tokens=args.max_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p
                )
                print(response)
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
    else:
        print("❓ Please specify --prompt or --interactive mode")


if __name__ == "__main__":
    main()