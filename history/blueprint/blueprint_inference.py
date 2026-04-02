#!/usr/bin/env python3
"""
BluePrint WorldModel Interactive Inference
=========================================

Chat with a trained BluePrint model.
"""

import torch
from pathlib import Path
import sys
import logging
import argparse
import os
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
)

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from training.blueprint_dataset import validate_blueprint_syntax

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_model_and_tokenizer(model_path: str):
    """Load and prepare model and tokenizer."""
    logger.info(f"🧠 Loading model from {model_path}")

    # Check if this is a PEFT model
    import os
    from peft import PeftModel, PeftConfig
    from transformers import AutoModelForCausalLM

    adapter_config_path = os.path.join(model_path, "adapter_config.json")

    if os.path.exists(adapter_config_path):
        # This is a PEFT model
        logger.info("🔧 Detected PEFT model, loading base model and adapter...")
        try:
            # Load the PEFT configuration to get the base model
            config = PeftConfig.from_pretrained(model_path)
            base_model_path = config.base_model_name_or_path
            
            # Handle relative paths - make sure we resolve relative to the current working directory
            if not os.path.isabs(base_model_path):
                # If it starts with ../, resolve relative to current directory
                if base_model_path.startswith('../'):
                    base_model_path = os.path.normpath(base_model_path)
                else:
                    # Otherwise, resolve relative to the adapter directory
                    base_model_path = os.path.join(model_path, base_model_path)
                    base_model_path = os.path.normpath(base_model_path)
            
            logger.info(f"📁 Loading base model from: {base_model_path}")

            # Load tokenizer from the adapter directory (it has the updated vocab)
            tokenizer = AutoTokenizer.from_pretrained(model_path)

            # Load the base model
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_path,
                trust_remote_code=True,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
                device_map="auto"
            )
            
            # Resize token embeddings if needed to match the tokenizer
            if len(tokenizer) != base_model.config.vocab_size:
                logger.info(f"🔧 Resizing token embeddings from {base_model.config.vocab_size} to {len(tokenizer)}")
                base_model.resize_token_embeddings(len(tokenizer))

            # Load the PEFT adapter on top
            model = PeftModel.from_pretrained(base_model, model_path)
            
        except Exception as e:
            logger.error(f"❌ Failed to load PEFT model: {e}")
            logger.info("🔄 Falling back to direct loading...")
            # Fallback to direct loading
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                trust_remote_code=True,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
                device_map="auto"
            )
    else:
        # Regular model
        logger.info("📦 Loading regular model...")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map="auto"
        )

    # Add pad token if missing
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    logger.info(f"✅ Model loaded with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    return model, tokenizer

def chat(model, tokenizer, user_query: str):
    """Generate a response to a user query."""
    
    # Format input
    input_text = f"User: {user_query}\n\nAssistant: "
    
    # Tokenize and move to device
    inputs = tokenizer(input_text, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_new_tokens=1536,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.1,
            no_repeat_ngram_size=3
        )
    
    # Decode
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=False)
    
    # Extract only the assistant's response
    response_start = generated_text.find("Assistant:")
    if response_start != -1:
        response = generated_text[response_start + len("Assistant:"):].strip()
    else:
        response = generated_text[len(input_text):]

    return response

def main():
    """Main interactive chat loop."""
    parser = argparse.ArgumentParser(description="Interactive BluePrint WorldModel Chat")
    parser.add_argument("--model", "--model_path", dest="model_path", default="blueprint_model_output", help="Path to the trained model")
    parser.add_argument("--test", help="Test with a single query (non-interactive)")
    args = parser.parse_args()

    # Hardware detection
    print("=== BluePrint WorldModel Interactive Inference ===")
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name()
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"✅ GPU: {gpu_name} ({total_memory:.1f}GB)")
    else:
        print("⚪ Using CPU")

    # Load model
    try:
        model, tokenizer = setup_model_and_tokenizer(args.model_path)
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        return

    # Test mode
    if args.test:
        print(f"\n🧪 Testing with query: {args.test}")
        response = chat(model, tokenizer, args.test)
        print("\n🤖 Assistant:")
        print(response)
        
        # Validate format
        is_valid, errors = validate_blueprint_syntax(response)
        if is_valid:
            print("\n✅ Blueprint format is valid.")
        else:
            print("\n❌ Invalid Blueprint format:")
            for error in errors:
                print(f"   - {error}")
        return

    # Interactive loop
    print("\n\n💬 Chat with the BluePrint model. Type 'exit' or 'quit' to end.")
    while True:
        try:
            user_query = input("\n👤 You: ")
            if user_query.lower() in ["exit", "quit"]:
                break
            
            response = chat(model, tokenizer, user_query)
            
            print("\n🤖 Assistant:")
            print(response)

            # Validate format
            is_valid, errors = validate_blueprint_syntax(response)
            if is_valid:
                print("\n✅ Blueprint format is valid.")
            else:
                print("\n❌ Invalid Blueprint format:")
                for error in errors:
                    print(f"   - {error}")

        except KeyboardInterrupt:
            print("\n\nExiting...")
            break
        except Exception as e:
            logger.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
