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

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Add pad token if missing
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
        device_map="auto"
    )
    
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
    parser.add_argument("--model_path", default="blueprint_model_output", help="Path to the trained model")
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
