#!/usr/bin/env python3
"""
Simple WorldModel runner that bypasses the problematic stopping criteria.
Uses direct model generation with proper parameters.
"""
import os
import sys
import asyncio
import argparse
from pathlib import Path

# Add src to Python path
sys.path.append(str(Path(__file__).parent / "src"))

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from src.core.tagParser import TagParser
from src.execution.vmInterface import VMInterface

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

class SimpleWorldModelRunner:
    """Simple WorldModel runner with direct generation."""
    
    def __init__(self, model_path: str, enable_execution: bool = True):
        """Initialize runner."""
        setup_rocm_environment()
        
        print("🔄 Loading model and tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=torch.float16
        )
        
        self.tag_parser = TagParser()
        self.vm_interface = VMInterface() if enable_execution else None
        
        print("✅ WorldModel ready!")
    
    async def process_prompt(self, prompt: str, verbose: bool = False) -> dict:
        """Process a prompt with full WorldModel capabilities."""
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        print(f"📝 Processing: {prompt}")
        if verbose:
            print(f"Input tokens: {inputs.input_ids.shape[1]}")
        
        # Generate with optimal parameters  
        with torch.no_grad():
            outputs = self.model.generate(
                inputs.input_ids,
                max_new_tokens=128,
                min_new_tokens=15,
                temperature=0.6,
                do_sample=True,
                top_p=0.8,
                repetition_penalty=1.5,  # Strong repetition penalty
                pad_token_id=self.tokenizer.eos_token_id,
                early_stopping=False,
                attention_mask=inputs.attention_mask,
                no_repeat_ngram_size=3  # Prevent 3-gram repetition
            )
        
        # Decode response
        generated_text = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:], 
            skip_special_tokens=True
        )
        
        # Parse WorldModel tags
        parsed_result = self.tag_parser.parse(generated_text)
        
        # Execute code if present
        execution_results = []
        if parsed_result.model_tags and self.vm_interface:
            for i, model_tag in enumerate(parsed_result.model_tags):
                if verbose:
                    print(f"\n💻 Executing code block {i+1}:")
                    print(model_tag.content)
                
                try:
                    exec_result = await self.vm_interface.execute_code(
                        model_tag.language or 'python',
                        model_tag.content
                    )
                    
                    execution_results.append({
                        'status': exec_result.status.value,
                        'code': model_tag.content,
                        'language': model_tag.language,
                        'stdout': exec_result.stdout,
                        'stderr': exec_result.stderr,
                        'return_code': exec_result.return_code,
                        'execution_time': exec_result.execution_time
                    })
                    
                    if verbose:
                        print(f"   Status: {exec_result.status.value}")
                        if exec_result.stdout:
                            print(f"   Output: {exec_result.stdout}")
                        if exec_result.stderr:
                            print(f"   Error: {exec_result.stderr}")
                
                except Exception as e:
                    execution_results.append({
                        'status': 'execution_error',
                        'code': model_tag.content,
                        'language': model_tag.language,
                        'error': str(e)
                    })
        
        return {
            'prompt': prompt,
            'generated_text': generated_text,
            'parsed_tags': {
                'think_tags': [{'content': t.content} for t in parsed_result.think_tags],
                'model_tags': [{'content': t.content, 'language': t.language} for t in parsed_result.model_tags],
                'requires_tags': [{'content': t.content} for t in parsed_result.requires_tags]
            },
            'execution_results': execution_results,
            'tokens_generated': len(outputs[0]) - inputs.input_ids.shape[1]
        }
    
    async def interactive_session(self, verbose: bool = False):
        """Run interactive WorldModel session."""
        print("🌍 WorldModel Interactive Session")
        print("Features: Structured reasoning • Code execution")
        print("Type 'quit' to exit, 'help' for commands")
        print("=" * 70)
        
        while True:
            try:
                user_input = input("\n👤 You: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    break
                elif user_input.lower() == 'help':
                    print("""
🌍 WorldModel Commands:
  help     - Show this help
  quit     - Exit the session  
  verbose  - Toggle verbose output
  
🏷️ WorldModel Tags:
  <think>   - Model reasoning process
  <model>   - Code to execute
  <requires> - Execution requirements
  
✨ Example prompts:
  "Count the R's in strawberry"
  "Calculate 15% tip on $67.50" 
  "Write Python code to find prime numbers"
                    """)
                    continue
                elif user_input.lower() == 'verbose':
                    verbose = not verbose
                    print(f"Verbose mode: {'ON' if verbose else 'OFF'}")
                    continue
                elif not user_input:
                    continue
                
                # Process the prompt
                result = await self.process_prompt(user_input, verbose=verbose)
                
                # Show thinking if verbose
                if verbose and result['parsed_tags']['think_tags']:
                    print("\n🧠 <think>:")
                    for think_tag in result['parsed_tags']['think_tags']:
                        print(f"   {think_tag['content']}")
                
                # Show main response
                print(f"\n🤖 WorldModel:")
                print(result['generated_text'])
                
                # Show execution results if any
                if result['execution_results']:
                    print(f"\n⚡ Execution Results:")
                    for i, exec_result in enumerate(result['execution_results']):
                        print(f"   Execution {i+1}: {exec_result['status']}")
                        if exec_result.get('stdout'):
                            print(f"      Output: {exec_result['stdout']}")
                        if exec_result.get('stderr'):
                            print(f"      Error: {exec_result['stderr']}")
                
            except KeyboardInterrupt:
                print("\n👋 Session ended.")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
                if verbose:
                    import traceback
                    traceback.print_exc()

async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Simple WorldModel Inference",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('-m', '--model', required=True,
                       help='Path to fine-tuned WorldModel')
    parser.add_argument('-p', '--prompt',
                       help='Single prompt to process')
    parser.add_argument('-i', '--interactive', action='store_true',
                       help='Interactive session')
    parser.add_argument('--no-execution', action='store_true',
                       help='Disable code execution')
    parser.add_argument('-v', '--verbose', action='store_true',
                       help='Show detailed processing')
    
    args = parser.parse_args()
    
    # Initialize WorldModel runner
    runner = SimpleWorldModelRunner(
        model_path=args.model,
        enable_execution=not args.no_execution
    )
    
    if args.prompt:
        # Single prompt mode
        result = await runner.process_prompt(args.prompt, verbose=args.verbose)
        print(f"\n🤖 Response:")
        print(result['generated_text'])
        
        if result['execution_results']:
            print(f"\n⚡ Execution Results:")
            for exec_result in result['execution_results']:
                if exec_result.get('stdout'):
                    print(f"   {exec_result['stdout']}")
    
    elif args.interactive:
        # Interactive mode
        await runner.interactive_session(verbose=args.verbose)
    
    else:
        print("❓ Please specify --prompt or --interactive mode")
        print("Example: python simple_worldmodel.py -m ./model -i")

if __name__ == "__main__":
    asyncio.run(main())