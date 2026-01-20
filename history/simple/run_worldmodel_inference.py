#!/usr/bin/env python3
"""
WorldModel Inference Engine
===========================

Complete inference system that:
1. Loads trained WorldModel
2. Generates structured responses with <think>, <model>, <requires>
3. Executes the generated code safely
4. Returns results

Usage:
    python3 run_worldmodel_inference.py "Calculate 25% of 400"
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import re
import subprocess
import tempfile
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import argparse

class WorldModelInference:
    """Complete WorldModel inference system."""
    
    def __init__(self, model_path: str):
        """Initialize with trained model."""
        print(f"Loading WorldModel from: {model_path}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.float32,
            device_map={"": 0} if torch.cuda.is_available() else None,
            low_cpu_mem_usage=True
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        print(f"✅ Model loaded successfully")
        print(f"   Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    
    def generate_response(self, user_prompt: str, max_tokens: int = 300) -> str:
        """Generate WorldModel response."""
        # Format prompt
        formatted_prompt = f"<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n"
        
        # Tokenize
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        # Generate
        print("🤖 Generating response...")
        self.model.eval()
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.1
            )
        
        # Decode
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract just the assistant part
        assistant_start = formatted_prompt
        if assistant_start in response:
            generated_text = response[len(assistant_start):].strip()
        else:
            generated_text = response.strip()
        
        return generated_text
    
    def parse_worldmodel_response(self, response: str) -> Dict[str, str]:
        """Parse WorldModel structured response."""
        parsed = {
            'thinking': '',
            'code': '',
            'requires': '',
            'explanation': ''
        }
        
        # Extract <think> content
        think_match = re.search(r'<think>(.*?)</think>', response, re.DOTALL)
        if think_match:
            parsed['thinking'] = think_match.group(1).strip()
        
        # Extract <model> content
        model_match = re.search(r'<model>(.*?)</model>', response, re.DOTALL)
        if model_match:
            parsed['code'] = model_match.group(1).strip()
        
        # Extract <requires> content
        requires_match = re.search(r'<requires>(.*?)</requires>', response, re.DOTALL)
        if requires_match:
            parsed['requires'] = requires_match.group(1).strip()
        
        # Extract explanation (everything after </requires>)
        explanation_match = re.search(r'</requires>\s*(.*)', response, re.DOTALL)
        if explanation_match:
            parsed['explanation'] = explanation_match.group(1).strip()
        elif not any([think_match, model_match, requires_match]):
            # If no structured format, treat whole response as explanation
            parsed['explanation'] = response.strip()
        
        return parsed
    
    def execute_code(self, code: str, language: str = "python") -> Dict[str, str]:
        """Safely execute generated code."""
        if not code.strip():
            return {
                'status': 'error',
                'stdout': '',
                'stderr': 'No code to execute',
                'execution_time': 0
            }
        
        # Create temporary file
        if language == "python":
            suffix = ".py"
            cmd_prefix = ["python3"]
        else:
            return {
                'status': 'error',
                'stdout': '',
                'stderr': f'Unsupported language: {language}',
                'execution_time': 0
            }
        
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix=suffix, delete=False) as f:
                f.write(code)
                temp_file = f.name
            
            print(f"💻 Executing code...")
            start_time = time.time()
            
            # Execute with timeout
            result = subprocess.run(
                cmd_prefix + [temp_file],
                capture_output=True,
                text=True,
                timeout=30,  # 30 second timeout
                cwd=tempfile.gettempdir()  # Safe working directory
            )
            
            execution_time = time.time() - start_time
            
            return {
                'status': 'success' if result.returncode == 0 else 'error',
                'stdout': result.stdout.strip(),
                'stderr': result.stderr.strip(),
                'execution_time': execution_time,
                'return_code': result.returncode
            }
        
        except subprocess.TimeoutExpired:
            return {
                'status': 'timeout',
                'stdout': '',
                'stderr': 'Code execution timed out (30s limit)',
                'execution_time': 30
            }
        except Exception as e:
            return {
                'status': 'error',
                'stdout': '',
                'stderr': f'Execution error: {str(e)}',
                'execution_time': 0
            }
        finally:
            # Clean up temp file
            try:
                os.unlink(temp_file)
            except:
                pass
    
    def process_query(self, user_query: str) -> Dict:
        """Complete WorldModel processing pipeline."""
        print(f"\n🔍 Processing: {user_query}")
        print("=" * 60)
        
        # Generate response
        raw_response = self.generate_response(user_query)
        print(f"📝 Raw response: {raw_response[:100]}...")
        
        # Parse structured response
        parsed = self.parse_worldmodel_response(raw_response)
        
        # Show parsed components
        print(f"\n🧠 Thinking: {parsed['thinking'][:80]}..." if parsed['thinking'] else "🧠 Thinking: (none)")
        print(f"💻 Code: {len(parsed['code'])} characters" if parsed['code'] else "💻 Code: (none)")
        print(f"📋 Requires: {parsed['requires']}" if parsed['requires'] else "📋 Requires: (none)")
        
        # Execute code if present
        execution_result = None
        if parsed['code']:
            execution_result = self.execute_code(parsed['code'])
            
            print(f"\n⚡ Execution Status: {execution_result['status']}")
            if execution_result['stdout']:
                print(f"📤 Output: {execution_result['stdout']}")
            if execution_result['stderr']:
                print(f"❌ Error: {execution_result['stderr']}")
            print(f"⏱️  Time: {execution_result['execution_time']:.3f}s")
        
        # Final explanation
        if parsed['explanation']:
            print(f"\n💬 Explanation: {parsed['explanation']}")
        
        return {
            'query': user_query,
            'raw_response': raw_response,
            'parsed': parsed,
            'execution': execution_result
        }

def main():
    """Main CLI interface."""
    parser = argparse.ArgumentParser(
        description="WorldModel Inference Engine with secure code execution",
        epilog="By default, code runs in isolated QEMU VMs for security. Use --no-sandbox to disable."
    )
    parser.add_argument('query', nargs='?', help='Query to process')
    parser.add_argument('--model', default='./worldmodel_rocm_output/final_model', 
                       help='Path to trained model')
    parser.add_argument('--interactive', action='store_true', 
                       help='Start interactive session')
    parser.add_argument('--no-sandbox', action='store_true', 
                       help='Disable QEMU sandbox (use direct execution - less secure)')
    parser.add_argument('--sandbox-memory', default='512M',
                       help='Memory allocation for sandbox VM (default: 512M)')
    parser.add_argument('--sandbox-timeout', type=int, default=30,
                       help='Sandbox execution timeout in seconds (default: 30)')
    
    args = parser.parse_args()
    
    # Determine sandbox mode (default: enabled)
    use_sandbox = not args.no_sandbox
    
    if use_sandbox:
        print("🚀 WorldModel Inference Engine (Secure Mode)")
    else:
        print("🚀 WorldModel Inference Engine (Direct Mode)")
    print("=" * 50)
    
    # Check if model exists
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        print(f"Available models:")
        for model_dir in Path('.').glob('worldmodel_**/final_model'):
            print(f"  {model_dir}")
        return 1
    
    # Initialize inference engine (with or without sandbox)
    try:
        if use_sandbox:
            # Try to import and use sandbox
            try:
                import sys
                sys.path.insert(0, str(Path(__file__).parent / "sandbox" / "src"))
                from worldmodel_sandbox import SandboxedWorldModelInference
                
                sandbox_config = {
                    'vm_name': 'worldmodel-inference',
                    'memory': args.sandbox_memory,
                    'timeout': args.sandbox_timeout,
                    'persistent': False
                }
                
                engine = SandboxedWorldModelInference(str(model_path), sandbox_config)
                print(f"🔒 Sandbox enabled: VM memory={args.sandbox_memory}, timeout={args.sandbox_timeout}s")
                print("   Use --no-sandbox to disable secure execution")
                
            except ImportError as e:
                print(f"⚠️  Sandbox not available: {e}")
                print("   To enable secure execution, run: ./sandbox/setup.sh")
                print("   Falling back to direct execution...")
                engine = WorldModelInference(str(model_path))
        else:
            print("⚠️  Running in direct execution mode (less secure)")
            print("   Remove --no-sandbox flag for secure execution")
            engine = WorldModelInference(str(model_path))
            
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return 1
    
    # Interactive mode
    if args.interactive or not args.query:
        print(f"\n💬 Interactive WorldModel Session")
        print(f"Type 'quit' to exit, 'help' for examples")
        print("=" * 50)
        
        example_queries = [
            "Calculate 15% of 200",
            "What is 12 × 7?",
            "Count the vowels in 'hello world'",
            "Find the area of a circle with radius 10",
            "Is 97 a prime number?"
        ]
        
        while True:
            try:
                query = input("\n🤔 Your question: ").strip()
                
                if query.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                elif query.lower() == 'help':
                    print(f"\n📚 Example queries:")
                    for i, example in enumerate(example_queries, 1):
                        print(f"  {i}. {example}")
                    continue
                elif not query:
                    continue
                
                result = engine.process_query(query)
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
    
    # Single query mode
    else:
        result = engine.process_query(args.query)
        
        # Output as JSON if desired
        # print(json.dumps(result, indent=2))
    
    return 0

if __name__ == "__main__":
    sys.exit(main())