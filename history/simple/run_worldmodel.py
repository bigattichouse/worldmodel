#!/usr/bin/env python3
"""
Proper WorldModel inference script using the original system architecture.
This integrates with the full WorldModel pipeline including code execution,
tag parsing, and structured reasoning.
"""

import os
import sys
import argparse
from pathlib import Path

# Add src to Python path
sys.path.append(str(Path(__file__).parent / "src"))

from src.core.inferenceEngine import InferenceEngine, GenerationMode, GenerationConfig
from src.core.tagParser import TagParser
from src.core.uncertaintyDetection import UncertaintyDetector
from src.execution.vmInterface import VMInterface
from src.execution.approvalSystem import ApprovalSystem
from src.execution.requirementValidator import RequirementValidator
from src.memory.ragSystem import RAGSystem
from src.utils.config import ConfigManager
from src.utils.logging import get_logger


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


class WorldModelRunner:
    """Runner for the complete WorldModel system."""
    
    def __init__(self, model_path: str, base_model_path: str = None, 
                 enable_execution: bool = True, require_approval: bool = True):
        """Initialize WorldModel runner."""
        setup_rocm_environment()
        
        # Determine actual model path for inference engine
        model_dir = Path(model_path)
        if (model_dir / "adapter_config.json").exists():
            # LoRA adapter - need base model
            if not base_model_path:
                base_model_path = "../model/Qwen3-0.6B"
            # For InferenceEngine, we'll need to handle LoRA loading
            actual_model_path = base_model_path
            self.is_lora = True
            self.adapter_path = str(model_dir)
        else:
            # Merged model or base model
            actual_model_path = str(model_dir)
            self.is_lora = False
            self.adapter_path = None
        
        # Initialize core systems
        self.config_manager = ConfigManager()
        self.logger = get_logger('worldmodel_runner')
        
        # Initialize inference engine with proper model path
        self.inference_engine = InferenceEngine(model_path=actual_model_path)
        
        # Load LoRA adapter if needed
        if self.is_lora:
            self._load_lora_adapter()
        
        # Initialize supporting systems
        self.tag_parser = TagParser()
        self.uncertainty_detector = UncertaintyDetector()
        self.vm_interface = VMInterface() if enable_execution else None
        if require_approval:
            from src.utils.config import ApprovalConfig
            approval_config = ApprovalConfig(
                require_user_approval=True,
                auto_approve_low_risk=True
            )
            self.approval_system = ApprovalSystem(approval_config)
        else:
            self.approval_system = None
        self.requirement_validator = RequirementValidator()
        
        # Initialize RAG system
        try:
            self.rag_system = RAGSystem()
        except Exception as e:
            self.logger.warning(f"RAG system unavailable: {e}")
            self.rag_system = None
        
        self.logger.info("WorldModel runner initialized")
        
    def _load_lora_adapter(self):
        """Load LoRA adapter into the inference engine."""
        try:
            from peft import PeftModel
            # Replace the model with LoRA version
            base_model = self.inference_engine.model
            self.inference_engine.model = PeftModel.from_pretrained(
                base_model, self.adapter_path
            )
            self.logger.info(f"LoRA adapter loaded from {self.adapter_path}")
        except Exception as e:
            self.logger.error(f"Failed to load LoRA adapter: {e}")
            raise
    
    async def process_prompt(self, prompt: str, verbose: bool = False) -> dict:
        """
        Process a prompt through the full WorldModel pipeline.
        
        Returns:
            dict: Complete result including generated text, execution results, etc.
        """
        self.logger.info(f"Processing prompt: {prompt[:100]}...")
        
        # Setup generation config for WorldModel
        gen_config = GenerationConfig(
            enable_thinking=True,
            enable_code_execution=self.vm_interface is not None,
            enable_rag_retrieval=self.rag_system is not None,
            require_approval=self.approval_system is not None,
            temperature=0.8,
            max_new_tokens=1024,
            do_sample=True,
            top_p=0.9,
            repetition_penalty=1.1
        )
        
        # Generate using WorldModel mode
        import time
        start_time = time.time()
        result = self.inference_engine.generate(
            prompt, 
            mode=GenerationMode.WORLDMODEL, 
            generation_config=gen_config
        )
        generation_time = time.time() - start_time
        
        # Parse the generated text for tags
        parsed_result = self.tag_parser.parse(result.text)
        
        # Process execution if model tags are present
        execution_results = []
        if parsed_result.model_tags and self.vm_interface:
            for i, model_tag in enumerate(parsed_result.model_tags):
                if verbose:
                    print(f"\n💻 Executing code block {i+1}:")
                    print(model_tag.content)
                
                # Note: Requirement validation will be done after execution
                # to provide proper ExecutionResult for validation
                
                # Request approval if needed
                if self.approval_system:
                    approved = await self.approval_system.request_approval(
                        model_tag.content, 
                        model_tag.language or 'python'
                    )
                    if not approved:
                        execution_results.append({
                            'status': 'approval_denied', 
                            'code': model_tag.content,
                            'language': model_tag.language
                        })
                        continue
                
                # Execute the code
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
        
        # Analyze uncertainty
        uncertainty_analysis = None
        if self.uncertainty_detector:
            try:
                uncertainty_analysis = self.uncertainty_detector.detect_uncertainty(
                    result.text, context=prompt
                )
            except Exception as e:
                self.logger.warning(f"Uncertainty analysis failed: {e}")
        
        # Return comprehensive result
        return {
            'prompt': prompt,
            'generated_text': result.text,
            'parsed_tags': {
                'think_tags': [{'content': t.content} for t in parsed_result.think_tags],
                'model_tags': [{'content': t.content, 'language': t.language} for t in parsed_result.model_tags],
                'requires_tags': [{'content': t.content} for t in parsed_result.requires_tags]
            },
            'execution_results': execution_results,
            'uncertainty_analysis': {
                'should_think': uncertainty_analysis.should_think,
                'confidence': uncertainty_analysis.metrics.confidence,
                'trigger_reason': uncertainty_analysis.metrics.trigger_reason
            } if uncertainty_analysis else None,
            'generation_time': generation_time,
            'tokens_generated': result.tokens_generated,
            'finish_reason': result.finish_reason
        }
    
    async def interactive_session(self, verbose: bool = False):
        """Run interactive WorldModel session."""
        print("🌍 WorldModel Interactive Session")
        print("Features: Structured reasoning • Code execution • Requirement validation")
        print("Type 'quit' to exit, 'help' for commands")
        print("=" * 70)
        
        while True:
            try:
                user_input = input("\n👤 You: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    break
                elif user_input.lower() == 'help':
                    self._show_help()
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
                
                # Show code if any
                if result['parsed_tags']['model_tags']:
                    print(f"\n💻 Generated code:")
                    for model_tag in result['parsed_tags']['model_tags']:
                        lang = model_tag['language'] or 'python'
                        print(f"   Language: {lang}")
                        print(f"   Code: {model_tag['content']}")
                
                # Show requirements if any
                if result['parsed_tags']['requires_tags']:
                    print(f"\n📋 Requirements:")
                    for req_tag in result['parsed_tags']['requires_tags']:
                        print(f"   - {req_tag['content']}")
                
                # Show main response
                print(f"\n🤖 WorldModel: {result['generated_text']}")
                
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
    
    def _show_help(self):
        """Show help information."""
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


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="WorldModel Inference with Full Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('-m', '--model', required=True,
                       help='Path to fine-tuned WorldModel')
    parser.add_argument('-b', '--base-model',
                       help='Path to base model (for LoRA adapters)')
    parser.add_argument('-p', '--prompt',
                       help='Single prompt to process')
    parser.add_argument('-i', '--interactive', action='store_true',
                       help='Interactive session')
    parser.add_argument('--no-execution', action='store_true',
                       help='Disable code execution')
    parser.add_argument('--no-approval', action='store_true', 
                       help='Disable approval system')
    parser.add_argument('-v', '--verbose', action='store_true',
                       help='Show detailed processing')
    parser.add_argument('-o', '--output',
                       help='Output file for results')
    
    args = parser.parse_args()
    
    # Initialize WorldModel runner
    print("🔄 Initializing WorldModel...")
    runner = WorldModelRunner(
        model_path=args.model,
        base_model_path=args.base_model,
        enable_execution=not args.no_execution,
        require_approval=not args.no_approval
    )
    print("✅ WorldModel ready!")
    
    if args.prompt:
        # Single prompt mode
        print(f"\n📝 Processing: {args.prompt}")
        result = await runner.process_prompt(args.prompt, verbose=args.verbose)
        
        if args.output:
            import json
            with open(args.output, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"💾 Results saved to {args.output}")
        else:
            print(f"\n🤖 Response: {result['generated_text']}")
            
            if result['execution_results']:
                print(f"\n⚡ Execution Results:")
                for exec_result in result['execution_results']:
                    print(f"   {exec_result['status']}: {exec_result.get('stdout', 'No output')}")
    
    elif args.interactive:
        # Interactive mode
        await runner.interactive_session(verbose=args.verbose)
    
    else:
        print("❓ Please specify --prompt or --interactive mode")
        print("Example: python run_worldmodel.py -m ./model -i")


if __name__ == "__main__":
    import time
    import asyncio
    asyncio.run(main())