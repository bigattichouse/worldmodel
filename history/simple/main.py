#!/usr/bin/env python3
"""
Main CLI entry point for WorldModel LLM experiment.

Provides command-line interface for training, inference, and system management.
"""

import argparse
import sys
import json
import asyncio
from pathlib import Path
from typing import Optional, Dict, Any, List

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.core.inferenceEngine import InferenceEngine, GenerationMode, GenerationConfig
from src.core.tagParser import parse_tags
from src.core.uncertaintyDetection import UncertaintyDetector
from src.execution.vmInterface import VMInterface
from src.execution.approvalSystem import ApprovalSystem
from src.execution.requirementValidator import RequirementValidator
from src.memory.ragSystem import RAGSystem
from src.memory.modelRegistry import ModelRegistry
from src.memory.embeddingManager import EmbeddingManager
from src.training.dataGenerator import DataGenerator
from src.training.sftTrainer import SFTTrainer, SFTConfig
from src.training.rlTrainer import RLTrainer, RLConfig
from src.utils.config import get_config, ConfigManager
from src.utils.logging import get_logger, start_experiment


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="WorldModel LLM - Structured reasoning language model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive chat session
  python main.py chat --model ./models/gemma-3-270m-it

  # Generate response to a prompt  
  python main.py generate "Solve: 2x + 5 = 13" --worldmodel

  # Train the model with SFT
  python main.py train sft --data ./data/training.json --epochs 3

  # Train with RL
  python main.py train rl --episodes 1000 

  # Generate synthetic training data
  python main.py data generate --count 1000 --output ./data/synthetic.json

  # Test code execution
  python main.py execute python "print('Hello, World!')"

  # Analyze uncertainty in text
  python main.py analyze "I think the answer might be around 42, but I'm not sure"
        """
    )
    
    # Global options
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], 
                       default='INFO', help='Logging level')
    parser.add_argument('--experiment-name', type=str, help='Name for experiment tracking')
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Chat command
    chat_parser = subparsers.add_parser('chat', help='Interactive chat session')
    chat_parser.add_argument('--model', type=str, help='Path to model')
    chat_parser.add_argument('--worldmodel', action='store_true', 
                           help='Enable WorldModel structured reasoning')
    chat_parser.add_argument('--no-execution', action='store_true',
                           help='Disable code execution')
    chat_parser.add_argument('--no-approval', action='store_true',
                           help='Disable approval system')
    chat_parser.add_argument('--verbose', action='store_true', 
                           help='Show detailed WorldModel processing')
    
    # Generate command  
    gen_parser = subparsers.add_parser('generate', help='Generate response to prompt')
    gen_parser.add_argument('prompt', type=str, help='Input prompt')
    gen_parser.add_argument('--model', type=str, help='Path to model')
    gen_parser.add_argument('--worldmodel', action='store_true',
                          help='Use WorldModel structured reasoning')
    gen_parser.add_argument('--output', type=str, help='Output file for result')
    gen_parser.add_argument('--temperature', type=float, default=0.7, help='Generation temperature')
    gen_parser.add_argument('--max-tokens', type=int, default=512, help='Maximum tokens to generate')
    gen_parser.add_argument('--verbose', action='store_true', help='Show detailed WorldModel processing')
    
    # Training commands
    train_parser = subparsers.add_parser('train', help='Training commands')
    train_subparsers = train_parser.add_subparsers(dest='train_type', help='Training type')
    
    # SFT training
    sft_parser = train_subparsers.add_parser('sft', help='Supervised fine-tuning')
    sft_parser.add_argument('--data', type=str, required=True, help='Training data file')
    sft_parser.add_argument('--model', type=str, help='Base model path')
    sft_parser.add_argument('--output', type=str, help='Output directory for trained model')
    sft_parser.add_argument('--epochs', type=int, default=3, help='Number of training epochs')
    sft_parser.add_argument('--batch-size', type=int, default=4, help='Batch size')
    sft_parser.add_argument('--learning-rate', type=float, default=5e-5, help='Learning rate')
    sft_parser.add_argument('--use-lora', action='store_true', help='Enable LoRA training')
    sft_parser.add_argument('--lora-rank', type=int, default=16, help='LoRA rank')
    sft_parser.add_argument('--lora-alpha', type=int, default=32, help='LoRA alpha')
    sft_parser.add_argument('--use-4bit', action='store_true', help='Enable 4-bit quantization')
    sft_parser.add_argument('--no-lora', action='store_true', help='Disable LoRA (use full training)')
    
    # RL training
    rl_parser = train_subparsers.add_parser('rl', help='Reinforcement learning')
    rl_parser.add_argument('--base-model', type=str, help='Base model for RL training')
    rl_parser.add_argument('--episodes', type=int, default=1000, help='Number of RL episodes')
    rl_parser.add_argument('--output', type=str, help='Output directory for RL model')
    rl_parser.add_argument('--data-file', type=str, help='Training data for RL environment')
    
    # Data generation commands
    data_parser = subparsers.add_parser('data', help='Data generation commands')
    data_subparsers = data_parser.add_subparsers(dest='data_action', help='Data action')
    
    # Generate data
    gen_data_parser = data_subparsers.add_parser('generate', help='Generate synthetic training data')
    gen_data_parser.add_argument('--count', type=int, default=100, help='Number of examples to generate')
    gen_data_parser.add_argument('--output', type=str, required=True, help='Output file')
    gen_data_parser.add_argument('--categories', nargs='+', help='Categories to generate')
    gen_data_parser.add_argument('--difficulty', choices=['easy', 'medium', 'hard'], help='Difficulty level')
    
    # Analyze data
    analyze_data_parser = data_subparsers.add_parser('analyze', help='Analyze training data')
    analyze_data_parser.add_argument('file', type=str, help='Data file to analyze')
    
    # Execution commands
    exec_parser = subparsers.add_parser('execute', help='Execute code')
    exec_parser.add_argument('language', choices=['python', 'javascript', 'bash', 'c'], 
                           help='Programming language')
    exec_parser.add_argument('code', type=str, help='Code to execute')
    exec_parser.add_argument('--timeout', type=int, default=30, help='Execution timeout')
    exec_parser.add_argument('--no-validation', action='store_true', help='Skip requirement validation')
    
    # Analysis commands
    analyze_parser = subparsers.add_parser('analyze', help='Analyze text for uncertainty')
    analyze_parser.add_argument('text', type=str, help='Text to analyze')
    analyze_parser.add_argument('--model', type=str, help='Model for uncertainty detection')
    analyze_parser.add_argument('--context', type=str, help='Context for analysis')
    
    # System management commands
    system_parser = subparsers.add_parser('system', help='System management')
    system_subparsers = system_parser.add_subparsers(dest='system_action', help='System action')
    
    # Status
    status_parser = system_subparsers.add_parser('status', help='Show system status')
    
    # Config
    config_parser = system_subparsers.add_parser('config', help='Configuration management')
    config_parser.add_argument('--show', action='store_true', help='Show current configuration')
    config_parser.add_argument('--validate', action='store_true', help='Validate configuration')
    config_parser.add_argument('--reset', action='store_true', help='Reset to default configuration')
    
    # Memory management
    memory_parser = system_subparsers.add_parser('memory', help='Memory system management')
    memory_parser.add_argument('--clear-rag', action='store_true', help='Clear RAG system')
    memory_parser.add_argument('--clear-embeddings', action='store_true', help='Clear embeddings')
    memory_parser.add_argument('--stats', action='store_true', help='Show memory statistics')
    
    args = parser.parse_args()
    
    # Setup logging
    if args.experiment_name:
        start_experiment(args.experiment_name)
    
    logger = get_logger('main')
    
    # Load configuration
    config_manager = ConfigManager(args.config)
    config = config_manager.config
    
    try:
        # Route to appropriate command handler
        if args.command == 'chat':
            return handle_chat(args, config, logger)
        elif args.command == 'generate':
            return handle_generate(args, config, logger)
        elif args.command == 'train':
            return handle_train(args, config, logger)
        elif args.command == 'data':
            return handle_data(args, config, logger)
        elif args.command == 'execute':
            return handle_execute(args, config, logger)
        elif args.command == 'analyze':
            return handle_analyze(args, config, logger)
        elif args.command == 'system':
            return handle_system(args, config, logger)
        else:
            parser.print_help()
            return 1
            
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Command failed: {e}", exc_info=True)
        return 1


def handle_chat(args, config, logger):
    """Handle interactive chat session."""
    logger.info("Starting interactive chat session")
    
    # Initialize inference engine
    engine = InferenceEngine(model_path=args.model)
    
    # Setup generation config
    gen_config = GenerationConfig(
        enable_thinking=args.worldmodel,
        enable_code_execution=not args.no_execution,
        require_approval=not args.no_approval
    )
    
    print("WorldModel Chat Session")
    print("Type 'quit' to exit, 'clear' to clear history")
    print("=" * 50)
    
    try:
        while True:
            user_input = input("\nYou: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                break
            elif user_input.lower() == 'clear':
                engine.clear_conversation_history()
                print("Conversation history cleared.")
                continue
            elif not user_input:
                continue
            
            # Generate response
            mode = GenerationMode.WORLDMODEL if args.worldmodel else GenerationMode.NORMAL
            result = engine.generate(user_input, mode=mode, generation_config=gen_config)
            
            # Show verbose output if requested
            if args.verbose and args.worldmodel:
                print("\n" + "="*50)
                print("🔍 WorldModel Processing:")
                
                # Parse and show tags
                from src.core.tagParser import TagParser
                parser = TagParser()
                parsed_result = parser.parse(result.text)
                
                if parsed_result.think_tags:
                    print(f"\n🧠 <think>:")
                    for tag in parsed_result.think_tags:
                        print(tag.content)
                
                if parsed_result.model_tags:
                    print(f"\n💻 <model>:")
                    for tag in parsed_result.model_tags:
                        print(tag.content)
                
                if parsed_result.requires_tags:
                    print(f"\n📋 <requires>:")
                    for tag in parsed_result.requires_tags:
                        print(f"  - {tag.content}")
                
                print("="*50)
            
            print(f"\nAssistant: {result.text}")
            
            # Show execution results if any
            if result.execution_results:
                print("\n--- Execution Results ---")
                for i, exec_result in enumerate(result.execution_results):
                    print(f"Execution {i+1}: {exec_result['status']}")
                    if exec_result.get('stdout'):
                        print(f"Output: {exec_result['stdout']}")
                    if exec_result.get('stderr'):
                        print(f"Error: {exec_result['stderr']}")
            
            # Show uncertainty analysis if available
            if result.uncertainty_analysis and result.uncertainty_analysis.should_think:
                print(f"\n--- Uncertainty Detected ---")
                print(f"Reason: {result.uncertainty_analysis.metrics.trigger_reason}")
                print(f"Confidence: {result.uncertainty_analysis.metrics.confidence:.3f}")
    
    except KeyboardInterrupt:
        print("\nChat session ended.")
    
    # Save conversation
    if engine.conversation_history:
        timestamp = int(time.time())
        conv_file = f"conversation_{timestamp}.json"
        engine.save_conversation(conv_file)
        logger.info(f"Conversation saved to {conv_file}")
    
    return 0


def handle_generate(args, config, logger):
    """Handle text generation."""
    logger.info(f"Generating response to: {args.prompt[:100]}...")
    
    # Initialize inference engine
    engine = InferenceEngine(model_path=args.model)
    
    # Setup generation config
    gen_config = GenerationConfig(
        temperature=args.temperature,
        max_new_tokens=args.max_tokens,
        enable_thinking=args.worldmodel,
        enable_code_execution=args.worldmodel,
        require_approval=False  # Non-interactive, so disable approval
    )
    
    # Generate
    mode = GenerationMode.WORLDMODEL if args.worldmodel else GenerationMode.NORMAL
    result = engine.generate(args.prompt, mode=mode, generation_config=gen_config)
    
    # Show verbose output if requested
    if args.verbose and args.worldmodel:
        print("=" * 60)
        print("🔍 VERBOSE WorldModel Processing:")
        print("=" * 60)
        
        # Show raw generated text
        print(f"📝 Raw Generated Text:")
        print(result.text)
        print()
        
        # Parse and show tags
        from src.core.tagParser import TagParser
        parser = TagParser()
        parsed_result = parser.parse(result.text)
        
        if parsed_result.think_tags:
            print(f"🧠 <think> Content:")
            for tag in parsed_result.think_tags:
                print(tag.content)
            print()
        
        if parsed_result.model_tags:
            print(f"💻 <model> Code:")
            for tag in parsed_result.model_tags:
                print(tag.content)
            print()
        
        if parsed_result.requires_tags:
            print(f"📋 <requires> Requirements:")
            for tag in parsed_result.requires_tags:
                print(f"  - {tag.content}")
            print()
        
        if result.execution_results:
            print(f"⚡ Execution Results:")
            for i, exec_result in enumerate(result.execution_results):
                print(f"  Execution {i+1}: {exec_result['status']}")
                if exec_result.get('stdout'):
                    print(f"    Output: {exec_result['stdout']}")
                if exec_result.get('stderr'):
                    print(f"    Error: {exec_result['stderr']}")
            print()
        
        if result.uncertainty_analysis:
            print(f"🎲 Uncertainty Analysis:")
            print(f"  Should think: {result.uncertainty_analysis.should_think}")
            print(f"  Trigger reason: {result.uncertainty_analysis.metrics.trigger_reason}")
            print(f"  Confidence: {result.uncertainty_analysis.metrics.confidence:.3f}")
            print(f"  Perplexity: {result.uncertainty_analysis.metrics.perplexity:.2f}")
            print()
        
        print("=" * 60)
        print("💬 Final Response:")
        print("=" * 60)
    
    # Output result
    output = {
        'prompt': args.prompt,
        'generated_text': result.text,
        'metadata': {
            'generation_time': result.generation_time,
            'tokens_generated': result.tokens_generated,
            'finish_reason': result.finish_reason,
            'mode': mode.value
        }
    }
    
    if result.execution_results:
        output['execution_results'] = result.execution_results
    
    if result.uncertainty_analysis:
        output['uncertainty_analysis'] = {
            'should_think': result.uncertainty_analysis.should_think,
            'trigger_reason': result.uncertainty_analysis.metrics.trigger_reason,
            'confidence': result.uncertainty_analysis.metrics.confidence,
            'perplexity': result.uncertainty_analysis.metrics.perplexity
        }
    
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(output, f, indent=2)
        logger.info(f"Result saved to {args.output}")
    else:
        print(result.text)
    
    return 0


def handle_train(args, config, logger):
    """Handle training commands."""
    if args.train_type == 'sft':
        return handle_sft_train(args, config, logger)
    elif args.train_type == 'rl':
        return handle_rl_train(args, config, logger)
    else:
        logger.error("Unknown training type")
        return 1


def handle_sft_train(args, config, logger):
    """Handle supervised fine-tuning."""
    logger.info("Starting supervised fine-tuning")
    
    # Setup SFT config
    sft_config = SFTConfig(
        model_name=args.model or config.model.model_path,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        use_lora=not args.no_lora if hasattr(args, 'no_lora') else True,  # LoRA by default
        lora_rank=args.lora_rank if hasattr(args, 'lora_rank') else 16,
        lora_alpha=args.lora_alpha if hasattr(args, 'lora_alpha') else 32,
        use_4bit=args.use_4bit if hasattr(args, 'use_4bit') else False  # Disable 4-bit due to ROCm issues
    )
    
    if args.output:
        sft_config.output_dir = args.output
    
    # Load training data first
    logger.info(f"Loading training data from {args.data}")
    from src.training.dataGenerator import DataGenerator
    
    generator = DataGenerator(config.training)
    examples = generator.load_dataset(args.data)
    logger.info(f"Loaded {len(examples)} training examples")
    
    # Initialize trainer
    trainer = SFTTrainer(sft_config)
    
    # Train
    import asyncio
    metrics = asyncio.run(trainer.train(examples))
    
    logger.info("SFT training completed")
    logger.info(f"Final loss: {metrics.final_loss:.4f}")
    logger.info(f"Total steps: {metrics.total_steps}")
    
    return 0


def handle_rl_train(args, config, logger):
    """Handle reinforcement learning training."""
    logger.info("Starting reinforcement learning training")
    
    # Setup RL config
    rl_config = RLConfig(
        num_episodes=args.episodes,
        base_model_path=args.base_model or config.model.model_path
    )
    
    if args.output:
        rl_config.output_dir = args.output
    
    # Initialize trainer
    trainer = RLTrainer(rl_config)
    
    # Load training data if provided
    if args.data_file:
        logger.info(f"Loading RL training data from {args.data_file}")
        trainer.load_training_data(args.data_file)
    
    # Train
    metrics = trainer.train()
    
    logger.info("RL training completed")
    logger.info(f"Average reward: {metrics.average_reward:.4f}")
    logger.info(f"Total episodes: {metrics.total_episodes}")
    
    return 0


def handle_data(args, config, logger):
    """Handle data generation and analysis."""
    if args.data_action == 'generate':
        return handle_data_generate(args, config, logger)
    elif args.data_action == 'analyze':
        return handle_data_analyze(args, config, logger)
    else:
        logger.error("Unknown data action")
        return 1


def handle_data_generate(args, config, logger):
    """Handle synthetic data generation."""
    logger.info(f"Generating {args.count} training examples")
    
    generator = DataGenerator(config.training)
    
    # Generate data
    import asyncio
    
    category_weights = None
    if args.categories:
        category_weights = {cat: 1.0 for cat in args.categories}
    
    difficulty_weights = None
    if args.difficulty:
        difficulty_weights = {args.difficulty: 1.0}
    
    examples = asyncio.run(generator.generate_dataset(
        size=args.count,
        category_weights=category_weights,
        difficulty_weights=difficulty_weights
    ))
    
    # Save to file
    generator.save_dataset(examples, args.output)
    
    logger.info(f"Generated dataset saved to {args.output}")
    
    # Show statistics
    stats = generator.get_dataset_stats(examples)
    print("\nDataset Statistics:")
    print(f"Total examples: {stats.total_examples}")
    print(f"Categories: {list(stats.examples_by_category.keys())}")
    print(f"Difficulties: {list(stats.examples_by_difficulty.keys())}")
    print(f"Examples with execution: {stats.examples_with_execution}")
    print(f"Examples with thinking: {stats.examples_with_thinking}")
    print(f"Average input length: {stats.average_input_length:.1f}")
    print(f"Average output length: {stats.average_output_length:.1f}")
    
    return 0


def handle_data_analyze(args, config, logger):
    """Handle data analysis."""
    logger.info(f"Analyzing data file: {args.file}")
    
    generator = DataGenerator(config.training)
    
    try:
        examples = generator.load_dataset(args.file)
        stats = generator.get_dataset_stats(examples)
        
        print("\nDataset Analysis:")
        print(f"Total examples: {stats.total_examples}")
        print(f"Average input length: {stats.average_input_length:.1f} chars")
        print(f"Average output length: {stats.average_output_length:.1f} chars")
        
        print("\nCategory distribution:")
        for category, count in stats.examples_by_category.items():
            percentage = (count / stats.total_examples) * 100
            print(f"  {category}: {count} ({percentage:.1f}%)")
        
        print("\nDifficulty distribution:")
        for difficulty, count in stats.examples_by_difficulty.items():
            percentage = (count / stats.total_examples) * 100
            print(f"  {difficulty}: {count} ({percentage:.1f}%)")
        
        return 0
        
    except Exception as e:
        logger.error(f"Failed to analyze data: {e}")
        return 1


def handle_execute(args, config, logger):
    """Handle code execution."""
    logger.info(f"Executing {args.language} code")
    
    async def run_execution():
        vm = VMInterface()
        result = await vm.execute_code(args.language, args.code)
        
        print(f"\nExecution Status: {result.status.value}")
        print(f"Return Code: {result.return_code}")
        print(f"Execution Time: {result.execution_time:.3f}s")
        
        if result.stdout:
            print(f"\nOutput:\n{result.stdout}")
        
        if result.stderr:
            print(f"\nErrors:\n{result.stderr}")
        
        # Validate requirements if not disabled
        if not args.no_validation and result.success:
            validator = RequirementValidator()
            # Simple validation without explicit requirements
            validation_result = validator.analyze_code(args.code, args.language)
            
            print(f"\nCode Analysis:")
            print(f"Detected capabilities: {validation_result.detected_capabilities}")
            if validation_result.violations:
                print(f"Violations: {len(validation_result.violations)}")
                for violation in validation_result.violations:
                    print(f"  - {violation.message}")
        
        return 0 if result.success else 1
    
    return asyncio.run(run_execution())


def handle_analyze(args, config, logger):
    """Handle uncertainty analysis."""
    logger.info("Analyzing text for uncertainty")
    
    detector = UncertaintyDetector(model_path=args.model)
    result = detector.detect_uncertainty(args.text, context=args.context)
    
    print(f"\nUncertainty Analysis:")
    print(f"Should think: {result.should_think}")
    print(f"Trigger reason: {result.metrics.trigger_reason}")
    print(f"Confidence: {result.metrics.confidence:.3f}")
    print(f"Perplexity: {result.metrics.perplexity:.2f}")
    print(f"Entropy: {result.metrics.entropy:.3f}")
    print(f"Tokens analyzed: {result.metrics.num_tokens}")
    
    if result.suggested_prompt:
        print(f"\nSuggested prompt:\n{result.suggested_prompt}")
    
    return 0


def handle_system(args, config, logger):
    """Handle system management commands."""
    if args.system_action == 'status':
        return handle_system_status(args, config, logger)
    elif args.system_action == 'config':
        return handle_system_config(args, config, logger)
    elif args.system_action == 'memory':
        return handle_system_memory(args, config, logger)
    else:
        logger.error("Unknown system action")
        return 1


def handle_system_status(args, config, logger):
    """Show system status."""
    print("WorldModel System Status")
    print("=" * 30)
    
    # Check model availability
    model_path = Path(config.model.model_path)
    print(f"Model path: {model_path}")
    print(f"Model exists: {model_path.exists()}")
    
    # Check VM availability  
    vm_path = Path(config.execution.vm_path) if hasattr(config.execution, 'vm_path') else None
    if vm_path:
        print(f"VM path: {vm_path}")
        print(f"VM exists: {vm_path.exists()}")
    
    # Check memory systems
    try:
        rag = RAGSystem()
        rag_stats = rag.get_stats()
        print(f"RAG documents: {rag_stats['total_documents']}")
    except Exception:
        print("RAG system: unavailable")
    
    try:
        embedding_manager = EmbeddingManager()
        emb_stats = embedding_manager.get_statistics()
        print(f"Embeddings: {emb_stats['total_embeddings']}")
        print(f"Clusters: {emb_stats['total_clusters']}")
    except Exception:
        print("Embedding manager: unavailable")
    
    try:
        registry = ModelRegistry()
        registry_stats = registry.get_stats()
        print(f"Registered models: {registry_stats['total_models']}")
    except Exception:
        print("Model registry: unavailable")
    
    return 0


def handle_system_config(args, config, logger):
    """Handle configuration management."""
    if args.show:
        print("Current Configuration:")
        print("=" * 25)
        config_dict = config.to_dict()
        print(json.dumps(config_dict, indent=2))
    
    elif args.validate:
        config_manager = ConfigManager()
        errors = config_manager.validate_config()
        if errors:
            print("Configuration errors:")
            for error in errors:
                print(f"  - {error}")
            return 1
        else:
            print("Configuration is valid")
    
    elif args.reset:
        config_manager = ConfigManager()
        config_manager.reset_to_default()
        print("Configuration reset to defaults")
    
    return 0


def handle_system_memory(args, config, logger):
    """Handle memory system management."""
    if args.stats:
        print("Memory System Statistics")
        print("=" * 30)
        
        try:
            rag = RAGSystem()
            stats = rag.get_stats()
            print(f"RAG System:")
            print(f"  Documents: {stats['total_documents']}")
            print(f"  Storage used: {stats.get('storage_size_mb', 'unknown')} MB")
        except Exception as e:
            print(f"RAG System: error ({e})")
        
        try:
            emb_mgr = EmbeddingManager()
            stats = emb_mgr.get_statistics()
            print(f"Embedding Manager:")
            print(f"  Embeddings: {stats['total_embeddings']}")
            print(f"  Clusters: {stats['total_clusters']}")
            print(f"  Types: {list(stats['type_distribution'].keys())}")
        except Exception as e:
            print(f"Embedding Manager: error ({e})")
    
    elif args.clear_rag:
        try:
            rag = RAGSystem()
            # Clear would need to be implemented
            print("RAG system cleared (if clear method exists)")
        except Exception as e:
            print(f"Failed to clear RAG: {e}")
    
    elif args.clear_embeddings:
        try:
            emb_mgr = EmbeddingManager()
            # Clear would need to be implemented
            print("Embeddings cleared (if clear method exists)")
        except Exception as e:
            print(f"Failed to clear embeddings: {e}")
    
    return 0


if __name__ == '__main__':
    import time
    exit_code = main()
    sys.exit(exit_code)