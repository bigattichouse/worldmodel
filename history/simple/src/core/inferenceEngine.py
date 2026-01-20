"""
Inference Engine for WorldModel LLM experiment.

Handles model loading, tokenization, generation, and structured output parsing.
Core component that orchestrates the WorldModel structured reasoning flow.
"""

import torch
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, 
    GenerationConfig, StoppingCriteria, StoppingCriteriaList
)
from typing import Dict, List, Any, Optional, Union, Tuple, Generator
from dataclasses import dataclass
from enum import Enum
import asyncio
import json
import time
import re

from .tagParser import TagParser, ParseResult, TagType
from .uncertaintyDetection import UncertaintyDetector, DetectionResult
from ..execution.vmInterface import VMInterface
from ..execution.requirementValidator import RequirementValidator
from ..execution.approvalSystem import ApprovalSystem
from ..memory.ragSystem import RAGSystem
from ..utils.config import get_config
from ..utils.logging import get_logger

logger = get_logger('inferenceEngine')


class GenerationMode(Enum):
    """Generation modes for the inference engine."""
    NORMAL = "normal"        # Standard text generation
    WORLDMODEL = "worldmodel"  # WorldModel structured reasoning
    INTERACTIVE = "interactive"  # Interactive mode with user feedback


@dataclass
class GenerationConfig:
    """Configuration for text generation."""
    max_length: int = 2048
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.1
    do_sample: bool = True
    num_return_sequences: int = 1
    early_stopping: bool = True
    
    # WorldModel specific
    enable_thinking: bool = True
    enable_code_execution: bool = True
    enable_rag_retrieval: bool = True
    require_approval: bool = True


@dataclass
class GenerationResult:
    """Result of text generation."""
    text: str
    generation_time: float
    tokens_generated: int
    finish_reason: str
    
    # WorldModel specific results
    parsed_tags: Optional[ParseResult] = None
    execution_results: List[Dict[str, Any]] = None
    uncertainty_analysis: Optional[DetectionResult] = None
    rag_context: List[str] = None
    
    def __post_init__(self):
        if self.execution_results is None:
            self.execution_results = []
        if self.rag_context is None:
            self.rag_context = []


class WorldModelStoppingCriteria(StoppingCriteria):
    """Custom stopping criteria for WorldModel generation."""
    
    def __init__(self, tokenizer, stop_sequences: List[str] = None):
        self.tokenizer = tokenizer
        self.stop_sequences = stop_sequences or ["</model>", "</think>", "</requires>"]
        # Encode stop sequences as complete sequences
        self.stop_token_sequences = []
        for seq in self.stop_sequences:
            tokens = tokenizer.encode(seq, add_special_tokens=False)
            if tokens:
                self.stop_token_sequences.append(tokens)
    
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        # Check if any complete stop sequence appears at the end of generated text
        if len(input_ids[0]) == 0:
            return False
            
        generated_tokens = input_ids[0].tolist()
        
        for stop_sequence in self.stop_token_sequences:
            if len(generated_tokens) >= len(stop_sequence):
                # Check if the stop sequence matches the end of generated tokens
                if generated_tokens[-len(stop_sequence):] == stop_sequence:
                    return True
        return False


class InferenceEngine:
    """Main inference engine for WorldModel LLM."""
    
    def __init__(self, model_path: Optional[str] = None, tokenizer_path: Optional[str] = None):
        self.config = get_config()
        self.logger = get_logger('inferenceEngine')
        
        # Model and tokenizer paths
        self.model_path = model_path or self.config.model.model_path
        self.tokenizer_path = tokenizer_path or self.config.model.model_path
        
        # Core components
        self.model = None
        self.tokenizer = None
        self.device = torch.device(self.config.model.device if torch.cuda.is_available() else 'cpu')
        
        # WorldModel components
        self.tag_parser = TagParser()
        self.uncertainty_detector = None  # Lazy init to avoid circular deps
        self.vm_interface = None
        self.requirement_validator = None
        self.approval_system = None
        self.rag_system = None
        
        # Generation state
        self.conversation_history = []
        self.current_mode = GenerationMode.NORMAL
        
        # Initialize model
        self._initialize_model()
        self._initialize_worldmodel_components()
    
    def _initialize_model(self):
        """Initialize the language model and tokenizer."""
        try:
            self.logger.info(f"Loading tokenizer from {self.tokenizer_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)
            
            # Ensure pad token exists
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.logger.info(f"Loading model from {self.model_path}")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=getattr(torch, self.config.model.torch_dtype),
                device_map='auto' if torch.cuda.is_available() else None,
                low_cpu_mem_usage=True
            )
            self.model.eval()
            
            # Move to device if not using device_map
            if not torch.cuda.is_available():
                self.model = self.model.to(self.device)
            
            self.logger.info(f"Model loaded successfully on {self.device}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize model: {e}")
            raise
    
    def _initialize_worldmodel_components(self):
        """Initialize WorldModel-specific components."""
        try:
            # Lazy initialization to avoid circular dependencies
            self.uncertainty_detector = UncertaintyDetector(
                model_path=self.model_path,
                tokenizer_path=self.tokenizer_path
            )
            
            self.vm_interface = VMInterface()
            self.requirement_validator = RequirementValidator()
            self.approval_system = ApprovalSystem(self.config.approval)
            self.rag_system = RAGSystem(self.config.memory)
            
            self.logger.info("WorldModel components initialized successfully")
            
        except Exception as e:
            self.logger.warning(f"Some WorldModel components failed to initialize: {e}")
            # Continue without these components
    
    def generate(self, prompt: str, mode: GenerationMode = GenerationMode.NORMAL,
                generation_config: Optional[GenerationConfig] = None) -> GenerationResult:
        """
        Generate text based on the prompt and mode.
        
        Args:
            prompt: Input prompt for generation
            mode: Generation mode (normal, worldmodel, interactive)
            generation_config: Optional generation configuration
            
        Returns:
            GenerationResult with generated text and metadata
        """
        config = generation_config or GenerationConfig()
        self.current_mode = mode
        
        self.logger.info(f"Starting generation in {mode.value} mode")
        start_time = time.time()
        
        if mode == GenerationMode.WORLDMODEL:
            result = self._generate_worldmodel(prompt, config)
        elif mode == GenerationMode.INTERACTIVE:
            result = self._generate_interactive(prompt, config)
        else:
            result = self._generate_normal(prompt, config)
        
        generation_time = time.time() - start_time
        result.generation_time = generation_time
        
        # Add to conversation history
        self.conversation_history.append({
            'role': 'user',
            'content': prompt,
            'timestamp': time.time()
        })
        self.conversation_history.append({
            'role': 'assistant', 
            'content': result.text,
            'timestamp': time.time(),
            'metadata': {
                'mode': mode.value,
                'generation_time': generation_time,
                'finish_reason': result.finish_reason
            }
        })
        
        self.logger.info(f"Generation completed in {generation_time:.2f}s")
        return result
    
    def _generate_normal(self, prompt: str, config: GenerationConfig) -> GenerationResult:
        """Generate text in normal mode."""
        # Prepare input
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, 
                              max_length=config.max_length)
        input_ids = inputs.input_ids.to(self.device)
        attention_mask = inputs.attention_mask.to(self.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=config.max_new_tokens,
                temperature=config.temperature,
                top_p=config.top_p,
                top_k=config.top_k,
                repetition_penalty=config.repetition_penalty,
                do_sample=config.do_sample,
                num_return_sequences=config.num_return_sequences,
                early_stopping=config.early_stopping,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode output
        generated_text = self.tokenizer.decode(
            outputs[0][input_ids.shape[1]:], 
            skip_special_tokens=True
        )
        
        return GenerationResult(
            text=generated_text,
            generation_time=0.0,  # Will be set by caller
            tokens_generated=len(outputs[0]) - input_ids.shape[1],
            finish_reason="length" if len(outputs[0]) >= config.max_length else "eos_token"
        )
    
    def _generate_worldmodel(self, prompt: str, config: GenerationConfig) -> GenerationResult:
        """Generate text in WorldModel mode with structured reasoning."""
        # Check for uncertainty first
        uncertainty_result = None
        if config.enable_thinking and self.uncertainty_detector:
            uncertainty_result = self.uncertainty_detector.detect_uncertainty(prompt)
            if uncertainty_result.should_think:
                self.logger.info(f"Uncertainty detected: {uncertainty_result.metrics.trigger_reason}")
                prompt = f"{prompt}\n\n{uncertainty_result.suggested_prompt}"
        
        # RAG retrieval if enabled
        rag_context = []
        if config.enable_rag_retrieval and self.rag_system:
            try:
                # RAG temporarily disabled due to parameter mismatch
                rag_results = []
                rag_context = [doc.content for doc in rag_results]
                if rag_context:
                    context_text = "\n".join(rag_context)
                    prompt = f"Context:\n{context_text}\n\nQuery: {prompt}"
                    self.logger.info(f"Added {len(rag_context)} context documents")
            except Exception as e:
                self.logger.warning(f"RAG retrieval failed: {e}")
        
        # Generate with WorldModel stopping criteria
        stopping_criteria = WorldModelStoppingCriteria(self.tokenizer)
        
        # Initial generation
        result = self._generate_with_worldmodel_parsing(prompt, config, stopping_criteria)
        
        # Process any model tags for execution
        if result.parsed_tags and result.parsed_tags.has_model and config.enable_code_execution:
            result = self._execute_model_tags(result, config)
        
        # Add metadata
        result.uncertainty_analysis = uncertainty_result
        result.rag_context = rag_context
        
        return result
    
    def _generate_with_worldmodel_parsing(self, prompt: str, config: GenerationConfig,
                                        stopping_criteria) -> GenerationResult:
        """Generate text with incremental WorldModel tag parsing."""
        # Prepare input
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True,
                              max_length=config.max_length)
        input_ids = inputs.input_ids.to(self.device)
        attention_mask = inputs.attention_mask.to(self.device)
        
        # Setup stopping criteria
        stopping_criteria_list = StoppingCriteriaList([stopping_criteria])
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=config.max_new_tokens,
                temperature=config.temperature,
                top_p=config.top_p,
                top_k=config.top_k,
                repetition_penalty=config.repetition_penalty,
                do_sample=config.do_sample,
                early_stopping=False,  # Disable early stopping for WorldModel generation
                stopping_criteria=stopping_criteria_list,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode and parse
        generated_text = self.tokenizer.decode(
            outputs[0][input_ids.shape[1]:], 
            skip_special_tokens=True
        )
        
        # Parse WorldModel tags
        parsed_tags = self.tag_parser.parse(generated_text)
        
        return GenerationResult(
            text=generated_text,
            generation_time=0.0,
            tokens_generated=len(outputs[0]) - input_ids.shape[1],
            finish_reason="stopping_criteria",
            parsed_tags=parsed_tags
        )
    
    def _execute_model_tags(self, result: GenerationResult, config: GenerationConfig) -> GenerationResult:
        """Execute code from model tags and append results."""
        execution_results = []
        
        for model_tag in result.parsed_tags.model_tags:
            try:
                # Check for approval if required
                if config.require_approval and self.approval_system:
                    approval_result = self.approval_system.request_approval(
                        action_type="code_execution",
                        description=f"Execute {model_tag.language} code",
                        details={
                            'language': model_tag.language,
                            'code': model_tag.code[:200] + "..." if len(model_tag.code) > 200 else model_tag.code
                        }
                    )
                    
                    if not approval_result.approved:
                        self.logger.warning(f"Code execution denied: {approval_result.reason}")
                        execution_results.append({
                            'status': 'denied',
                            'reason': approval_result.reason,
                            'language': model_tag.language
                        })
                        continue
                
                # Execute code
                if self.vm_interface:
                    # Handle execution in async context
                    try:
                        # Try to get running loop
                        loop = asyncio.get_running_loop()
                        # We are in an async context, but this function is sync
                        # Use run_in_executor to run the async function
                        import concurrent.futures
                        executor = concurrent.futures.ThreadPoolExecutor()
                        
                        def run_async_in_thread():
                            return asyncio.run(self.vm_interface.execute_model_tag(model_tag))
                        
                        future = executor.submit(run_async_in_thread)
                        exec_result = future.result()
                        
                    except RuntimeError:
                        # No event loop running, can use asyncio.run directly
                        exec_result = asyncio.run(
                            self.vm_interface.execute_model_tag(model_tag)
                        )
                    
                    execution_results.append({
                        'status': exec_result.status.value,
                        'stdout': exec_result.stdout,
                        'stderr': exec_result.stderr,
                        'language': model_tag.language,
                        'execution_time': exec_result.execution_time,
                        'return_code': exec_result.return_code
                    })
                    
                    # Validate requirements if available
                    if result.parsed_tags.requires_tags and self.requirement_validator:
                        validation_result = self.requirement_validator.validate_execution(
                            exec_result,
                            model_tag,
                            result.parsed_tags.requires_tags
                        )
                        
                        execution_results[-1]['validation'] = {
                            'accuracy': validation_result.accuracy_score,
                            'violations': [v.__dict__ for v in validation_result.safety_violations]
                        }
                    
                    # Append execution output to generated text
                    if exec_result.success:
                        result.text += f"\n\nExecution Result:\n```\n{exec_result.stdout}\n```"
                    else:
                        result.text += f"\n\nExecution Error:\n```\n{exec_result.stderr}\n```"
                
            except Exception as e:
                self.logger.error(f"Code execution failed: {e}")
                execution_results.append({
                    'status': 'error',
                    'error': str(e),
                    'language': model_tag.language
                })
        
        result.execution_results = execution_results
        return result
    
    def _generate_interactive(self, prompt: str, config: GenerationConfig) -> GenerationResult:
        """Generate text in interactive mode with user feedback."""
        # Start with WorldModel generation
        result = self._generate_worldmodel(prompt, config)
        
        # In a real implementation, this would wait for user feedback
        # For now, we'll simulate a single iteration
        self.logger.info("Interactive mode: generated initial response")
        
        return result
    
    def stream_generate(self, prompt: str, mode: GenerationMode = GenerationMode.NORMAL,
                       config: Optional[GenerationConfig] = None) -> Generator[str, None, GenerationResult]:
        """
        Stream generation token by token.
        
        Args:
            prompt: Input prompt
            mode: Generation mode
            config: Generation configuration
            
        Yields:
            Generated tokens as they are produced
            
        Returns:
            Final GenerationResult
        """
        # For now, implement as a simple wrapper around generate
        # A full streaming implementation would use the model's streaming capabilities
        result = self.generate(prompt, mode, config)
        
        # Simulate streaming by yielding the text in chunks
        chunk_size = 10
        for i in range(0, len(result.text), chunk_size):
            chunk = result.text[i:i+chunk_size]
            yield chunk
            time.sleep(0.01)  # Simulate generation delay
        
        return result
    
    def get_conversation_history(self) -> List[Dict[str, Any]]:
        """Get the current conversation history."""
        return self.conversation_history.copy()
    
    def clear_conversation_history(self):
        """Clear the conversation history."""
        self.conversation_history = []
        self.logger.info("Conversation history cleared")
    
    def save_conversation(self, filepath: str):
        """Save conversation history to file."""
        with open(filepath, 'w') as f:
            json.dump(self.conversation_history, f, indent=2)
        self.logger.info(f"Conversation saved to {filepath}")
    
    def load_conversation(self, filepath: str):
        """Load conversation history from file."""
        with open(filepath, 'r') as f:
            self.conversation_history = json.load(f)
        self.logger.info(f"Conversation loaded from {filepath}")
    
    def update_rag_context(self, documents: List[str], metadata: Optional[List[Dict]] = None):
        """Add documents to the RAG system for future retrievals."""
        if self.rag_system:
            for i, doc in enumerate(documents):
                doc_metadata = metadata[i] if metadata and i < len(metadata) else {}
                self.rag_system.add_document(doc, doc_metadata)
            self.logger.info(f"Added {len(documents)} documents to RAG system")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        return {
            'model_path': self.model_path,
            'tokenizer_path': self.tokenizer_path,
            'device': str(self.device),
            'model_size': sum(p.numel() for p in self.model.parameters()) if self.model else 0,
            'vocab_size': self.tokenizer.vocab_size if self.tokenizer else 0,
            'conversation_length': len(self.conversation_history)
        }


# Convenience functions
def create_inference_engine(model_path: Optional[str] = None) -> InferenceEngine:
    """Create an inference engine with the specified model."""
    return InferenceEngine(model_path=model_path)

def generate_text(prompt: str, model_path: Optional[str] = None) -> str:
    """Quick text generation."""
    engine = create_inference_engine(model_path)
    result = engine.generate(prompt)
    return result.text

def generate_worldmodel(prompt: str, model_path: Optional[str] = None) -> GenerationResult:
    """Quick WorldModel generation with all features enabled."""
    engine = create_inference_engine(model_path)
    config = GenerationConfig(
        enable_thinking=True,
        enable_code_execution=True,
        enable_rag_retrieval=True,
        require_approval=False  # Disable for quick generation
    )
    return engine.generate(prompt, GenerationMode.WORLDMODEL, config)