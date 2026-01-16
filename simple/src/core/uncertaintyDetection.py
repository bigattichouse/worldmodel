"""
Uncertainty detection for WorldModel LLM experiment.

Implements perplexity-based detection to determine when the model should
enter <think>/<model> mode due to knowledge gaps or uncertainty.
"""

import torch
import torch.nn.functional as F
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

from ..utils.config import get_config
from ..utils.logging import get_logger

logger = get_logger('uncertaintyDetection')


@dataclass
class UncertaintyMetrics:
    """Container for uncertainty detection metrics."""
    perplexity: float
    confidence: float
    entropy: float
    max_prob: float
    min_prob: float
    num_tokens: int
    triggered: bool
    trigger_reason: str = ""
    token_perplexities: List[float] = None
    
    def __post_init__(self):
        if self.token_perplexities is None:
            self.token_perplexities = []


@dataclass
class DetectionResult:
    """Result of uncertainty detection analysis."""
    should_think: bool
    metrics: UncertaintyMetrics
    suggested_prompt: Optional[str] = None
    confidence_score: float = 0.0


class UncertaintyDetector:
    """Detects uncertainty using perplexity-based analysis."""
    
    def __init__(self, model_path: Optional[str] = None, tokenizer_path: Optional[str] = None):
        self.config = get_config()
        self.logger = get_logger('uncertaintyDetection')
        
        # Model and tokenizer paths
        self.model_path = model_path or self.config.model.model_path
        self.tokenizer_path = tokenizer_path or self.config.model.model_path
        
        # Detection thresholds
        self.perplexity_threshold = self.config.uncertainty.perplexity_threshold
        self.confidence_threshold = self.config.uncertainty.confidence_threshold
        self.min_tokens = self.config.uncertainty.min_tokens_for_detection
        
        # Model components
        self.model = None
        self.tokenizer = None
        self.device = torch.device(self.config.model.device if torch.cuda.is_available() else 'cpu')
        
        # Initialize if paths exist
        self._lazy_init()
    
    def _lazy_init(self):
        """Initialize model and tokenizer lazily."""
        try:
            if self.tokenizer is None:
                self.logger.info(f"Loading tokenizer from {self.tokenizer_path}")
                self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)
                
                # Ensure pad token exists
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
            
            if self.model is None:
                self.logger.info(f"Loading model from {self.model_path}")
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    torch_dtype=getattr(torch, self.config.model.torch_dtype),
                    device_map='auto' if torch.cuda.is_available() else None
                )
                self.model.eval()
                
                # Move to device if not using device_map
                if not torch.cuda.is_available():
                    self.model = self.model.to(self.device)
                    
        except Exception as e:
            self.logger.error(f"Failed to initialize model/tokenizer: {e}")
            raise
    
    def detect_uncertainty(self, text: str, context: Optional[str] = None) -> DetectionResult:
        """
        Detect uncertainty in the given text.
        
        Args:
            text: Text to analyze for uncertainty
            context: Optional context/conversation history
            
        Returns:
            DetectionResult with uncertainty analysis
        """
        self.logger.debug(f"Analyzing uncertainty for text of length {len(text)}")
        
        # Ensure model is loaded
        self._lazy_init()
        
        # Combine context and text if provided
        full_text = f"{context}\n{text}" if context else text
        
        # Calculate perplexity and related metrics
        metrics = self._calculate_uncertainty_metrics(full_text)
        
        # Determine if thinking mode should be triggered
        should_think, trigger_reason = self._should_trigger_thinking(metrics)
        
        # Generate suggested prompt if thinking should be triggered
        suggested_prompt = None
        confidence_score = 1.0 - metrics.confidence  # Higher uncertainty = lower confidence
        
        if should_think:
            suggested_prompt = self._generate_thinking_prompt(text, metrics)
        
        # Update metrics with trigger information
        metrics.triggered = should_think
        metrics.trigger_reason = trigger_reason
        
        result = DetectionResult(
            should_think=should_think,
            metrics=metrics,
            suggested_prompt=suggested_prompt,
            confidence_score=confidence_score
        )
        
        self.logger.info(f"Uncertainty detection: triggered={should_think}, "
                        f"perplexity={metrics.perplexity:.2f}, "
                        f"confidence={metrics.confidence:.3f}")
        
        return result
    
    def _calculate_uncertainty_metrics(self, text: str) -> UncertaintyMetrics:
        """Calculate comprehensive uncertainty metrics for the text."""
        # Tokenize the input
        inputs = self.tokenizer(
            text, 
            return_tensors="pt", 
            truncation=True, 
            max_length=self.config.model.max_length,
            padding=False
        )
        input_ids = inputs.input_ids.to(self.device)
        
        if input_ids.size(1) < self.min_tokens:
            self.logger.warning(f"Text too short ({input_ids.size(1)} tokens) for reliable detection")
        
        # Get model predictions
        with torch.no_grad():
            outputs = self.model(input_ids)
            logits = outputs.logits
        
        # Calculate perplexity
        perplexity, token_perplexities = self._calculate_perplexity(input_ids, logits)
        
        # Calculate confidence metrics
        probs = F.softmax(logits, dim=-1)
        max_probs = torch.max(probs, dim=-1)[0]
        min_probs = torch.min(probs, dim=-1)[0]
        
        # Calculate entropy (uncertainty measure)
        entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)
        
        # Aggregate metrics
        avg_max_prob = torch.mean(max_probs).item()
        avg_min_prob = torch.mean(min_probs).item()
        avg_entropy = torch.mean(entropy).item()
        
        # Confidence is inverse of normalized entropy
        # Normalize entropy by log(vocab_size) to get value between 0-1
        vocab_size = self.tokenizer.vocab_size
        normalized_entropy = avg_entropy / np.log(vocab_size)
        confidence = 1.0 - normalized_entropy
        
        return UncertaintyMetrics(
            perplexity=perplexity,
            confidence=confidence,
            entropy=avg_entropy,
            max_prob=avg_max_prob,
            min_prob=avg_min_prob,
            num_tokens=input_ids.size(1),
            triggered=False,  # Will be set by caller
            token_perplexities=token_perplexities
        )
    
    def _calculate_perplexity(self, input_ids: torch.Tensor, logits: torch.Tensor) -> Tuple[float, List[float]]:
        """Calculate perplexity for the input sequence."""
        # Shift input_ids and logits for next token prediction
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()
        
        # Calculate cross entropy loss for each position
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        shift_logits_flat = shift_logits.view(-1, shift_logits.size(-1))
        shift_labels_flat = shift_labels.view(-1)
        
        losses = loss_fct(shift_logits_flat, shift_labels_flat)
        losses = losses.view(shift_labels.shape)
        
        # Calculate perplexity
        avg_loss = torch.mean(losses).item()
        perplexity = np.exp(avg_loss)
        
        # Get per-token perplexities
        token_perplexities = [np.exp(loss.item()) for loss in losses.squeeze()]
        
        return perplexity, token_perplexities
    
    def _should_trigger_thinking(self, metrics: UncertaintyMetrics) -> Tuple[bool, str]:
        """Determine if thinking mode should be triggered based on metrics."""
        triggers = []
        
        # Check perplexity threshold
        if metrics.perplexity > self.perplexity_threshold:
            triggers.append(f"high_perplexity({metrics.perplexity:.1f})")
        
        # Check confidence threshold
        if metrics.confidence < self.confidence_threshold:
            triggers.append(f"low_confidence({metrics.confidence:.3f})")
        
        # Check for high entropy (uncertainty)
        vocab_size = self.tokenizer.vocab_size
        max_entropy = np.log(vocab_size)
        if metrics.entropy > 0.8 * max_entropy:
            triggers.append(f"high_entropy({metrics.entropy:.2f})")
        
        # Check for highly variable token perplexities (indicates uncertainty)
        if len(metrics.token_perplexities) > 1:
            perp_std = np.std(metrics.token_perplexities)
            perp_mean = np.mean(metrics.token_perplexities)
            if perp_std > 0.5 * perp_mean:  # High relative variance
                triggers.append(f"variable_perplexity(std={perp_std:.1f})")
        
        # Check if text is too short for reliable detection
        if metrics.num_tokens < self.min_tokens:
            return False, "insufficient_tokens"
        
        should_trigger = len(triggers) > 0
        trigger_reason = ",".join(triggers) if triggers else "none"
        
        return should_trigger, trigger_reason
    
    def _generate_thinking_prompt(self, text: str, metrics: UncertaintyMetrics) -> str:
        """Generate a prompt to encourage thinking mode."""
        base_prompt = "The previous response shows uncertainty. Consider using structured reasoning:"
        
        suggestions = []
        
        if metrics.perplexity > self.perplexity_threshold:
            suggestions.append("Break down the problem into steps using <think> tags")
        
        if metrics.confidence < self.confidence_threshold:
            suggestions.append("Look up relevant information or create models to verify facts")
        
        if metrics.entropy > 0.8 * np.log(self.tokenizer.vocab_size):
            suggestions.append("Consider multiple approaches and evaluate which is most appropriate")
        
        if not suggestions:
            suggestions.append("Use <think> to reason through the problem systematically")
        
        return f"{base_prompt}\n\n" + "\n".join(f"- {s}" for s in suggestions)
    
    def analyze_conversation(self, conversation_history: List[Dict[str, str]]) -> List[DetectionResult]:
        """
        Analyze uncertainty across an entire conversation.
        
        Args:
            conversation_history: List of message dicts with 'role' and 'content'
            
        Returns:
            List of DetectionResult for each message
        """
        results = []
        context = ""
        
        for message in conversation_history:
            content = message.get('content', '')
            role = message.get('role', 'unknown')
            
            if role == 'assistant':
                # Analyze assistant responses for uncertainty
                result = self.detect_uncertainty(content, context)
                results.append(result)
            
            # Add to context for next analysis
            context += f"{role}: {content}\n"
        
        return results
    
    def get_confidence_score(self, text: str) -> float:
        """Get a simple confidence score for the text (0-1, higher is more confident)."""
        metrics = self._calculate_uncertainty_metrics(text)
        return metrics.confidence
    
    def batch_detect(self, texts: List[str], contexts: Optional[List[str]] = None) -> List[DetectionResult]:
        """
        Detect uncertainty for multiple texts in batch.
        
        Args:
            texts: List of texts to analyze
            contexts: Optional list of contexts (same length as texts)
            
        Returns:
            List of DetectionResult for each text
        """
        if contexts and len(contexts) != len(texts):
            raise ValueError("contexts must be same length as texts")
        
        results = []
        for i, text in enumerate(texts):
            context = contexts[i] if contexts else None
            result = self.detect_uncertainty(text, context)
            results.append(result)
        
        return results
    
    def update_thresholds(self, perplexity_threshold: Optional[float] = None,
                         confidence_threshold: Optional[float] = None):
        """Update detection thresholds dynamically."""
        if perplexity_threshold is not None:
            self.perplexity_threshold = perplexity_threshold
            self.logger.info(f"Updated perplexity threshold to {perplexity_threshold}")
        
        if confidence_threshold is not None:
            self.confidence_threshold = confidence_threshold
            self.logger.info(f"Updated confidence threshold to {confidence_threshold}")


# Convenience functions
def detect_uncertainty(text: str, context: Optional[str] = None, 
                      model_path: Optional[str] = None) -> DetectionResult:
    """Convenience function for uncertainty detection."""
    detector = UncertaintyDetector(model_path=model_path)
    return detector.detect_uncertainty(text, context)

def get_confidence_score(text: str, model_path: Optional[str] = None) -> float:
    """Get confidence score for text."""
    detector = UncertaintyDetector(model_path=model_path)
    return detector.get_confidence_score(text)

def should_think(text: str, context: Optional[str] = None, 
                model_path: Optional[str] = None) -> bool:
    """Simple boolean check if thinking mode should be triggered."""
    result = detect_uncertainty(text, context, model_path)
    return result.should_think