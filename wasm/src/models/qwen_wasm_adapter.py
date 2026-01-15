"""
Qwen WASM Adapter
================

Adapts pre-trained Qwen model to support WASM stream processing.
Adds WASM capabilities while preserving existing language understanding.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, List, Any
from transformers import AutoModel, AutoTokenizer, AutoConfig

from .cross_modal_attention import CrossModalAttention
from ..execution.wasm_executor import WASMExecutor
from ..tokenization.wat_tokenizer import WATTokenizer


class QwenWASMAdapter(nn.Module):
    """
    Adapter that adds WASM processing capabilities to pre-trained Qwen model.
    
    Architecture:
    - Loads frozen/unfrozen Qwen for text processing
    - Adds parallel WASM stream with cross-modal attention
    - Integrates WASM execution during forward pass
    """
    
    def __init__(
        self,
        model_path: str,
        wasm_vocab_size: int = 8000,
        cross_modal_layers: List[int] = [3, 7, 11],
        freeze_text_layers: bool = False
    ):
        super().__init__()
        
        # Load pre-trained Qwen
        print(f"Loading Qwen model from {model_path}")
        from transformers import AutoModelForCausalLM
        self.text_model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
        self.text_tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.config = self.text_model.config
        
        # Optionally freeze text model weights
        if freeze_text_layers:
            for param in self.text_model.parameters():
                param.requires_grad = False
            print("✅ Text model layers frozen")
        
        # WASM stream components
        self.wasm_vocab_size = wasm_vocab_size
        self.wasm_tokenizer = WATTokenizer(vocab_size=wasm_vocab_size)
        
        # WASM embeddings and processing
        hidden_size = self.config.hidden_size
        self.wasm_embeddings = nn.Embedding(wasm_vocab_size, hidden_size)
        
        # WASM transformer layers (simpler than text layers)
        self.wasm_layers = nn.ModuleList([
            WASMProcessingLayer(hidden_size, self.config.num_attention_heads)
            for _ in range(len(cross_modal_layers))  # Fewer WASM layers
        ])
        
        # Cross-modal attention at specified layers
        self.cross_modal_layers = cross_modal_layers
        self.cross_modal_attention = nn.ModuleDict({
            str(layer_idx): CrossModalAttention(
                d_model=hidden_size,
                num_heads=self.config.num_attention_heads,
                dropout=0.1
            ) for layer_idx in cross_modal_layers
        })
        
        # WASM execution engine
        self.wasm_executor = WASMExecutor()
        
        # Output heads
        self.wasm_lm_head = nn.Linear(hidden_size, wasm_vocab_size, bias=False)
        
        print(f"✅ QwenWASMAdapter initialized")
        print(f"   Text layers: {len(self.text_model.encoder.layer) if hasattr(self.text_model, 'encoder') else 'N/A'}")
        print(f"   WASM layers: {len(self.wasm_layers)}")
        print(f"   Cross-modal fusion at layers: {cross_modal_layers}")
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        wasm_ids: Optional[torch.Tensor] = None,
        wasm_attention_mask: Optional[torch.Tensor] = None,
        execute_wasm: bool = True,
        return_dict: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with text + WASM processing.
        
        Args:
            input_ids: Text token IDs [batch_size, seq_len]
            attention_mask: Text attention mask
            wasm_ids: WASM token IDs [batch_size, wasm_seq_len]
            wasm_attention_mask: WASM attention mask  
            execute_wasm: Whether to execute WASM code
            return_dict: Whether to return dictionary
        """
        batch_size, seq_len = input_ids.shape
        
        # Process text through Qwen
        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True
        )
        
        text_hidden_states = text_outputs.hidden_states  # All layer outputs
        text_final_hidden = text_outputs.hidden_states[-1]  # Last layer hidden state
        text_logits = text_outputs.logits
        
        # Process WASM stream (if provided)
        wasm_hidden = None
        execution_results = []
        
        if wasm_ids is not None:
            wasm_hidden = self.wasm_embeddings(wasm_ids)
            
            # Process through WASM layers with cross-modal attention
            wasm_layer_idx = 0
            
            for layer_idx in range(len(text_hidden_states)):
                # Cross-modal fusion at specified layers
                if layer_idx in self.cross_modal_layers:
                    text_layer_hidden = text_hidden_states[layer_idx]
                    
                    # Apply cross-modal attention
                    enhanced_text, enhanced_wasm = self.cross_modal_attention[str(layer_idx)](
                        text_layer_hidden, wasm_hidden, attention_mask, wasm_attention_mask
                    )
                    
                    # Process WASM through its own layer
                    if wasm_layer_idx < len(self.wasm_layers):
                        wasm_hidden = self.wasm_layers[wasm_layer_idx](enhanced_wasm)
                        wasm_layer_idx += 1
                    
                    # Execute WASM at this layer
                    if execute_wasm:
                        execution_result = self._execute_wasm_at_layer(wasm_hidden, layer_idx)
                        execution_results.append(execution_result)
        
        # Generate WASM output logits
        wasm_logits = None
        if wasm_hidden is not None:
            wasm_logits = self.wasm_lm_head(wasm_hidden)
        
        # Prepare outputs
        outputs = {
            "logits": text_logits,
            "last_hidden_state": text_final_hidden,
            "hidden_states": text_hidden_states,
            "wasm_logits": wasm_logits,
            "wasm_hidden": wasm_hidden,
            "execution_results": execution_results
        }
        
        return outputs if return_dict else (text_logits, wasm_logits)
    
    def _execute_wasm_at_layer(self, wasm_hidden: torch.Tensor, layer_idx: int) -> Dict[str, Any]:
        """Execute WASM code at a specific layer."""
        try:
            # Convert WASM hidden states to tokens
            wasm_logits = self.wasm_lm_head(wasm_hidden)
            wasm_tokens = torch.argmax(wasm_logits, dim=-1)
            
            # For now, simulate execution with simple calculator
            # In full implementation, would convert tokens to WAT and execute
            result = self._simulate_calculator_execution(wasm_tokens)
            
            return {
                "layer": layer_idx,
                "success": True,
                "result": result,
                "computed_token": f"<computed>{result}</computed>"
            }
            
        except Exception as e:
            return {
                "layer": layer_idx,
                "success": False,
                "error": str(e),
                "computed_token": "<error>execution_failed</error>"
            }
    
    def _simulate_calculator_execution(self, wasm_tokens: torch.Tensor) -> float:
        """Simulate basic calculator execution for testing."""
        # Placeholder: return a simple calculation result
        # In real implementation, would parse WASM tokens and execute
        return 42.0
    
    def generate_with_wasm(
        self,
        input_text: str,
        max_length: int = 100,
        temperature: float = 0.7,
        execute_wasm: bool = True
    ) -> Dict[str, Any]:
        """Generate text with WASM computation support."""
        
        # Tokenize input
        inputs = self.text_tokenizer(input_text, return_tensors="pt")
        
        # Forward pass
        with torch.no_grad():
            outputs = self.forward(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                execute_wasm=execute_wasm
            )
        
        # Generate response (simplified)
        generated_ids = inputs.input_ids  # Placeholder
        generated_text = self.text_tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        
        return {
            "input": input_text,
            "output": generated_text,
            "execution_results": outputs["execution_results"],
            "wasm_executed": len(outputs["execution_results"]) > 0
        }


class WASMProcessingLayer(nn.Module):
    """Simplified transformer layer for WASM code processing."""
    
    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()
        self.self_attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True
        )
        self.norm1 = nn.LayerNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Linear(hidden_size * 4, hidden_size),
            nn.Dropout(0.1)
        )
        self.norm2 = nn.LayerNorm(hidden_size)
    
    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        # Self-attention
        residual = hidden_states
        hidden_states = self.norm1(hidden_states)
        
        attn_output, _ = self.self_attention(
            hidden_states, hidden_states, hidden_states,
            key_padding_mask=attention_mask if attention_mask is not None else None
        )
        hidden_states = residual + attn_output
        
        # Feed forward
        residual = hidden_states
        hidden_states = self.norm2(hidden_states)
        hidden_states = residual + self.ffn(hidden_states)
        
        return hidden_states