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
        freeze_text_layers: bool = False,
        use_sandbox: bool = True,
        sandbox_config: Optional[Dict] = None
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
        self.cross_modal_indices = cross_modal_layers
        self.cross_modal_layers = nn.ModuleDict({
            str(layer_idx): CrossModalAttention(
                d_model=hidden_size,
                num_heads=self.config.num_attention_heads,
                dropout=0.1
            ) for layer_idx in cross_modal_layers
        })
        
        # WASM execution engine with sandbox config
        self.wasm_executor = WASMExecutor(
            timeout=5,
            use_sandbox=use_sandbox,
            sandbox_config=sandbox_config
        )
        
        # WASM tokenizer for token conversion (will be set during training)
        self.wasm_tokenizer = None
        
        # Output heads
        self.wasm_lm_head = nn.Linear(hidden_size, wasm_vocab_size, bias=False)
        
        print(f"✅ QwenWASMAdapter initialized")
        print(f"   Text layers: {len(self.text_model.encoder.layer) if hasattr(self.text_model, 'encoder') else 'N/A'}")
        print(f"   WASM layers: {len(self.wasm_layers)}")
        print(f"   Cross-modal fusion at layers: {self.cross_modal_indices}")
    
    def set_wasm_tokenizer(self, wasm_tokenizer):
        """Set the WASM tokenizer for token-to-WAT conversion."""
        self.wasm_tokenizer = wasm_tokenizer
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        wasm_ids: Optional[torch.Tensor] = None,
        wasm_attention_mask: Optional[torch.Tensor] = None,
        execute_wasm: bool = True,
        return_dict: bool = True,
        **kwargs  # Handle additional trainer parameters like 'labels'
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
        
        # Process WASM stream 
        wasm_hidden = None
        execution_results = []
        
        if wasm_ids is not None:
            # Use provided WASM tokens
            wasm_hidden = self.wasm_embeddings(wasm_ids)
        else:
            # Bootstrap WASM stream from text context for mathematical queries
            wasm_hidden = self._bootstrap_wasm_from_text(text_final_hidden)
            # Create attention mask for bootstrapped WASM sequence
            batch_size, wasm_seq_len, _ = wasm_hidden.shape
            wasm_attention_mask = torch.ones(batch_size, wasm_seq_len, dtype=torch.long, device=wasm_hidden.device)
            
            # Process through WASM layers with cross-modal attention
            wasm_layer_idx = 0
            
            for layer_idx in range(len(text_hidden_states)):
                # Cross-modal fusion at specified layers
                if layer_idx in self.cross_modal_indices:
                    text_layer_hidden = text_hidden_states[layer_idx]
                    
                    # Apply cross-modal attention
                    enhanced_text, enhanced_wasm = self.cross_modal_layers[str(layer_idx)](
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
        
        # Integrate execution results into text generation
        if execution_results and execute_wasm:
            # Inject computed tokens into the logits for next token prediction
            enhanced_logits = self._inject_computed_tokens(text_logits, execution_results)
        else:
            enhanced_logits = text_logits
        
        # Prepare outputs
        outputs = {
            "logits": enhanced_logits,
            "last_hidden_state": text_final_hidden,
            "hidden_states": text_hidden_states,
            "wasm_logits": wasm_logits,
            "wasm_hidden": wasm_hidden,
            "execution_results": execution_results
        }
        
        return outputs if return_dict else (enhanced_logits, wasm_logits)
    
    def _execute_wasm_at_layer(self, wasm_hidden: torch.Tensor, layer_idx: int) -> Dict[str, Any]:
        """Execute WASM code at a specific layer during forward pass."""
        try:
            # Convert WASM hidden states to tokens
            wasm_logits = self.wasm_lm_head(wasm_hidden)
            wasm_tokens = torch.argmax(wasm_logits, dim=-1)
            
            # Convert tokens to WAT code
            wat_code = self._tokens_to_wat(wasm_tokens)
            
            if wat_code and wat_code.strip():
                # Extract inputs from context (simple approach for arithmetic)
                inputs = self._extract_inputs_from_context(layer_idx)
                
                # Execute WASM using the WASM executor
                execution_result = self.wasm_executor.execute_wat(
                    wat_code=wat_code,
                    inputs=inputs,
                    api_calls=[]  # No external calls during forward pass
                )
                
                if execution_result["success"]:
                    result = execution_result.get("result", 0.0)
                    return {
                        "layer": layer_idx,
                        "success": True,
                        "result": result,
                        "wat_code": wat_code,
                        "inputs": inputs,
                        "computed_token": f"<computed>{result}</computed>"
                    }
                else:
                    return {
                        "layer": layer_idx,
                        "success": False,
                        "error": execution_result.get("error", "execution_failed"),
                        "wat_code": wat_code,
                        "inputs": inputs,
                        "computed_token": "<error>execution_failed</error>"
                    }
            else:
                # No valid WASM code generated yet
                return {
                    "layer": layer_idx,
                    "success": False,
                    "error": "no_valid_wat_generated",
                    "computed_token": ""
                }
            
        except Exception as e:
            return {
                "layer": layer_idx,
                "success": False,
                "error": str(e),
                "computed_token": "<error>execution_failed</error>"
            }
    
    def _tokens_to_wat(self, wasm_tokens: torch.Tensor) -> str:
        """Convert WASM tokens back to WAT code."""
        # This is a critical component - convert model output tokens to executable WAT
        try:
            # Get first sequence (batch_size=1 during generation)
            if len(wasm_tokens.shape) > 1:
                tokens = wasm_tokens[0]  # Take first in batch
            else:
                tokens = wasm_tokens
            
            # Convert to CPU and list if needed
            if hasattr(tokens, 'cpu'):
                tokens = tokens.cpu().tolist()
            elif hasattr(tokens, 'tolist'):
                tokens = tokens.tolist()
            else:
                tokens = list(tokens)
            
            # Use WASM tokenizer to decode if available
            if self.wasm_tokenizer is not None:
                try:
                    wat_code = self.wasm_tokenizer.decode_wat(tokens)
                    if wat_code and wat_code.strip():
                        return wat_code
                except Exception as e:
                    print(f"Warning: Tokenizer decode failed: {e}")
            
            # Fallback: generate simple arithmetic WAT based on token patterns
            return self._generate_arithmetic_wat(tokens)
            
        except Exception as e:
            print(f"Warning: Failed to convert tokens to WAT: {e}")
            return ""
    
    def _generate_arithmetic_wat(self, tokens) -> str:
        """Generate basic arithmetic WAT code from token patterns."""
        # This is a simplified approach for proof of concept
        # Real implementation would have full token->WAT conversion
        
        # Simulate different arithmetic operations based on token patterns
        import random
        operations = [
            # Simple addition
            """(module
  (func $add (param f64 f64) (result f64)
    local.get 0
    local.get 1
    f64.add))""",
            
            # Multiplication  
            """(module
  (func $mult (param f64 f64) (result f64)
    local.get 0
    local.get 1
    f64.mul))""",
            
            # Square function
            """(module
  (func $square (param f64) (result f64)
    local.get 0
    local.get 0
    f64.mul))"""
        ]
        
        # For now, randomly select operation (in real training, this would be deterministic)
        return random.choice(operations)
    
    def _extract_inputs_from_context(self, layer_idx: int) -> List[float]:
        """Extract numerical inputs from the current context for WASM execution."""
        # For proof of concept, return some example inputs
        # Real implementation would parse numbers from the input text
        
        # Common arithmetic examples
        examples = [
            [17.0, 23.0],  # 17 * 23
            [5.0, 3.0],    # 5 + 3
            [12.0],        # square of 12
            [144.0],       # sqrt of 144
            [8.0, 2.0],    # 8 / 2
        ]
        
        # Select based on layer (deterministic for same layer)
        idx = layer_idx % len(examples)
        return examples[idx]
    
    def _bootstrap_wasm_from_text(self, text_hidden: torch.Tensor) -> torch.Tensor:
        """Bootstrap WASM stream from text context for mathematical queries."""
        # Create initial WASM hidden state from text representation
        
        # Project text hidden state to WASM embedding space
        batch_size, seq_len, hidden_dim = text_hidden.shape
        
        # Create a simple projection (in real training, this would be learned)
        # For now, use last token's hidden state as WASM context
        last_token_hidden = text_hidden[:, -1:, :]  # Shape: [batch, 1, hidden]
        
        # Expand to create a small WASM sequence for processing
        wasm_seq_len = 8  # Small sequence for arithmetic operations
        wasm_hidden = last_token_hidden.expand(batch_size, wasm_seq_len, hidden_dim)
        
        # Add some variation to different positions
        position_offsets = torch.linspace(0.0, 1.0, wasm_seq_len).unsqueeze(0).unsqueeze(-1)
        if text_hidden.is_cuda:
            position_offsets = position_offsets.cuda()
        
        wasm_hidden = wasm_hidden + 0.1 * position_offsets * torch.randn_like(wasm_hidden)
        
        return wasm_hidden
    
    def _inject_computed_tokens(self, text_logits: torch.Tensor, execution_results: List[Dict]) -> torch.Tensor:
        """Inject computed results into token generation logits."""
        try:
            # Start with original logits
            enhanced_logits = text_logits.clone()
            
            # Get successful execution results
            successful_results = [r for r in execution_results if r.get("success", False)]
            
            if successful_results:
                # Get the computed token strings
                computed_tokens = []
                for result in successful_results:
                    computed_token = result.get("computed_token", "")
                    if computed_token and computed_token.strip():
                        computed_tokens.append(computed_token)
                
                if computed_tokens:
                    # Tokenize computed results
                    computed_text = " ".join(computed_tokens)
                    computed_token_ids = self.text_tokenizer.encode(computed_text, add_special_tokens=False)
                    
                    # Boost probabilities for computed tokens in next prediction
                    for token_id in computed_token_ids:
                        if token_id < enhanced_logits.size(-1):
                            # Increase logit for this token (making it more likely)
                            enhanced_logits[..., -1, token_id] += 2.0  # Boost by 2.0
            
            return enhanced_logits
            
        except Exception as e:
            print(f"Warning: Failed to inject computed tokens: {e}")
            return text_logits
    
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
    
    def save_pretrained(self, save_directory: str):
        """Save the WASM adapter model."""
        import os
        import json
        
        os.makedirs(save_directory, exist_ok=True)
        
        # Save the underlying text model
        self.text_model.save_pretrained(save_directory)
        
        # Save WASM-specific components
        wasm_state = {
            "wasm_embeddings": self.wasm_embeddings.state_dict(),
            "wasm_layers": [layer.state_dict() for layer in self.wasm_layers],
            "cross_modal_layers": {str(i): layer.state_dict() for i, layer in self.cross_modal_layers.items()},
            "wasm_lm_head": self.wasm_lm_head.state_dict(),
            "cross_modal_indices": self.cross_modal_indices,
            "wasm_vocab_size": self.wasm_vocab_size
        }
        
        torch.save(wasm_state, os.path.join(save_directory, "wasm_components.pt"))
        
        # Save adapter config
        adapter_config = {
            "model_type": "QwenWASMAdapter",
            "wasm_vocab_size": self.wasm_vocab_size,
            "cross_modal_layers": self.cross_modal_indices,
            "hidden_size": self.config.hidden_size,
            "num_wasm_layers": len(self.wasm_layers)
        }
        
        with open(os.path.join(save_directory, "wasm_adapter_config.json"), "w") as f:
            json.dump(adapter_config, f, indent=2)
    
    @classmethod
    def from_pretrained(cls, model_path: str, **kwargs):
        """Load a WASM adapter model."""
        import os
        import json
        
        # Load adapter config
        config_path = os.path.join(model_path, "wasm_adapter_config.json")
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                adapter_config = json.load(f)
        else:
            # Fallback to defaults for older models
            adapter_config = {
                "wasm_vocab_size": 8000,
                "cross_modal_layers": [3, 7, 11]
            }
        
        # Create adapter with loaded config
        adapter = cls(
            model_path=model_path,
            wasm_vocab_size=adapter_config.get("wasm_vocab_size", 8000),
            cross_modal_layers=adapter_config.get("cross_modal_layers", [3, 7, 11]),
            **kwargs
        )
        
        # Load WASM components if they exist
        wasm_components_path = os.path.join(model_path, "wasm_components.pt")
        if os.path.exists(wasm_components_path):
            wasm_state = torch.load(wasm_components_path, map_location="cpu")
            
            adapter.wasm_embeddings.load_state_dict(wasm_state["wasm_embeddings"])
            
            for i, layer_state in enumerate(wasm_state["wasm_layers"]):
                adapter.wasm_layers[i].load_state_dict(layer_state)
            
            for layer_idx, state_dict in wasm_state["cross_modal_layers"].items():
                adapter.cross_modal_layers[int(layer_idx)].load_state_dict(state_dict)
            
            adapter.wasm_lm_head.load_state_dict(wasm_state["wasm_lm_head"])
        
        return adapter


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