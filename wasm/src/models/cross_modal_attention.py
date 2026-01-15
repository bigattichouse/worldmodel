"""
Cross-Modal Attention Mechanism
==============================

Implements Flamingo-style cross-modal attention between text and WASM streams.
Based on SOTA multimodal architectures from DeepMind and similar research.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class CrossModalAttention(nn.Module):
    """Cross-attention between text and WASM modalities."""
    
    def __init__(
        self, 
        d_model: int = 768,
        num_heads: int = 12,
        dropout: float = 0.1
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        # Text-to-WASM attention
        self.text_to_wasm = MultiHeadCrossAttention(d_model, num_heads, dropout)
        
        # WASM-to-text attention  
        self.wasm_to_text = MultiHeadCrossAttention(d_model, num_heads, dropout)
        
        # Output projections
        self.text_output_proj = nn.Linear(d_model, d_model)
        self.wasm_output_proj = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self, 
        text_hidden: torch.Tensor,
        wasm_hidden: torch.Tensor,
        text_mask: Optional[torch.Tensor] = None,
        wasm_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            text_hidden: [batch_size, text_seq_len, d_model]
            wasm_hidden: [batch_size, wasm_seq_len, d_model] 
            text_mask: [batch_size, text_seq_len]
            wasm_mask: [batch_size, wasm_seq_len]
        
        Returns:
            enhanced_text: [batch_size, text_seq_len, d_model]
            enhanced_wasm: [batch_size, wasm_seq_len, d_model]
        """
        # Text queries attend to WASM keys/values
        text_enhanced = self.text_to_wasm(
            query=text_hidden,
            key=wasm_hidden, 
            value=wasm_hidden,
            key_mask=wasm_mask
        )
        
        # WASM queries attend to text keys/values
        wasm_enhanced = self.wasm_to_text(
            query=wasm_hidden,
            key=text_hidden,
            value=text_hidden, 
            key_mask=text_mask
        )
        
        # Residual connections and projections
        text_output = self.text_output_proj(text_enhanced) + text_hidden
        wasm_output = self.wasm_output_proj(wasm_enhanced) + wasm_hidden
        
        return self.dropout(text_output), self.dropout(wasm_output)


class MultiHeadCrossAttention(nn.Module):
    """Standard multi-head cross attention implementation."""
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self, 
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        batch_size, seq_len = query.shape[:2]
        
        # Project to Q, K, V
        Q = self.q_proj(query)  # [batch, seq_len, d_model]
        K = self.k_proj(key)    # [batch, key_len, d_model]  
        V = self.v_proj(value)  # [batch, key_len, d_model]
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention (convert mask to correct type if needed)
        if key_mask is not None and key_mask.dtype not in [torch.bool, torch.float16, torch.float32]:
            key_mask = key_mask.to(torch.bool)
        
        attention_output = F.scaled_dot_product_attention(
            Q, K, V,
            attn_mask=key_mask,
            dropout_p=self.dropout.p if self.training else 0.0
        )
        
        # Reshape and project
        attention_output = attention_output.transpose(1, 2).contiguous()
        attention_output = attention_output.view(batch_size, seq_len, self.d_model)
        
        return self.out_proj(attention_output)