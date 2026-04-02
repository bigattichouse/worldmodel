"""
WebAssembly Text (WAT) Tokenizer
==============================

Tokenizes WAT format for transformer training.
Handles WASM opcodes, function definitions, and control structures.
"""

from typing import List, Dict, Set
import re
from transformers import PreTrainedTokenizer


class WATTokenizer:
    """Custom tokenizer for WebAssembly Text format."""
    
    def __init__(self, vocab_size: int = 8000, **kwargs):
        self.vocab_size = vocab_size
        self._build_wat_vocabulary()
        self._build_vocab_mappings()
    
    def _build_wat_vocabulary(self):
        """Build vocabulary specific to WAT syntax."""
        # Core WASM opcodes
        self.wasm_opcodes = {
            # Arithmetic
            "i32.add", "i32.sub", "i32.mul", "i32.div_s", "i32.div_u",
            "i64.add", "i64.sub", "i64.mul", "i64.div_s", "i64.div_u", 
            "f32.add", "f32.sub", "f32.mul", "f32.div",
            "f64.add", "f64.sub", "f64.mul", "f64.div",
            
            # Control flow
            "block", "loop", "if", "else", "end", "br", "br_if", "return",
            
            # Memory
            "local.get", "local.set", "local.tee",
            "global.get", "global.set",
            
            # Function calls
            "call", "call_indirect",
            
            # Comparison
            "i32.eq", "i32.ne", "i32.lt_s", "i32.gt_s", "i32.le_s", "i32.ge_s",
            "f32.eq", "f32.ne", "f32.lt", "f32.gt", "f32.le", "f32.ge",
        }
        
        # Structure tokens
        self.structure_tokens = {
            "module", "func", "param", "result", "local", "export", "import"
        }
        
        # Data types
        self.data_types = {"i32", "i64", "f32", "f64"}
        
        # Special tokens
        self.special_tokens = {
            "[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]",
            "<wat_start>", "<wat_end>", "<computed>", "</computed>"
        }
    
    def _build_vocab_mappings(self):
        """Build token to ID mappings."""
        # Combine all vocabulary items
        all_tokens = (
            list(self.special_tokens) + 
            list(self.wasm_opcodes) + 
            list(self.structure_tokens) + 
            list(self.data_types) + 
            ["(", ")", "$", "local", "global", "memory", "table"] +
            [f"tok_{i}" for i in range(100)]  # Generic tokens for numbers/identifiers
        )
        
        # Create bidirectional mappings
        self.token_to_id = {token: i for i, token in enumerate(all_tokens[:self.vocab_size])}
        self.id_to_token = {i: token for token, i in self.token_to_id.items()}
        
        # Special token IDs
        self.pad_token = "[PAD]"
        self.unk_token = "[UNK]"
        self.pad_token_id = self.token_to_id.get("[PAD]", 0)
        self.unk_token_id = self.token_to_id.get("[UNK]", 1)
    
    def get_vocab(self):
        """Return vocabulary dictionary."""
        return self.token_to_id.copy()
    
    def tokenize_wat(self, wat_code: str) -> List[str]:
        """Tokenize WAT code into meaningful tokens."""
        # Remove comments and normalize whitespace
        wat_code = re.sub(r';;.*$', '', wat_code, flags=re.MULTILINE)
        wat_code = re.sub(r'\s+', ' ', wat_code).strip()
        
        tokens = []
        i = 0
        
        while i < len(wat_code):
            if wat_code[i] == '(':
                tokens.append('(')
                i += 1
            elif wat_code[i] == ')':
                tokens.append(')')
                i += 1
            elif wat_code[i] == ' ':
                i += 1
            else:
                # Extract word/number
                start = i
                while i < len(wat_code) and wat_code[i] not in '() ':
                    i += 1
                word = wat_code[start:i]
                
                if word:
                    tokens.append(word)
        
        return tokens
    
    def encode_wat(self, wat_code: str) -> List[int]:
        """Convert WAT code to token IDs."""
        tokens = self.tokenize_wat(wat_code)
        token_ids = []
        for token in tokens:
            token_id = self.token_to_id.get(token, self.unk_token_id)
            token_ids.append(token_id)
        return token_ids
    
    def decode_wat(self, token_ids: List[int]) -> str:
        """Convert token IDs back to WAT code."""
        tokens = []
        for token_id in token_ids:
            token = self.id_to_token.get(token_id, self.unk_token)
            tokens.append(token)
        return " ".join(tokens)