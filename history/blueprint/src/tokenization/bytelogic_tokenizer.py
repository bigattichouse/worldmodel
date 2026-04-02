"""
ByteLogic Tokenizer
==================

Tokenizes ByteLogic syntax for transformer training and execution.
Handles datalog-style logic programming constructs, mathematical calculations,
and structured reasoning patterns.
"""

from typing import List, Dict, Set, Tuple, Optional
import re
from transformers import PreTrainedTokenizer


class ByteLogicTokenizer:
    """Custom tokenizer for ByteLogic language syntax."""
    
    def __init__(self, vocab_size: int = 12000, **kwargs):
        self.vocab_size = vocab_size
        self._build_bytelogic_vocabulary()
        self._build_vocab_mappings()
        self._compile_patterns()
    
    def _build_bytelogic_vocabulary(self):
        """Build vocabulary specific to ByteLogic syntax."""
        
        # Core ByteLogic keywords
        self.bytelogic_keywords = {
            # Logic programming
            "REL", "FACT", "RULE", "SCAN", "JOIN", "EMIT", "SOLVE", "QUERY", "MATCH",
            
            # Calculation constructs  
            "CALC", "INPUT", "LET", "RESULT", "IF", "THEN", "ELSE", "END", "WHERE",
            
            # Loop constructs
            "FOR", "WHILE", "IN", "RANGE", "BREAK", "CONTINUE",
            
            # Mathematical functions
            "POW", "ABS", "MIN", "MAX", "SQRT", "SIN", "COS", "TAN", 
            "ASIN", "ACOS", "ATAN", "LOG", "LOG10", "EXP", "CEIL", "FLOOR",
            
            # String functions
            "LENGTH", "CHAR_AT",
            
            # Execution
            "EXEC"
        }
        
        # Operators and symbols
        self.operators = {
            "+", "-", "*", "/", "%", "=", "==", "!=", "<>", "<", ">", "<=", ">=",
            "(", ")", ",", ":", "?", "$"
        }
        
        # Data types
        self.data_types = {"i32", "f32", "f64", "string"}
        
        # Special computation tokens
        self.special_tokens = {
            "[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]",
            "<computation>", "</computation>", 
            "<result>", "</result>",
            "<logic>", "</logic>",
            "<error>", "</error>"
        }
        
        # Common atom names for logic programming
        self.common_atoms = {
            "alice", "bob", "charlie", "david", "eve", "frank",
            "parent", "child", "sibling", "grandparent", "ancestor",
            "friend", "knows", "likes", "manages", "reports_to",
            "pizza", "pasta", "salad", "ice_cream",
            "engineering", "marketing", "management",
            "true", "false"
        }
    
    def _build_vocab_mappings(self):
        """Build token to ID mappings."""
        # Combine all vocabulary items with priority order
        all_tokens = (
            list(self.special_tokens) +           # Special tokens first
            list(self.bytelogic_keywords) +       # Keywords
            list(self.operators) +                # Operators  
            list(self.data_types) +               # Data types
            list(self.common_atoms) +             # Common atoms
            [f"var_{i}" for i in range(100)] +    # Variable placeholders $0, $1, etc.
            [f"num_{i}" for i in range(1000)] +   # Number placeholders
            [f"atom_{i}" for i in range(500)] +   # Generic atom placeholders
            [f"tok_{i}" for i in range(1000)]     # Generic tokens
        )
        
        # Create bidirectional mappings
        self.token_to_id = {token: i for i, token in enumerate(all_tokens[:self.vocab_size])}
        self.id_to_token = {i: token for token, i in self.token_to_id.items()}
        
        # Special token IDs
        self.pad_token = "[PAD]"
        self.unk_token = "[UNK]"
        self.pad_token_id = self.token_to_id.get("[PAD]", 0)
        self.unk_token_id = self.token_to_id.get("[UNK]", 1)
        
        # Computation token IDs
        self.computation_start_id = self.token_to_id.get("<computation>", 2)
        self.computation_end_id = self.token_to_id.get("</computation>", 3)
    
    def _compile_patterns(self):
        """Compile regex patterns for tokenization."""
        # Pattern for computation tokens
        self.computation_pattern = re.compile(
            r'<computation>\s*(.*?)\s*</computation>',
            re.DOTALL | re.IGNORECASE
        )
        
        # Pattern for variables ($0, $1, $2, etc.)
        self.variable_pattern = re.compile(r'\$(\d+)')
        
        # Pattern for numbers (integers and floats)
        self.number_pattern = re.compile(r'-?\d+\.?\d*')
        
        # Pattern for quoted strings
        self.string_pattern = re.compile(r'"([^"]*)"')
        
        # Pattern for identifiers/atoms
        self.identifier_pattern = re.compile(r'[a-zA-Z_][a-zA-Z0-9_]*')
    
    def get_vocab(self):
        """Return vocabulary dictionary."""
        return self.token_to_id.copy()
    
    def extract_computation_tokens(self, text: str) -> List[Tuple[str, str]]:
        """
        Extract computation tokens from text.
        
        Returns:
            List of (full_match, bytelog_code) tuples
        """
        matches = []
        for match in self.computation_pattern.finditer(text):
            full_match = match.group(0)
            bytelog_code = match.group(1).strip()
            matches.append((full_match, bytelog_code))
        return matches
    
    def tokenize_bytelogic(self, bytelog_code: str) -> List[str]:
        """Tokenize ByteLogic code into meaningful tokens."""
        # Remove comments and normalize whitespace
        bytelog_code = re.sub(r';.*$', '', bytelog_code, flags=re.MULTILINE)
        bytelog_code = re.sub(r'//.*$', '', bytelog_code, flags=re.MULTILINE)
        bytelog_code = re.sub(r'\s+', ' ', bytelog_code).strip()
        
        tokens = []
        i = 0
        
        while i < len(bytelog_code):
            # Skip whitespace
            if bytelog_code[i].isspace():
                i += 1
                continue
            
            # Handle operators and symbols (multi-character first)
            if i < len(bytelog_code) - 1:
                two_char = bytelog_code[i:i+2]
                if two_char in {'==', '!=', '<>', '<=', '>='}:
                    tokens.append(two_char)
                    i += 2
                    continue
            
            if bytelog_code[i] in self.operators:
                tokens.append(bytelog_code[i])
                i += 1
                continue
            
            # Handle quoted strings
            if bytelog_code[i] == '"':
                end_quote = bytelog_code.find('"', i + 1)
                if end_quote != -1:
                    string_token = bytelog_code[i:end_quote + 1]
                    tokens.append(string_token)
                    i = end_quote + 1
                    continue
                else:
                    # Unclosed quote - treat as single character
                    tokens.append(bytelog_code[i])
                    i += 1
                    continue
            
            # Handle variables ($0, $1, etc.)
            if bytelog_code[i] == '$':
                match = self.variable_pattern.match(bytelog_code, i)
                if match:
                    tokens.append(match.group(0))
                    i = match.end()
                    continue
                else:
                    tokens.append('$')
                    i += 1
                    continue
            
            # Handle numbers
            if bytelog_code[i].isdigit() or (bytelog_code[i] == '-' and i + 1 < len(bytelog_code) and bytelog_code[i + 1].isdigit()):
                match = self.number_pattern.match(bytelog_code, i)
                if match:
                    tokens.append(match.group(0))
                    i = match.end()
                    continue
            
            # Handle identifiers/keywords/atoms
            if bytelog_code[i].isalpha() or bytelog_code[i] == '_':
                match = self.identifier_pattern.match(bytelog_code, i)
                if match:
                    word = match.group(0)
                    tokens.append(word)
                    i = match.end()
                    continue
            
            # Handle any other character
            tokens.append(bytelog_code[i])
            i += 1
        
        return tokens
    
    def encode_bytelogic(self, bytelog_code: str) -> List[int]:
        """Convert ByteLogic code to token IDs."""
        tokens = self.tokenize_bytelogic(bytelog_code)
        token_ids = []
        
        for token in tokens:
            # Special handling for variables
            if token.startswith('$') and token[1:].isdigit():
                var_num = int(token[1:])
                token_id = self.token_to_id.get(f"var_{var_num}", self.unk_token_id)
            
            # Special handling for numbers
            elif self.number_pattern.fullmatch(token):
                if '.' in token:
                    # Float
                    token_id = self.token_to_id.get(f"num_float", self.unk_token_id)
                else:
                    # Integer
                    num_val = int(token) if int(token) < 1000 else 999
                    token_id = self.token_to_id.get(f"num_{abs(num_val)}", self.unk_token_id)
            
            # Special handling for quoted strings
            elif token.startswith('"') and token.endswith('"'):
                # Extract content and map to atom if known
                content = token[1:-1]
                if content in self.common_atoms:
                    token_id = self.token_to_id.get(content, self.unk_token_id)
                else:
                    token_id = self.token_to_id.get("atom_string", self.unk_token_id)
            
            # Regular token lookup
            else:
                token_id = self.token_to_id.get(token, self.unk_token_id)
            
            token_ids.append(token_id)
        
        return token_ids
    
    def decode_bytelogic(self, token_ids: List[int]) -> str:
        """Convert token IDs back to ByteLogic code."""
        tokens = []
        for token_id in token_ids:
            token = self.id_to_token.get(token_id, self.unk_token)
            
            # Handle special token types
            if token.startswith("var_"):
                var_num = token.split("_")[1]
                tokens.append(f"${var_num}")
            elif token.startswith("num_"):
                if token == "num_float":
                    tokens.append("0.0")
                else:
                    num = token.split("_")[1]
                    tokens.append(num)
            elif token.startswith("atom_"):
                tokens.append("unknown_atom")
            else:
                tokens.append(token)
        
        return " ".join(tokens)
    
    def validate_bytelogic_syntax(self, bytelog_code: str) -> Tuple[bool, Optional[str]]:
        """
        Basic syntax validation for ByteLogic code.
        
        Returns:
            (is_valid, error_message)
        """
        try:
            tokens = self.tokenize_bytelogic(bytelog_code)
            
            # Basic structure validation
            has_rel_or_calc = any(token in {"REL", "CALC"} for token in tokens)
            has_solve_or_result = any(token in {"SOLVE", "RESULT"} for token in tokens)
            
            if not has_rel_or_calc:
                return False, "Missing REL declaration or CALC definition"
            
            if "REL" in tokens and "SOLVE" not in tokens:
                return False, "Logic program missing SOLVE statement"
            
            if "CALC" in tokens and "RESULT" not in tokens:
                return False, "Calculation missing RESULT statement"
            
            # Check for balanced parentheses
            paren_count = 0
            for token in tokens:
                if token == "(":
                    paren_count += 1
                elif token == ")":
                    paren_count -= 1
                if paren_count < 0:
                    return False, "Unmatched closing parenthesis"
            
            if paren_count != 0:
                return False, "Unmatched opening parenthesis"
            
            # Check for valid variable usage
            variables_used = set()
            for token in tokens:
                if token.startswith('$') and token[1:].isdigit():
                    variables_used.add(int(token[1:]))
            
            # Variables should be sequential starting from 0
            if variables_used:
                max_var = max(variables_used)
                expected_vars = set(range(max_var + 1))
                if variables_used != expected_vars:
                    return False, f"Non-sequential variable usage: expected {expected_vars}, got {variables_used}"
            
            return True, None
            
        except Exception as e:
            return False, f"Syntax validation error: {str(e)}"
    
    def format_for_training(self, text: str, computation_code: str, expected_result: str) -> Dict:
        """
        Format a training example with computation tokens.
        
        Args:
            text: Human-readable input/question
            computation_code: ByteLogic code
            expected_result: Expected output result
            
        Returns:
            Formatted training example
        """
        # Validate syntax
        is_valid, error = self.validate_bytelogic_syntax(computation_code)
        
        return {
            "input": text,
            "computation_code": computation_code,
            "expected_result": expected_result,
            "syntax_valid": is_valid,
            "syntax_error": error,
            "token_count": len(self.tokenize_bytelogic(computation_code)),
            "formatted_output": f"{text} <computation>\n{computation_code}\n</computation> → {expected_result}"
        }
    
    def process_computation_in_text(self, text: str) -> List[Dict]:
        """
        Extract and process computation tokens from text.
        
        Returns:
            List of computation token information
        """
        computations = []
        matches = self.extract_computation_tokens(text)
        
        for full_match, bytelog_code in matches:
            is_valid, error = self.validate_bytelogic_syntax(bytelog_code)
            
            computations.append({
                "full_match": full_match,
                "bytelog_code": bytelog_code,
                "tokens": self.tokenize_bytelogic(bytelog_code),
                "token_ids": self.encode_bytelogic(bytelog_code),
                "is_valid": is_valid,
                "error": error
            })
        
        return computations