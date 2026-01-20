"""
ByteLog Lexer

Tokenizes ByteLog source code according to the lexical specification.
Converts source text into a stream of tokens for parsing.
"""

import re
import enum
from dataclasses import dataclass
from typing import List, Iterator, Optional


class TokenType(enum.Enum):
    # Keywords
    REL = "REL"
    FACT = "FACT" 
    RULE = "RULE"
    SCAN = "SCAN"
    JOIN = "JOIN"
    EMIT = "EMIT"
    MATCH = "MATCH"
    SOLVE = "SOLVE"
    QUERY = "QUERY"
    
    # Symbols
    COLON = ":"
    COMMA = ","
    WILDCARD = "?"
    
    # Literals
    VARIABLE = "VARIABLE"
    INTEGER = "INTEGER"
    IDENTIFIER = "IDENTIFIER"
    
    # Special
    EOF = "EOF"
    NEWLINE = "NEWLINE"


@dataclass
class Token:
    type: TokenType
    value: str
    line: int
    column: int


class LexicalError(Exception):
    def __init__(self, message: str, line: int, column: int):
        self.message = message
        self.line = line
        self.column = column
        super().__init__(f"Line {line}:{column}: {message}")


class Lexer:
    """ByteLog lexer that converts source text to tokens."""
    
    KEYWORDS = {
        'REL': TokenType.REL,
        'FACT': TokenType.FACT,
        'RULE': TokenType.RULE,
        'SCAN': TokenType.SCAN,
        'JOIN': TokenType.JOIN,
        'EMIT': TokenType.EMIT,
        'MATCH': TokenType.MATCH,
        'SOLVE': TokenType.SOLVE,
        'QUERY': TokenType.QUERY,
    }
    
    TOKEN_PATTERNS = [
        # Comments (skip)
        (r';[^\n]*', None),
        (r'//[^\n]*', None),
        
        # Symbols
        (r':', TokenType.COLON),
        (r',', TokenType.COMMA),
        (r'\?', TokenType.WILDCARD),
        
        # Variables: $0, $1, $2, ...
        (r'\$\d+', TokenType.VARIABLE),
        
        # Integer literals (including negative)
        (r'-?\d+', TokenType.INTEGER),
        
        # Identifiers (keywords checked separately)
        (r'[a-zA-Z_][a-zA-Z0-9_]*', TokenType.IDENTIFIER),
    ]
    
    def __init__(self, source: str):
        self.source = source
        self.pos = 0
        self.line = 1
        self.column = 1
        self.tokens = []
        
        # Compile regex patterns
        self.patterns = []
        for pattern, token_type in self.TOKEN_PATTERNS:
            self.patterns.append((re.compile(pattern), token_type))
    
    def current_char(self) -> Optional[str]:
        """Get current character or None if at end."""
        if self.pos >= len(self.source):
            return None
        return self.source[self.pos]
    
    def peek_char(self, offset: int = 1) -> Optional[str]:
        """Look ahead at character without advancing position."""
        peek_pos = self.pos + offset
        if peek_pos >= len(self.source):
            return None
        return self.source[peek_pos]
    
    def advance(self) -> Optional[str]:
        """Move to next character and return it."""
        if self.pos >= len(self.source):
            return None
        
        char = self.source[self.pos]
        self.pos += 1
        
        if char == '\n':
            self.line += 1
            self.column = 1
        else:
            self.column += 1
            
        return char
    
    def skip_whitespace(self):
        """Skip whitespace characters except newlines."""
        while self.current_char() in [' ', '\t', '\r']:
            self.advance()
    
    def match_token(self) -> Optional[Token]:
        """Try to match a token at current position."""
        start_pos = self.pos
        start_column = self.column
        
        # Try each pattern
        for pattern, token_type in self.patterns:
            match = pattern.match(self.source, self.pos)
            if match:
                value = match.group(0)
                
                # Move position forward
                for _ in range(len(value)):
                    self.advance()
                
                # Skip comments
                if token_type is None:
                    return self.match_token()
                
                # Check if identifier is actually a keyword
                if token_type == TokenType.IDENTIFIER:
                    upper_value = value.upper()
                    if upper_value in self.KEYWORDS:
                        token_type = self.KEYWORDS[upper_value]
                
                return Token(token_type, value, self.line, start_column)
        
        return None
    
    def tokenize(self) -> List[Token]:
        """Tokenize the entire source and return list of tokens."""
        tokens = []
        
        while self.pos < len(self.source):
            # Skip whitespace
            self.skip_whitespace()
            
            # Check for end of input
            if self.pos >= len(self.source):
                break
            
            # Handle newlines specially
            if self.current_char() == '\n':
                tokens.append(Token(TokenType.NEWLINE, '\n', self.line, self.column))
                self.advance()
                continue
            
            # Try to match a token
            token = self.match_token()
            if token:
                tokens.append(token)
            else:
                # Error: unexpected character
                char = self.current_char()
                raise LexicalError(f"Unexpected character '{char}'", self.line, self.column)
        
        # Add EOF token
        tokens.append(Token(TokenType.EOF, '', self.line, self.column))
        return tokens
    
    def tokens_iter(self) -> Iterator[Token]:
        """Return iterator over tokens."""
        for token in self.tokenize():
            yield token


def tokenize(source: str) -> List[Token]:
    """Convenience function to tokenize source text."""
    lexer = Lexer(source)
    return lexer.tokenize()