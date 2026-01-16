"""
Tag parser for WorldModel LLM experiment.

Extracts and validates <think>/<model>/<requires> blocks from model output,
providing clean API for tag extraction and validation.
"""

import re
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum

from ..utils.logging import get_logger

logger = get_logger('tagParser')


class TagType(Enum):
    """Supported tag types in the WorldModel system."""
    THINK = "think"
    MODEL = "model"
    REQUIRES = "requires"


@dataclass
class ParsedTag:
    """Represents a parsed tag with its content and metadata."""
    tag_type: TagType
    content: str
    start_pos: int
    end_pos: int
    attributes: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.attributes is None:
            self.attributes = {}


@dataclass
class ModelTag(ParsedTag):
    """Specialized tag for <model> content with language and code."""
    language: str = ""
    code: str = ""
    
    def __post_init__(self):
        super().__post_init__()
        self.tag_type = TagType.MODEL
        self._parse_model_content()
    
    def _parse_model_content(self):
        """Parse model tag content to extract language and code."""
        # Pattern: <model>language: code</model> or <model>code</model>
        content = self.content.strip()
        
        # Check for language prefix
        if ':' in content:
            parts = content.split(':', 1)
            if len(parts) == 2:
                self.language = parts[0].strip().lower()
                self.code = parts[1].strip()
            else:
                self.language = "unknown"
                self.code = content
        else:
            # No language specified, try to infer or default to python
            self.language = self._infer_language(content)
            self.code = content
        
        self.attributes.update({
            'language': self.language,
            'code': self.code
        })
    
    def _infer_language(self, code: str) -> str:
        """Infer programming language from code content."""
        code_lower = code.lower().strip()
        
        # Simple heuristics for language detection
        if any(keyword in code_lower for keyword in ['console.log', 'const ', 'let ', 'var ', 'function']):
            return "javascript"
        elif any(keyword in code_lower for keyword in ['#include', 'printf', 'int main']):
            return "c"
        elif any(keyword in code_lower for keyword in ['echo', 'ls', 'grep', 'cat', 'cd']):
            return "bash"
        elif any(keyword in code_lower for keyword in ['print(', 'import ', 'def ', 'class ']):
            return "python"
        else:
            # Default to python for unrecognized code
            return "python"


@dataclass
class RequiresTag(ParsedTag):
    """Specialized tag for <requires> content with parsed requirements."""
    requirements: List[str] = None
    
    def __post_init__(self):
        super().__post_init__()
        self.tag_type = TagType.REQUIRES
        self.requirements = []
        self._parse_requirements()
    
    def _parse_requirements(self):
        """Parse requirements from comma-separated list."""
        content = self.content.strip()
        if content:
            # Split by comma and clean up each requirement
            raw_requirements = [req.strip() for req in content.split(',')]
            self.requirements = [req for req in raw_requirements if req]
        
        self.attributes['requirements'] = self.requirements


@dataclass
class ParseResult:
    """Result of parsing model output for tags."""
    original_text: str
    think_tags: List[ParsedTag]
    model_tags: List[ModelTag]
    requires_tags: List[RequiresTag]
    parsing_errors: List[str]
    
    @property
    def has_think(self) -> bool:
        """Check if any think tags were found."""
        return len(self.think_tags) > 0
    
    @property
    def has_model(self) -> bool:
        """Check if any model tags were found."""
        return len(self.model_tags) > 0
    
    @property
    def has_requires(self) -> bool:
        """Check if any requires tags were found."""
        return len(self.requires_tags) > 0
    
    @property
    def has_tags(self) -> bool:
        """Check if any tags were found."""
        return self.has_think or self.has_model or self.has_requires
    
    def get_first_model(self) -> Optional[ModelTag]:
        """Get the first model tag if available."""
        return self.model_tags[0] if self.model_tags else None
    
    def get_first_requires(self) -> Optional[RequiresTag]:
        """Get the first requires tag if available."""
        return self.requires_tags[0] if self.requires_tags else None
    
    def get_all_requirements(self) -> List[str]:
        """Get all requirements from all requires tags."""
        all_reqs = []
        for req_tag in self.requires_tags:
            all_reqs.extend(req_tag.requirements)
        return all_reqs


class TagParser:
    """Parser for extracting and validating WorldModel tags."""
    
    def __init__(self):
        self.logger = get_logger('tagParser')
        
        # Regex patterns for different tag types
        self.tag_patterns = {
            TagType.THINK: re.compile(r'<think>(.*?)</think>', re.DOTALL | re.IGNORECASE),
            TagType.MODEL: re.compile(r'<model>(.*?)</model>', re.DOTALL | re.IGNORECASE),
            TagType.REQUIRES: re.compile(r'<requires>(.*?)</requires>', re.DOTALL | re.IGNORECASE)
        }
        
        # Valid requirement categories
        self.valid_languages = {'python', 'javascript', 'bash', 'c', 'js'}
        self.valid_categories = {
            'math', 'data_processing', 'web', 'file', 'system', 'computation',
            'file_read', 'file_write', 'network', 'database', 'ml', 'visualization'
        }
    
    def parse(self, text: str) -> ParseResult:
        """Parse text and extract all WorldModel tags."""
        self.logger.debug(f"Parsing text of length {len(text)}")
        
        result = ParseResult(
            original_text=text,
            think_tags=[],
            model_tags=[],
            requires_tags=[],
            parsing_errors=[]
        )
        
        try:
            # Extract think tags
            result.think_tags = self._extract_think_tags(text)
            
            # Extract model tags
            result.model_tags = self._extract_model_tags(text)
            
            # Extract requires tags
            result.requires_tags = self._extract_requires_tags(text)
            
            # Validate the parsed results
            self._validate_tags(result)
            
            self.logger.info(f"Parsed {len(result.think_tags)} think, "
                           f"{len(result.model_tags)} model, "
                           f"{len(result.requires_tags)} requires tags")
            
        except Exception as e:
            error_msg = f"Error parsing tags: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            result.parsing_errors.append(error_msg)
        
        return result
    
    def _extract_think_tags(self, text: str) -> List[ParsedTag]:
        """Extract all <think> tags from text."""
        tags = []
        pattern = self.tag_patterns[TagType.THINK]
        
        for match in pattern.finditer(text):
            tag = ParsedTag(
                tag_type=TagType.THINK,
                content=match.group(1).strip(),
                start_pos=match.start(),
                end_pos=match.end()
            )
            tags.append(tag)
        
        return tags
    
    def _extract_model_tags(self, text: str) -> List[ModelTag]:
        """Extract all <model> tags from text."""
        tags = []
        pattern = self.tag_patterns[TagType.MODEL]
        
        for match in pattern.finditer(text):
            tag = ModelTag(
                tag_type=TagType.MODEL,
                content=match.group(1).strip(),
                start_pos=match.start(),
                end_pos=match.end()
            )
            tags.append(tag)
        
        return tags
    
    def _extract_requires_tags(self, text: str) -> List[RequiresTag]:
        """Extract all <requires> tags from text."""
        tags = []
        pattern = self.tag_patterns[TagType.REQUIRES]
        
        for match in pattern.finditer(text):
            tag = RequiresTag(
                tag_type=TagType.REQUIRES,
                content=match.group(1).strip(),
                start_pos=match.start(),
                end_pos=match.end()
            )
            tags.append(tag)
        
        return tags
    
    def _validate_tags(self, result: ParseResult):
        """Validate extracted tags and add any validation errors."""
        # Validate model tags
        for model_tag in result.model_tags:
            self._validate_model_tag(model_tag, result.parsing_errors)
        
        # Validate requires tags
        for req_tag in result.requires_tags:
            self._validate_requires_tag(req_tag, result.parsing_errors)
        
        # Check for logical consistency
        self._validate_tag_consistency(result)
    
    def _validate_model_tag(self, model_tag: ModelTag, errors: List[str]):
        """Validate a model tag."""
        # Check if language is recognized
        if model_tag.language and model_tag.language not in self.valid_languages:
            errors.append(f"Unrecognized language: {model_tag.language}")
        
        # Check if code is not empty
        if not model_tag.code.strip():
            errors.append("Model tag contains empty code")
        
        # Basic syntax checks per language
        if model_tag.language == 'python':
            self._validate_python_code(model_tag.code, errors)
        elif model_tag.language in ['javascript', 'js']:
            self._validate_javascript_code(model_tag.code, errors)
    
    def _validate_python_code(self, code: str, errors: List[str]):
        """Basic validation for Python code."""
        import textwrap
        
        # Always try to dedent the code first to handle extracted indented blocks
        try:
            dedented_code = textwrap.dedent(code).strip()
            if dedented_code:  # Only validate non-empty code
                compile(dedented_code, '<model_tag>', 'exec')
        except SyntaxError as e:
            errors.append(f"Python syntax error: {str(e)}")
        except Exception:
            # For any other compilation issues, skip validation
            # This is for extracted code that might have context dependencies
            pass
    
    def _validate_javascript_code(self, code: str, errors: List[str]):
        """Basic validation for JavaScript code."""
        # Simple heuristic checks for common JS syntax issues
        if code.count('{') != code.count('}'):
            errors.append("JavaScript: Mismatched braces")
        if code.count('(') != code.count(')'):
            errors.append("JavaScript: Mismatched parentheses")
    
    def _validate_requires_tag(self, req_tag: RequiresTag, errors: List[str]):
        """Validate a requires tag."""
        for requirement in req_tag.requirements:
            self._validate_requirement(requirement, errors)
    
    def _validate_requirement(self, requirement: str, errors: List[str]):
        """Validate a single requirement string."""
        # Expected format: language:category(optional_domain)
        # Examples: python:math, python:web(wikipedia.com), bash:file_read
        
        if ':' not in requirement:
            errors.append(f"Invalid requirement format (missing ':'): {requirement}")
            return
        
        parts = requirement.split(':', 1)
        if len(parts) != 2:
            errors.append(f"Invalid requirement format: {requirement}")
            return
        
        language, category_part = parts
        language = language.strip().lower()
        category_part = category_part.strip().lower()
        
        # Validate language
        if language not in self.valid_languages:
            errors.append(f"Invalid language in requirement: {language}")
        
        # Extract category and optional domain
        category = category_part
        domain = None
        
        if '(' in category_part and ')' in category_part:
            paren_start = category_part.index('(')
            paren_end = category_part.rindex(')')
            category = category_part[:paren_start].strip()
            domain = category_part[paren_start+1:paren_end].strip()
        
        # Validate category
        if category not in self.valid_categories:
            errors.append(f"Invalid category in requirement: {category}")
        
        # Validate domain if present
        if domain and not self._validate_domain(domain):
            errors.append(f"Invalid domain in requirement: {domain}")
    
    def _validate_domain(self, domain: str) -> bool:
        """Validate a domain specification."""
        # Basic domain validation - could be expanded
        if not domain:
            return False
        
        # Allow read_only, write_only specifiers
        if domain in ['read_only', 'write_only']:
            return True
        
        # Allow website domains
        if '.' in domain and len(domain.split('.')) >= 2:
            return True
        
        return False
    
    def _validate_tag_consistency(self, result: ParseResult):
        """Validate consistency between different tag types."""
        # If there's a model tag, there should be corresponding requires
        if result.model_tags and not result.requires_tags:
            result.parsing_errors.append(
                "Model tag found without corresponding requires tag"
            )
        
        # Check if requires match the model language
        for model_tag in result.model_tags:
            model_lang = model_tag.language
            all_reqs = result.get_all_requirements()
            
            # Check if any requirement matches the model language
            has_matching_req = any(
                req.startswith(f"{model_lang}:")
                for req in all_reqs
            )
            
            if not has_matching_req:
                result.parsing_errors.append(
                    f"Model uses {model_lang} but no {model_lang} requirements found"
                )
    
    def extract_clean_text(self, text: str) -> str:
        """Extract text with all tags removed."""
        clean_text = text
        
        # Remove all tags in order
        for pattern in self.tag_patterns.values():
            clean_text = pattern.sub('', clean_text)
        
        # Clean up extra whitespace
        clean_text = re.sub(r'\s+', ' ', clean_text).strip()
        
        return clean_text
    
    def has_any_tags(self, text: str) -> bool:
        """Quick check if text contains any WorldModel tags."""
        for pattern in self.tag_patterns.values():
            if pattern.search(text):
                return True
        return False
    
    def get_tag_positions(self, text: str) -> List[Tuple[int, int, TagType]]:
        """Get positions of all tags in the text."""
        positions = []
        
        for tag_type, pattern in self.tag_patterns.items():
            for match in pattern.finditer(text):
                positions.append((match.start(), match.end(), tag_type))
        
        # Sort by start position
        positions.sort(key=lambda x: x[0])
        
        return positions


# Convenience functions
def parse_tags(text: str) -> ParseResult:
    """Parse text and return tag extraction results."""
    parser = TagParser()
    return parser.parse(text)

def extract_model_code(text: str) -> Optional[Tuple[str, str]]:
    """Extract first model code and language. Returns (language, code) or None."""
    result = parse_tags(text)
    model_tag = result.get_first_model()
    if model_tag:
        return (model_tag.language, model_tag.code)
    return None

def extract_requirements(text: str) -> List[str]:
    """Extract all requirements from text."""
    result = parse_tags(text)
    return result.get_all_requirements()

def has_think_tag(text: str) -> bool:
    """Check if text contains any think tags."""
    parser = TagParser()
    return bool(parser.tag_patterns[TagType.THINK].search(text))

def clean_text(text: str) -> str:
    """Remove all WorldModel tags from text."""
    parser = TagParser()
    return parser.extract_clean_text(text)