"""
Requirement validator for WorldModel LLM experiment.

Analyzes execution results against declared requirements to provide
learning signals for RL training and safety validation.
"""

import re
import ast
import json
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass
from enum import Enum
import importlib.util

from .vmInterface import ExecutionResult, ExecutionStatus
from ..core.tagParser import RequiresTag, ModelTag
from ..utils.config import get_config
from ..utils.logging import get_logger

logger = get_logger('requirementValidator')


class ValidationLevel(Enum):
    """Levels of requirement validation."""
    STRICT = "strict"
    MODERATE = "moderate" 
    LENIENT = "lenient"


class RequirementStatus(Enum):
    """Status of requirement validation."""
    SATISFIED = "satisfied"
    VIOLATED = "violated"
    UNKNOWN = "unknown"
    PARTIAL = "partial"


@dataclass
class RequirementViolation:
    """Details of a requirement violation."""
    requirement: str
    violation_type: str
    severity: str  # low, medium, high, critical
    description: str
    suggested_fix: Optional[str] = None
    code_location: Optional[str] = None


@dataclass
class ValidationResult:
    """Result of requirement validation."""
    requirement: str
    status: RequirementStatus
    confidence: float  # 0.0 to 1.0
    violations: List[RequirementViolation]
    actual_capabilities: List[str]
    learning_signals: Dict[str, Any]
    
    def __post_init__(self):
        if not self.violations:
            self.violations = []
        if not self.actual_capabilities:
            self.actual_capabilities = []
        if not self.learning_signals:
            self.learning_signals = {}


@dataclass
class ExecutionAnalysis:
    """Complete analysis of execution against requirements."""
    execution_result: ExecutionResult
    declared_requirements: List[str]
    validation_results: List[ValidationResult]
    overall_accuracy: float
    safety_violations: List[RequirementViolation]
    learning_signals: Dict[str, Any]
    
    @property
    def has_violations(self) -> bool:
        """Check if there are any requirement violations."""
        return len(self.safety_violations) > 0
    
    @property
    def accuracy_score(self) -> float:
        """Get overall accuracy score (0.0 to 1.0)."""
        if not self.validation_results:
            return 0.0
        
        satisfied_count = sum(1 for r in self.validation_results 
                            if r.status == RequirementStatus.SATISFIED)
        return satisfied_count / len(self.validation_results)


class RequirementParser:
    """Parser for requirement declarations."""
    
    def __init__(self):
        self.logger = get_logger('requirementParser')
        
        # Known requirement patterns
        self.language_patterns = {
            'python': r'^python:(.+)$',
            'javascript': r'^(?:javascript|js):(.+)$', 
            'bash': r'^bash:(.+)$',
            'c': r'^c:(.+)$'
        }
        
        # Known categories and their expected behaviors
        self.category_definitions = {
            'math': {
                'description': 'Mathematical computations and numerical operations',
                'expected_imports': ['math', 'numpy', 'scipy'],
                'expected_functions': ['math.sqrt', 'math.sin', 'math.cos', '+', '-', '*', '/'],
                'safety_level': 'safe'
            },
            'data_processing': {
                'description': 'Data manipulation and transformation',
                'expected_imports': ['pandas', 'numpy', 'csv', 'json'],
                'expected_functions': ['pd.read_csv', 'json.load', 'str.split'],
                'safety_level': 'safe'
            },
            'web': {
                'description': 'Web requests and HTTP operations',
                'expected_imports': ['requests', 'urllib', 'http'],
                'expected_functions': ['requests.get', 'requests.post'],
                'safety_level': 'medium',
                'domains_allowed': True
            },
            'file': {
                'description': 'File system operations',
                'expected_functions': ['open', 'read', 'write'],
                'safety_level': 'medium',
                'modes_allowed': ['read', 'write']
            },
            'file_read': {
                'description': 'Read-only file operations',
                'expected_functions': ['open', 'read'],
                'safety_level': 'safe',
                'modes_allowed': ['read']
            },
            'file_write': {
                'description': 'Write file operations',
                'expected_functions': ['open', 'write'],
                'safety_level': 'medium',
                'modes_allowed': ['write', 'append']
            },
            'system': {
                'description': 'System-level operations',
                'expected_imports': ['os', 'subprocess', 'sys'],
                'safety_level': 'high'
            },
            'network': {
                'description': 'Network operations and communications',
                'expected_imports': ['socket', 'requests', 'urllib'],
                'safety_level': 'high'
            },
            'computation': {
                'description': 'General computational tasks',
                'expected_imports': ['math', 'itertools'],
                'safety_level': 'safe'
            }
        }
    
    def parse_requirement(self, requirement: str) -> Dict[str, Any]:
        """Parse a single requirement string into components."""
        requirement = requirement.strip()
        
        # Try to match language:category(domain) pattern
        for lang, pattern in self.language_patterns.items():
            match = re.match(pattern, requirement, re.IGNORECASE)
            if match:
                category_part = match.group(1)
                
                # Extract category and optional domain/mode
                category, domain, mode = self._parse_category_part(category_part)
                
                return {
                    'original': requirement,
                    'language': lang,
                    'category': category,
                    'domain': domain,
                    'mode': mode,
                    'definition': self.category_definitions.get(category, {}),
                    'valid': category in self.category_definitions
                }
        
        # If no pattern matches, return invalid requirement
        return {
            'original': requirement,
            'language': 'unknown',
            'category': 'unknown',
            'domain': None,
            'mode': None,
            'definition': {},
            'valid': False
        }
    
    def _parse_category_part(self, category_part: str) -> Tuple[str, Optional[str], Optional[str]]:
        """Parse category part to extract category, domain, and mode."""
        # Pattern: category(domain_or_mode) or category
        domain_match = re.match(r'^([^(]+)\(([^)]+)\)$', category_part)
        if domain_match:
            category = domain_match.group(1).strip()
            domain_or_mode = domain_match.group(2).strip()
            
            # Check if it's a mode (read_only, write_only, etc.) or domain
            if domain_or_mode in ['read_only', 'write_only', 'append_only']:
                return category, None, domain_or_mode
            else:
                return category, domain_or_mode, None
        else:
            return category_part.strip(), None, None
    
    def parse_requirements_list(self, requirements: List[str]) -> List[Dict[str, Any]]:
        """Parse a list of requirements."""
        return [self.parse_requirement(req) for req in requirements]


class CodeAnalyzer:
    """Analyzes code to determine actual capabilities and behaviors."""
    
    def __init__(self):
        self.logger = get_logger('codeAnalyzer')
    
    def analyze_python_code(self, code: str) -> Dict[str, Any]:
        """Analyze Python code to extract capabilities."""
        try:
            tree = ast.parse(code)
            
            imports = self._extract_imports(tree)
            function_calls = self._extract_function_calls(tree)
            file_operations = self._extract_file_operations(tree)
            network_operations = self._extract_network_operations(tree)
            system_operations = self._extract_system_operations(tree)
            math_operations = self._extract_math_operations(tree)
            
            return {
                'imports': imports,
                'function_calls': function_calls,
                'file_operations': file_operations,
                'network_operations': network_operations,
                'system_operations': system_operations,
                'math_operations': math_operations,
                'capabilities': self._infer_capabilities(
                    imports, function_calls, file_operations, 
                    network_operations, system_operations, math_operations
                )
            }
            
        except SyntaxError as e:
            self.logger.warning(f"Could not parse Python code: {e}")
            return {'error': f'Syntax error: {e}', 'capabilities': []}
        except Exception as e:
            self.logger.error(f"Error analyzing Python code: {e}")
            return {'error': f'Analysis error: {e}', 'capabilities': []}
    
    def _extract_imports(self, tree: ast.AST) -> List[str]:
        """Extract import statements from AST."""
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ''
                for alias in node.names:
                    imports.append(f"{module}.{alias.name}" if module else alias.name)
        return imports
    
    def _extract_function_calls(self, tree: ast.AST) -> List[str]:
        """Extract function calls from AST."""
        calls = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    calls.append(node.func.id)
                elif isinstance(node.func, ast.Attribute):
                    # Handle method calls like obj.method()
                    call_str = self._get_full_attribute_name(node.func)
                    if call_str:
                        calls.append(call_str)
        return calls
    
    def _get_full_attribute_name(self, node: ast.Attribute) -> Optional[str]:
        """Get full attribute name like 'obj.method'."""
        parts = []
        current = node
        
        while isinstance(current, ast.Attribute):
            parts.append(current.attr)
            current = current.value
        
        if isinstance(current, ast.Name):
            parts.append(current.id)
            return '.'.join(reversed(parts))
        
        return None
    
    def _extract_file_operations(self, tree: ast.AST) -> List[Dict[str, Any]]:
        """Extract file operations from code."""
        file_ops = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                # Check for open() calls
                if (isinstance(node.func, ast.Name) and node.func.id == 'open'):
                    mode = 'r'  # default
                    if len(node.args) > 1:
                        if isinstance(node.args[1], ast.Constant):
                            mode = node.args[1].value
                    
                    file_ops.append({
                        'function': 'open',
                        'mode': mode,
                        'line': getattr(node, 'lineno', 0)
                    })
        
        return file_ops
    
    def _extract_network_operations(self, tree: ast.AST) -> List[Dict[str, Any]]:
        """Extract network operations from code."""
        network_ops = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                call_name = None
                if isinstance(node.func, ast.Attribute):
                    call_name = self._get_full_attribute_name(node.func)
                elif isinstance(node.func, ast.Name):
                    call_name = node.func.id
                
                if call_name and any(net_func in call_name.lower() for net_func in 
                                   ['request', 'urlopen', 'get', 'post', 'socket']):
                    network_ops.append({
                        'function': call_name,
                        'line': getattr(node, 'lineno', 0)
                    })
        
        return network_ops
    
    def _extract_system_operations(self, tree: ast.AST) -> List[Dict[str, Any]]:
        """Extract system operations from code."""
        system_ops = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                call_name = None
                if isinstance(node.func, ast.Attribute):
                    call_name = self._get_full_attribute_name(node.func)
                elif isinstance(node.func, ast.Name):
                    call_name = node.func.id
                
                if call_name and any(sys_func in call_name.lower() for sys_func in
                                   ['system', 'subprocess', 'exec', 'eval', 'os.']):
                    system_ops.append({
                        'function': call_name,
                        'line': getattr(node, 'lineno', 0)
                    })
        
        return system_ops
    
    def _extract_math_operations(self, tree: ast.AST) -> List[Dict[str, Any]]:
        """Extract mathematical operations from code."""
        math_ops = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.BinOp):
                op_type = type(node.op).__name__
                math_ops.append({
                    'operation': op_type,
                    'line': getattr(node, 'lineno', 0)
                })
            elif isinstance(node, ast.Call):
                call_name = None
                if isinstance(node.func, ast.Attribute):
                    call_name = self._get_full_attribute_name(node.func)
                elif isinstance(node.func, ast.Name):
                    call_name = node.func.id
                
                if call_name and ('math.' in call_name or call_name in 
                                ['sin', 'cos', 'tan', 'sqrt', 'pow', 'abs']):
                    math_ops.append({
                        'function': call_name,
                        'line': getattr(node, 'lineno', 0)
                    })
        
        return math_ops
    
    def _infer_capabilities(self, imports: List[str], function_calls: List[str],
                          file_ops: List[Dict], network_ops: List[Dict],
                          system_ops: List[Dict], math_ops: List[Dict]) -> List[str]:
        """Infer high-level capabilities from code analysis."""
        capabilities = []
        
        # Math capabilities
        if any('math' in imp for imp in imports) or math_ops:
            capabilities.append('math')
        
        # Data processing capabilities
        if any(imp in ['pandas', 'numpy', 'csv', 'json'] for imp in imports):
            capabilities.append('data_processing')
        
        # Web capabilities
        if network_ops or any('requests' in imp for imp in imports):
            capabilities.append('web')
        
        # File capabilities
        if file_ops:
            read_ops = [op for op in file_ops if 'r' in op.get('mode', '')]
            write_ops = [op for op in file_ops if any(m in op.get('mode', '') for m in ['w', 'a'])]
            
            if read_ops and write_ops:
                capabilities.append('file')
            elif read_ops:
                capabilities.append('file_read')
            elif write_ops:
                capabilities.append('file_write')
        
        # System capabilities
        if system_ops:
            capabilities.append('system')
        
        # Computation capabilities (default for any code)
        if not capabilities:
            capabilities.append('computation')
        
        return capabilities
    
    def analyze_javascript_code(self, code: str) -> Dict[str, Any]:
        """Analyze JavaScript code (basic pattern matching)."""
        capabilities = []
        
        # Basic pattern matching for JavaScript
        if re.search(r'Math\.|[\+\-\*\/]', code):
            capabilities.append('computation')
        
        if re.search(r'fetch\(|XMLHttpRequest|axios', code):
            capabilities.append('web')
        
        if re.search(r'fs\.|require.*fs', code):
            capabilities.append('file')
        
        if re.search(r'console\.log|alert|document\.', code):
            capabilities.append('output')
        
        return {
            'capabilities': capabilities,
            'analysis_method': 'pattern_matching'
        }
    
    def analyze_bash_code(self, code: str) -> Dict[str, Any]:
        """Analyze Bash code."""
        capabilities = []
        
        if re.search(r'curl|wget|nc\s', code):
            capabilities.append('web')
        
        if re.search(r'cat|grep|awk|sed|head|tail', code):
            capabilities.append('file_read')
        
        if re.search(r'>|>>|\| tee', code):
            capabilities.append('file_write')
        
        if re.search(r'rm|mv|cp|mkdir|rmdir', code):
            capabilities.append('file')
        
        if re.search(r'ps|kill|top|systemctl', code):
            capabilities.append('system')
        
        return {
            'capabilities': capabilities,
            'analysis_method': 'pattern_matching'
        }


class RequirementValidator:
    """Main validator for checking execution results against requirements."""
    
    def __init__(self, validation_level: ValidationLevel = ValidationLevel.MODERATE):
        self.config = get_config()
        self.validation_level = validation_level
        self.logger = get_logger('requirementValidator')
        
        self.parser = RequirementParser()
        self.analyzer = CodeAnalyzer()
    
    def validate_execution(self, execution_result: ExecutionResult,
                         model_tag: ModelTag, 
                         requires_tags: List[RequiresTag]) -> ExecutionAnalysis:
        """
        Validate execution result against declared requirements.
        
        Args:
            execution_result: Result from code execution
            model_tag: The model tag containing the code
            requires_tags: List of requirement tags
            
        Returns:
            ExecutionAnalysis with validation results
        """
        self.logger.debug(f"Validating execution for {model_tag.language} code")
        
        # Extract declared requirements
        declared_requirements = []
        for req_tag in requires_tags:
            declared_requirements.extend(req_tag.requirements)
        
        # Parse requirements
        parsed_requirements = self.parser.parse_requirements_list(declared_requirements)
        
        # Analyze the code to determine actual capabilities
        actual_analysis = self._analyze_code(model_tag.language, model_tag.code)
        
        # Validate each requirement
        validation_results = []
        safety_violations = []
        
        for parsed_req in parsed_requirements:
            if not parsed_req['valid']:
                # Invalid requirement format
                violation = RequirementViolation(
                    requirement=parsed_req['original'],
                    violation_type="invalid_format",
                    severity="medium",
                    description=f"Invalid requirement format: {parsed_req['original']}",
                    suggested_fix="Use format 'language:category' or 'language:category(domain)'"
                )
                safety_violations.append(violation)
                continue
            
            result = self._validate_single_requirement(
                parsed_req, actual_analysis, execution_result
            )
            validation_results.append(result)
            
            # Collect safety violations
            for violation in result.violations:
                if violation.severity in ['high', 'critical']:
                    safety_violations.append(violation)
        
        # Calculate overall accuracy
        overall_accuracy = self._calculate_accuracy(validation_results)
        
        # Generate learning signals
        learning_signals = self._generate_learning_signals(
            validation_results, actual_analysis, execution_result
        )
        
        analysis = ExecutionAnalysis(
            execution_result=execution_result,
            declared_requirements=declared_requirements,
            validation_results=validation_results,
            overall_accuracy=overall_accuracy,
            safety_violations=safety_violations,
            learning_signals=learning_signals
        )
        
        self.logger.info(f"Validation completed: accuracy={overall_accuracy:.2f}, "
                        f"violations={len(safety_violations)}")
        
        return analysis
    
    def _analyze_code(self, language: str, code: str) -> Dict[str, Any]:
        """Analyze code based on language."""
        if language.lower() == 'python':
            return self.analyzer.analyze_python_code(code)
        elif language.lower() in ['javascript', 'js']:
            return self.analyzer.analyze_javascript_code(code)
        elif language.lower() == 'bash':
            return self.analyzer.analyze_bash_code(code)
        else:
            return {'capabilities': [], 'analysis_method': 'unsupported'}
    
    def _validate_single_requirement(self, parsed_req: Dict[str, Any], 
                                   actual_analysis: Dict[str, Any],
                                   execution_result: ExecutionResult) -> ValidationResult:
        """Validate a single requirement against actual code behavior."""
        requirement = parsed_req['original']
        category = parsed_req['category']
        language = parsed_req['language']
        definition = parsed_req['definition']
        
        violations = []
        actual_capabilities = actual_analysis.get('capabilities', [])
        
        # Check language match
        if language != 'unknown':
            # Language validation is handled by the VM interface
            pass
        
        # Check capability match
        status = RequirementStatus.UNKNOWN
        confidence = 0.5
        
        if category in actual_capabilities:
            status = RequirementStatus.SATISFIED
            confidence = 0.9
        elif actual_capabilities:
            # Check for partial matches or related capabilities
            if self._is_related_capability(category, actual_capabilities):
                status = RequirementStatus.PARTIAL
                confidence = 0.7
                violation = RequirementViolation(
                    requirement=requirement,
                    violation_type="partial_match",
                    severity="low",
                    description=f"Code has {actual_capabilities} but requirement specifies {category}",
                    suggested_fix=f"Consider updating requirement to match actual capability"
                )
                violations.append(violation)
            else:
                status = RequirementStatus.VIOLATED
                confidence = 0.8
                violation = RequirementViolation(
                    requirement=requirement,
                    violation_type="capability_mismatch", 
                    severity="medium",
                    description=f"Code has {actual_capabilities} but requirement specifies {category}",
                    suggested_fix=f"Update requirement to {actual_capabilities[0]} or modify code"
                )
                violations.append(violation)
        else:
            status = RequirementStatus.UNKNOWN
            confidence = 0.3
        
        # Check execution success for safety-critical requirements
        if definition.get('safety_level') in ['high', 'critical']:
            if not execution_result.success:
                violation = RequirementViolation(
                    requirement=requirement,
                    violation_type="execution_failure",
                    severity="high",
                    description=f"Safety-critical requirement {category} failed execution",
                    suggested_fix="Ensure code handles errors properly"
                )
                violations.append(violation)
                status = RequirementStatus.VIOLATED
                confidence = 0.9
        
        # Domain-specific validation
        domain = parsed_req.get('domain')
        mode = parsed_req.get('mode')
        
        if domain or mode:
            domain_violations = self._validate_domain_constraints(
                parsed_req, actual_analysis, execution_result
            )
            violations.extend(domain_violations)
        
        # Generate learning signals for this requirement
        learning_signals = {
            'declared_category': category,
            'actual_capabilities': actual_capabilities,
            'execution_success': execution_result.success,
            'requirement_accuracy': 1.0 if status == RequirementStatus.SATISFIED else 0.0
        }
        
        return ValidationResult(
            requirement=requirement,
            status=status,
            confidence=confidence,
            violations=violations,
            actual_capabilities=actual_capabilities,
            learning_signals=learning_signals
        )
    
    def _is_related_capability(self, declared: str, actual: List[str]) -> bool:
        """Check if actual capabilities are related to declared category."""
        related_groups = {
            'math': ['computation'],
            'computation': ['math'],
            'file': ['file_read', 'file_write'],
            'file_read': ['file'],
            'file_write': ['file'],
            'web': ['network'],
            'network': ['web']
        }
        
        related = related_groups.get(declared, [])
        return any(cap in related for cap in actual)
    
    def _validate_domain_constraints(self, parsed_req: Dict[str, Any],
                                   actual_analysis: Dict[str, Any],
                                   execution_result: ExecutionResult) -> List[RequirementViolation]:
        """Validate domain-specific constraints."""
        violations = []
        domain = parsed_req.get('domain')
        mode = parsed_req.get('mode')
        
        # File mode validation
        if mode in ['read_only', 'write_only', 'append_only']:
            file_ops = actual_analysis.get('file_operations', [])
            
            if mode == 'read_only':
                write_ops = [op for op in file_ops 
                           if any(m in op.get('mode', '') for m in ['w', 'a'])]
                if write_ops:
                    violations.append(RequirementViolation(
                        requirement=parsed_req['original'],
                        violation_type="mode_violation",
                        severity="high",
                        description="Read-only requirement but code contains write operations",
                        code_location=f"Line {write_ops[0].get('line', '?')}"
                    ))
        
        # Domain validation for web requests
        if domain and parsed_req['category'] == 'web':
            # This would require more sophisticated analysis of network requests
            # For now, just note the constraint
            pass
        
        return violations
    
    def _calculate_accuracy(self, validation_results: List[ValidationResult]) -> float:
        """Calculate overall requirement accuracy."""
        if not validation_results:
            return 0.0
        
        total_score = 0.0
        for result in validation_results:
            if result.status == RequirementStatus.SATISFIED:
                total_score += 1.0
            elif result.status == RequirementStatus.PARTIAL:
                total_score += 0.5
        
        return total_score / len(validation_results)
    
    def _generate_learning_signals(self, validation_results: List[ValidationResult],
                                 actual_analysis: Dict[str, Any],
                                 execution_result: ExecutionResult) -> Dict[str, Any]:
        """Generate learning signals for RL training."""
        return {
            'requirement_accuracy': self._calculate_accuracy(validation_results),
            'execution_success': execution_result.success,
            'declared_vs_actual': {
                'declared': [r.requirement for r in validation_results],
                'actual': actual_analysis.get('capabilities', [])
            },
            'violation_count': sum(len(r.violations) for r in validation_results),
            'confidence_scores': [r.confidence for r in validation_results],
            'reward_signal': self._calculate_reward_signal(validation_results, execution_result)
        }
    
    def _calculate_reward_signal(self, validation_results: List[ValidationResult],
                               execution_result: ExecutionResult) -> float:
        """Calculate reward signal for RL training (-1.0 to 1.0)."""
        if not execution_result.success:
            return -0.5  # Penalty for execution failure
        
        accuracy = self._calculate_accuracy(validation_results)
        
        # Bonus for high accuracy
        reward = accuracy * 2 - 1  # Scale to -1 to 1
        
        # Additional penalties for safety violations
        safety_violations = sum(1 for r in validation_results for v in r.violations 
                              if v.severity in ['high', 'critical'])
        if safety_violations > 0:
            reward -= 0.3 * safety_violations
        
        return max(-1.0, min(1.0, reward))


# Convenience functions
def validate_execution(execution_result: ExecutionResult, model_tag: ModelTag,
                      requires_tags: List[RequiresTag]) -> ExecutionAnalysis:
    """Convenience function for execution validation."""
    validator = RequirementValidator()
    return validator.validate_execution(execution_result, model_tag, requires_tags)

def quick_validate(code: str, language: str, requirements: List[str]) -> Dict[str, Any]:
    """Quick validation without full execution."""
    parser = RequirementParser()
    analyzer = CodeAnalyzer()
    
    parsed_reqs = parser.parse_requirements_list(requirements)
    actual_analysis = analyzer.analyze_python_code(code) if language == 'python' else {}
    
    return {
        'declared_requirements': requirements,
        'actual_capabilities': actual_analysis.get('capabilities', []),
        'requirement_validity': all(req['valid'] for req in parsed_reqs)
    }