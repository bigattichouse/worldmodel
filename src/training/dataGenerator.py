import asyncio
import json
import random
import re
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union
import logging

from ..utils.config import TrainingConfig
from ..utils.logging import get_logger
from ..core.tagParser import TagParser, ModelTag, RequiresTag
from ..execution.vmInterface import VMInterface, ExecutionResult


@dataclass
class TrainingExample:
    input_text: str
    target_output: str
    metadata: Dict[str, Any] = None
    difficulty: str = "medium"  # easy, medium, hard
    category: str = "general"
    created_at: Optional[datetime] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now(timezone.utc)
        if self.metadata is None:
            self.metadata = {}
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        if self.created_at:
            data['created_at'] = self.created_at.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TrainingExample':
        data = data.copy()
        if 'created_at' in data and data['created_at']:
            data['created_at'] = datetime.fromisoformat(data['created_at'])
        return cls(**data)

@dataclass
class DatasetStats:
    total_examples: int
    examples_by_category: Dict[str, int]
    examples_by_difficulty: Dict[str, int]
    examples_with_execution: int
    examples_with_thinking: int
    average_input_length: float
    average_output_length: float

class ProblemTemplate:
    def __init__(self, name: str, description: str, template: str, 
                 variables: Dict[str, Any], category: str = "general",
                 difficulty: str = "medium", requires_execution: bool = False):
        self.name = name
        self.description = description
        self.template = template
        self.variables = variables
        self.category = category
        self.difficulty = difficulty
        self.requires_execution = requires_execution
    
    def generate_problem(self) -> Tuple[str, Dict[str, Any]]:
        """Generate a specific problem instance from this template."""
        # Randomly select values for variables
        instance_vars = {}
        for var_name, var_config in self.variables.items():
            if isinstance(var_config, dict):
                if var_config.get('type') == 'int':
                    min_val = var_config.get('min', 1)
                    max_val = var_config.get('max', 100)
                    instance_vars[var_name] = random.randint(min_val, max_val)
                elif var_config.get('type') == 'float':
                    min_val = var_config.get('min', 1.0)
                    max_val = var_config.get('max', 100.0)
                    instance_vars[var_name] = round(random.uniform(min_val, max_val), 2)
                elif var_config.get('type') == 'choice':
                    instance_vars[var_name] = random.choice(var_config.get('choices', []))
                elif var_config.get('type') == 'string':
                    instance_vars[var_name] = random.choice(var_config.get('options', ['item']))
                else:
                    instance_vars[var_name] = var_config
            else:
                instance_vars[var_name] = var_config
        
        # Format template with variables
        problem_text = self.template.format(**instance_vars)
        
        return problem_text, instance_vars

class DataGenerator:
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.logger = get_logger(__name__)
        self.tag_parser = TagParser()
        self.vm_interface = None  # Will be set if needed
        
        # Load problem templates
        self.templates = self._initialize_templates()
        
        self.logger.info("Data generator initialized")
    
    def _initialize_templates(self) -> List[ProblemTemplate]:
        """Initialize problem templates for generating synthetic data."""
        templates = []
        
        # Math problems
        templates.extend([
            ProblemTemplate(
                name="basic_arithmetic",
                description="Simple arithmetic operations",
                template="Calculate {a} {op} {b}",
                variables={
                    'a': {'type': 'int', 'min': 1, 'max': 100},
                    'b': {'type': 'int', 'min': 1, 'max': 100},
                    'op': {'type': 'choice', 'choices': ['+', '-', '*']}
                },
                category="math",
                difficulty="easy",
                requires_execution=True
            ),
            
            ProblemTemplate(
                name="quadratic_roots",
                description="Find roots of quadratic equations",
                template="Find the roots of the quadratic equation {a}x² + {b}x + {c} = 0",
                variables={
                    'a': {'type': 'int', 'min': 1, 'max': 5},
                    'b': {'type': 'int', 'min': -10, 'max': 10},
                    'c': {'type': 'int', 'min': -10, 'max': 10}
                },
                category="math",
                difficulty="medium",
                requires_execution=True
            ),
            
            ProblemTemplate(
                name="statistics_mean",
                description="Calculate mean of numbers",
                template="Calculate the mean of the following numbers: {numbers}",
                variables={
                    'numbers': {'type': 'string', 'generator': 'number_list'}
                },
                category="math",
                difficulty="easy",
                requires_execution=True
            )
        ])
        
        # Programming problems
        templates.extend([
            ProblemTemplate(
                name="fibonacci_sequence",
                description="Generate Fibonacci sequence",
                template="Generate the first {n} numbers in the Fibonacci sequence",
                variables={
                    'n': {'type': 'int', 'min': 5, 'max': 15}
                },
                category="programming",
                difficulty="medium",
                requires_execution=True
            ),
            
            ProblemTemplate(
                name="prime_checker",
                description="Check if number is prime",
                template="Write a function to check if {number} is a prime number",
                variables={
                    'number': {'type': 'int', 'min': 2, 'max': 100}
                },
                category="programming",
                difficulty="medium",
                requires_execution=True
            ),
            
            ProblemTemplate(
                name="list_manipulation",
                description="Basic list operations",
                template="Create a list of {count} {item_type} and {operation}",
                variables={
                    'count': {'type': 'int', 'min': 5, 'max': 10},
                    'item_type': {'type': 'choice', 'choices': ['numbers', 'strings', 'items']},
                    'operation': {'type': 'choice', 'choices': ['sort them', 'find the maximum', 'count unique items', 'reverse the order']}
                },
                category="programming",
                difficulty="easy",
                requires_execution=True
            )
        ])
        
        # Text processing
        templates.extend([
            ProblemTemplate(
                name="text_analysis",
                description="Analyze text properties",
                template="Analyze the following text: '{text}'. Count the number of {analysis_type}",
                variables={
                    'text': {'type': 'choice', 'choices': [
                        'Hello world, this is a test sentence with several words.',
                        'The quick brown fox jumps over the lazy dog.',
                        'Python is a powerful programming language for data science.',
                        'Machine learning and artificial intelligence are transforming technology.'
                    ]},
                    'analysis_type': {'type': 'choice', 'choices': ['words', 'characters', 'vowels', 'consonants']}
                },
                category="text",
                difficulty="easy",
                requires_execution=True
            ),
            
            ProblemTemplate(
                name="string_operations",
                description="String manipulation tasks",
                template="Take the string '{input_str}' and {operation}",
                variables={
                    'input_str': {'type': 'choice', 'choices': [
                        'hello world',
                        'Python Programming',
                        'Data Science',
                        'machine learning'
                    ]},
                    'operation': {'type': 'choice', 'choices': [
                        'convert to uppercase',
                        'reverse the string',
                        'remove all vowels',
                        'replace spaces with underscores'
                    ]}
                },
                category="text",
                difficulty="easy",
                requires_execution=True
            )
        ])
        
        # Logic problems
        templates.extend([
            ProblemTemplate(
                name="boolean_logic",
                description="Boolean logic evaluation",
                template="Evaluate the boolean expression: {expr1} {op1} {expr2} {op2} {expr3}",
                variables={
                    'expr1': {'type': 'choice', 'choices': ['True', 'False']},
                    'expr2': {'type': 'choice', 'choices': ['True', 'False']},
                    'expr3': {'type': 'choice', 'choices': ['True', 'False']},
                    'op1': {'type': 'choice', 'choices': ['and', 'or']},
                    'op2': {'type': 'choice', 'choices': ['and', 'or']}
                },
                category="logic",
                difficulty="medium",
                requires_execution=True
            )
        ])
        
        return templates
    
    def _generate_number_list(self, count: int = None) -> str:
        """Generate a comma-separated list of random numbers."""
        if count is None:
            count = random.randint(5, 10)
        numbers = [random.randint(1, 100) for _ in range(count)]
        return ', '.join(map(str, numbers))
    
    def _solve_arithmetic(self, a: int, op: str, b: int) -> Tuple[str, str]:
        """Generate solution for arithmetic problem."""
        thinking = f"<think>I need to calculate {a} {op} {b}. Let me compute this step by step.</think>"
        
        if op == '+':
            result = a + b
            code = f"result = {a} + {b}"
        elif op == '-':
            result = a - b
            code = f"result = {a} - {b}"
        elif op == '*':
            result = a * b
            code = f"result = {a} * {b}"
        else:
            result = "error"
            code = f"# Error: unsupported operation {op}"
        
        model_code = f"<model>\n{code}\nprint(result)\n</model>"
        requires = "<requires>python:math</requires>"
        
        output = f"{thinking}\n{model_code}\n{requires}"
        return output, str(result)
    
    def _solve_quadratic(self, a: int, b: int, c: int) -> Tuple[str, str]:
        """Generate solution for quadratic equation."""
        thinking = f"<think>I need to find roots of {a}x² + {b}x + {c} = 0 using the quadratic formula.</think>"
        
        code = f"""import math

a, b, c = {a}, {b}, {c}
discriminant = b**2 - 4*a*c

if discriminant > 0:
    root1 = (-b + math.sqrt(discriminant)) / (2*a)
    root2 = (-b - math.sqrt(discriminant)) / (2*a)
    print(f"Two real roots: {{root1:.2f}}, {{root2:.2f}}")
elif discriminant == 0:
    root = -b / (2*a)
    print(f"One real root: {{root:.2f}}")
else:
    real_part = -b / (2*a)
    imag_part = math.sqrt(-discriminant) / (2*a)
    print(f"Complex roots: {{real_part:.2f}} ± {{imag_part:.2f}}i")"""
        
        model_code = f"<model>\n{code}\n</model>"
        requires = "<requires>python:math</requires>"
        
        output = f"{thinking}\n{model_code}\n{requires}"
        
        # Calculate expected result
        discriminant = b**2 - 4*a*c
        if discriminant > 0:
            import math
            root1 = (-b + math.sqrt(discriminant)) / (2*a)
            root2 = (-b - math.sqrt(discriminant)) / (2*a)
            expected = f"Two real roots: {root1:.2f}, {root2:.2f}"
        elif discriminant == 0:
            root = -b / (2*a)
            expected = f"One real root: {root:.2f}"
        else:
            import math
            real_part = -b / (2*a)
            imag_part = math.sqrt(-discriminant) / (2*a)
            expected = f"Complex roots: {real_part:.2f} ± {imag_part:.2f}i"
        
        return output, expected
    
    def _solve_fibonacci(self, n: int) -> Tuple[str, str]:
        """Generate solution for Fibonacci sequence."""
        thinking = f"<think>I need to generate the first {n} Fibonacci numbers. Each number is the sum of the two preceding ones.</think>"
        
        code = f"""def fibonacci(n):
    if n <= 0:
        return []
    elif n == 1:
        return [0]
    elif n == 2:
        return [0, 1]
    
    fib = [0, 1]
    for i in range(2, n):
        fib.append(fib[i-1] + fib[i-2])
    return fib

result = fibonacci({n})
print(result)"""
        
        model_code = f"<model>\n{code}\n</model>"
        requires = "<requires>python:computation</requires>"
        
        output = f"{thinking}\n{model_code}\n{requires}"
        
        # Calculate expected result
        if n <= 0:
            expected = "[]"
        elif n == 1:
            expected = "[0]"
        elif n == 2:
            expected = "[0, 1]"
        else:
            fib = [0, 1]
            for i in range(2, n):
                fib.append(fib[i-1] + fib[i-2])
            expected = str(fib)
        
        return output, expected
    
    def _solve_prime_check(self, number: int) -> Tuple[str, str]:
        """Generate solution for prime checking."""
        thinking = f"<think>I need to check if {number} is prime. A prime number is only divisible by 1 and itself.</think>"
        
        code = f"""def is_prime(n):
    if n < 2:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True

number = {number}
result = is_prime(number)
print(f"{{number}} is {{'prime' if result else 'not prime'}}")"""
        
        model_code = f"<model>\n{code}\n</model>"
        requires = "<requires>python:math</requires>"
        
        output = f"{thinking}\n{model_code}\n{requires}"
        
        # Calculate expected result
        def is_prime(n):
            if n < 2:
                return False
            for i in range(2, int(n**0.5) + 1):
                if n % i == 0:
                    return False
            return True
        
        result = is_prime(number)
        expected = f"{number} is {'prime' if result else 'not prime'}"
        
        return output, expected
    
    def _solve_text_analysis(self, text: str, analysis_type: str) -> Tuple[str, str]:
        """Generate solution for text analysis."""
        thinking = f"<think>I need to analyze the text '{text}' and count {analysis_type}.</think>"
        
        if analysis_type == "words":
            code = f"""text = "{text}"
words = text.split()
count = len(words)
print(f"Number of words: {{count}}")"""
            expected_count = len(text.split())
            expected = f"Number of words: {expected_count}"
        
        elif analysis_type == "characters":
            code = f"""text = "{text}"
count = len(text)
print(f"Number of characters: {{count}}")"""
            expected_count = len(text)
            expected = f"Number of characters: {expected_count}"
        
        elif analysis_type == "vowels":
            code = f"""text = "{text}"
vowels = "aeiouAEIOU"
count = sum(1 for char in text if char in vowels)
print(f"Number of vowels: {{count}}")"""
            vowels = "aeiouAEIOU"
            expected_count = sum(1 for char in text if char in vowels)
            expected = f"Number of vowels: {expected_count}"
        
        elif analysis_type == "consonants":
            code = f"""text = "{text}"
consonants = sum(1 for char in text if char.isalpha() and char not in "aeiouAEIOU")
print(f"Number of consonants: {{consonants}}")"""
            expected_count = sum(1 for char in text if char.isalpha() and char not in "aeiouAEIOU")
            expected = f"Number of consonants: {expected_count}"
        
        else:
            code = f"# Unknown analysis type: {analysis_type}"
            expected = "Error: unknown analysis type"
        
        model_code = f"<model>\n{code}\n</model>"
        requires = "<requires>python:text</requires>"
        
        output = f"{thinking}\n{model_code}\n{requires}"
        return output, expected
    
    def _generate_solution(self, template: ProblemTemplate, problem_text: str, 
                          variables: Dict[str, Any]) -> Tuple[str, str]:
        """Generate the expected solution for a problem."""
        
        if template.name == "basic_arithmetic":
            return self._solve_arithmetic(variables['a'], variables['op'], variables['b'])
        
        elif template.name == "quadratic_roots":
            return self._solve_quadratic(variables['a'], variables['b'], variables['c'])
        
        elif template.name == "fibonacci_sequence":
            return self._solve_fibonacci(variables['n'])
        
        elif template.name == "prime_checker":
            return self._solve_prime_check(variables['number'])
        
        elif template.name == "text_analysis":
            return self._solve_text_analysis(variables['text'], variables['analysis_type'])
        
        elif template.name == "statistics_mean":
            # Handle special case for number list
            if variables['numbers'] == 'number_list':
                numbers_str = self._generate_number_list()
                variables['numbers'] = numbers_str
                problem_text = template.template.format(**variables)
            
            thinking = f"<think>I need to calculate the mean of these numbers: {variables['numbers']}</think>"
            code = f"""numbers = [{variables['numbers']}]
mean = sum(numbers) / len(numbers)
print(f"Mean: {{mean:.2f}}")"""
            
            model_code = f"<model>\n{code}\n</model>"
            requires = "<requires>python:math</requires>"
            
            output = f"{thinking}\n{model_code}\n{requires}"
            
            # Calculate expected result
            try:
                numbers = [float(x.strip()) for x in variables['numbers'].split(',')]
                expected_mean = sum(numbers) / len(numbers)
                expected = f"Mean: {expected_mean:.2f}"
            except:
                expected = "Error calculating mean"
            
            return output, expected
        
        else:
            # Generic solution template
            thinking = f"<think>I need to solve: {problem_text}</think>"
            model_code = f"<model>\n# Solution for: {problem_text}\nprint('Solution implementation needed')\n</model>"
            requires = "<requires>python:general</requires>"
            
            output = f"{thinking}\n{model_code}\n{requires}"
            return output, "Solution implementation needed"
    
    async def generate_example(self, template_name: str = None) -> TrainingExample:
        """Generate a single training example."""
        
        # Select template
        if template_name:
            templates = [t for t in self.templates if t.name == template_name]
            if not templates:
                raise ValueError(f"Template '{template_name}' not found")
            template = templates[0]
        else:
            template = random.choice(self.templates)
        
        # Generate problem instance
        problem_text, variables = template.generate_problem()
        
        # Generate solution
        target_output, expected_result = self._generate_solution(template, problem_text, variables)
        
        # Create metadata
        metadata = {
            'template_name': template.name,
            'template_description': template.description,
            'variables': variables,
            'requires_execution': template.requires_execution,
            'expected_result': expected_result
        }
        
        example = TrainingExample(
            input_text=problem_text,
            target_output=target_output,
            metadata=metadata,
            difficulty=template.difficulty,
            category=template.category
        )
        
        self.logger.debug(f"Generated example: {template.name} ({template.category})")
        return example
    
    async def generate_dataset(self, size: int, 
                              category_weights: Dict[str, float] = None,
                              difficulty_weights: Dict[str, float] = None) -> List[TrainingExample]:
        """Generate a dataset of training examples."""
        
        if category_weights is None:
            category_weights = {
                'math': 0.4,
                'programming': 0.3,
                'text': 0.2,
                'logic': 0.1
            }
        
        if difficulty_weights is None:
            difficulty_weights = {
                'easy': 0.4,
                'medium': 0.5,
                'hard': 0.1
            }
        
        examples = []
        
        for i in range(size):
            # Select category based on weights
            categories = list(category_weights.keys())
            weights = list(category_weights.values())
            selected_category = random.choices(categories, weights=weights)[0]
            
            # Select difficulty based on weights
            difficulties = list(difficulty_weights.keys())
            diff_weights = list(difficulty_weights.values())
            selected_difficulty = random.choices(difficulties, weights=diff_weights)[0]
            
            # Filter templates by category and difficulty
            suitable_templates = [
                t for t in self.templates 
                if t.category == selected_category and t.difficulty == selected_difficulty
            ]
            
            if not suitable_templates:
                # Fallback to any template from the category
                suitable_templates = [t for t in self.templates if t.category == selected_category]
                if not suitable_templates:
                    # Final fallback to any template
                    suitable_templates = self.templates
            
            template = random.choice(suitable_templates)
            
            try:
                example = await self.generate_example(template.name)
                examples.append(example)
                
                if (i + 1) % 100 == 0:
                    self.logger.info(f"Generated {i + 1}/{size} examples")
                    
            except Exception as e:
                self.logger.error(f"Error generating example from template {template.name}: {e}")
                continue
        
        self.logger.info(f"Generated {len(examples)} training examples")
        return examples
    
    def save_dataset(self, examples: List[TrainingExample], file_path: str) -> bool:
        """Save dataset to JSON file."""
        try:
            data = {
                'metadata': {
                    'total_examples': len(examples),
                    'created_at': datetime.now(timezone.utc).isoformat(),
                    'generator_version': '1.0'
                },
                'examples': [example.to_dict() for example in examples]
            }
            
            output_path = Path(file_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Saved {len(examples)} examples to {file_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving dataset: {e}")
            return False
    
    def load_dataset(self, file_path: str) -> List[TrainingExample]:
        """Load dataset from JSON file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            examples = []
            for example_data in data.get('examples', []):
                try:
                    example = TrainingExample.from_dict(example_data)
                    examples.append(example)
                except Exception as e:
                    self.logger.error(f"Error loading example: {e}")
                    continue
            
            self.logger.info(f"Loaded {len(examples)} examples from {file_path}")
            return examples
            
        except Exception as e:
            self.logger.error(f"Error loading dataset: {e}")
            return []
    
    def get_dataset_stats(self, examples: List[TrainingExample]) -> DatasetStats:
        """Calculate statistics for a dataset."""
        if not examples:
            return DatasetStats(0, {}, {}, 0, 0, 0.0, 0.0)
        
        # Count by category
        category_counts = {}
        for example in examples:
            category_counts[example.category] = category_counts.get(example.category, 0) + 1
        
        # Count by difficulty
        difficulty_counts = {}
        for example in examples:
            difficulty_counts[example.difficulty] = difficulty_counts.get(example.difficulty, 0) + 1
        
        # Count examples with execution and thinking
        execution_count = sum(1 for ex in examples if ex.metadata.get('requires_execution', False))
        thinking_count = sum(1 for ex in examples if '<think>' in ex.target_output)
        
        # Calculate average lengths
        input_lengths = [len(ex.input_text) for ex in examples]
        output_lengths = [len(ex.target_output) for ex in examples]
        
        avg_input_length = sum(input_lengths) / len(input_lengths)
        avg_output_length = sum(output_lengths) / len(output_lengths)
        
        return DatasetStats(
            total_examples=len(examples),
            examples_by_category=category_counts,
            examples_by_difficulty=difficulty_counts,
            examples_with_execution=execution_count,
            examples_with_thinking=thinking_count,
            average_input_length=avg_input_length,
            average_output_length=avg_output_length
        )

# Convenience functions
async def generate_training_data(config: TrainingConfig, size: int = None) -> List[TrainingExample]:
    """Generate training data using the default configuration."""
    if size is None:
        size = config.synthetic_data_size
    
    generator = DataGenerator(config)
    return await generator.generate_dataset(size)

async def quick_generate(size: int = 100, output_file: str = None) -> List[TrainingExample]:
    """Quickly generate training examples with default settings."""
    from ..utils.config import TrainingConfig
    
    config = TrainingConfig()
    generator = DataGenerator(config)
    examples = await generator.generate_dataset(size)
    
    if output_file:
        generator.save_dataset(examples, output_file)
    
    return examples