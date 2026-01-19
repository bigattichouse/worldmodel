"""
ByteLogic Training Dataset
=========================

Generates and manages training data for ByteLogic computation tokens.
Converts ByteLogic examples into model training format with validation.
"""

import json
import os
import random
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import logging

from ..tokenization.bytelogic_tokenizer import ByteLogicTokenizer
from ..execution.bytelogic_executor import ByteLogicExecutor

logger = logging.getLogger(__name__)


class ByteLogicDatasetGenerator:
    """Generates training datasets from ByteLogic examples."""
    
    def __init__(self, 
                 tokenizer: Optional[ByteLogicTokenizer] = None,
                 executor: Optional[ByteLogicExecutor] = None,
                 validate_examples: bool = True):
        """
        Initialize dataset generator.
        
        Args:
            tokenizer: ByteLogic tokenizer for validation
            executor: ByteLogic executor for result validation
            validate_examples: Whether to validate examples during generation
        """
        self.tokenizer = tokenizer or ByteLogicTokenizer()
        self.executor = executor or ByteLogicExecutor()
        self.validate_examples = validate_examples
        
        # Load base examples from our specification files
        self.base_examples = self._load_base_examples()
        
        # Statistics
        self.generation_stats = {
            "total_examples_generated": 0,
            "valid_examples": 0,
            "invalid_examples": 0,
            "compilation_failures": 0,
            "execution_failures": 0
        }
        
        logger.info(f"ByteLogicDatasetGenerator initialized")
        logger.info(f"  Base examples loaded: {len(self.base_examples)}")
        logger.info(f"  Validation enabled: {validate_examples}")
    
    def _load_base_examples(self) -> List[Dict]:
        """Load base examples from our specification."""
        examples = []
        
        # Family relationship examples
        examples.extend([
            {
                "id": "family_001",
                "category": "logic_programming",
                "subcategory": "family_relationships", 
                "difficulty": "beginner",
                "input": "Who are Alice's children?",
                "bytelogic_code": "REL parent\nFACT parent alice bob\nFACT parent alice charlie\nSOLVE\nQUERY parent alice ?",
                "expected_output": "I'll check Alice's children: <computation>\nREL parent\nFACT parent alice bob\nFACT parent alice charlie\nSOLVE\nQUERY parent alice ?\n</computation> → Bob and Charlie are Alice's children.",
                "expected_result": ["bob", "charlie"]
            },
            {
                "id": "family_002",
                "category": "logic_programming",
                "subcategory": "family_relationships",
                "difficulty": "intermediate", 
                "input": "Who are Alice's grandchildren?",
                "bytelogic_code": "REL parent\nREL grandparent\nFACT parent alice bob\nFACT parent alice charlie\nFACT parent bob david\nFACT parent charlie eve\nRULE grandparent: SCAN parent MATCH $0, JOIN parent $0, EMIT grandparent $1 $2\nSOLVE\nQUERY grandparent alice ?",
                "expected_output": "I'll find Alice's grandchildren using family rules: <computation>\nREL parent\nREL grandparent\nFACT parent alice bob\nFACT parent alice charlie\nFACT parent bob david\nFACT parent charlie eve\nRULE grandparent: SCAN parent MATCH $0, JOIN parent $0, EMIT grandparent $1 $2\nSOLVE\nQUERY grandparent alice ?\n</computation> → Alice's grandchildren are David and Eve.",
                "expected_result": ["david", "eve"]
            }
        ])
        
        # Mathematical examples
        examples.extend([
            {
                "id": "math_001",
                "category": "mathematical_computation",
                "subcategory": "percentage_calculations",
                "difficulty": "beginner",
                "input": "What's 15% of 240?",
                "bytelogic_code": "CALC percentage\n  INPUT $value $percent\n  LET $decimal = $percent / 100\n  RESULT $value * $decimal\nEND\nRESULT CALC percentage(240, 15)",
                "expected_output": "I'll calculate the percentage: <computation>\nCALC percentage\n  INPUT $value $percent\n  LET $decimal = $percent / 100\n  RESULT $value * $decimal\nEND\nRESULT CALC percentage(240, 15)\n</computation> → 36",
                "expected_result": [36]
            },
            {
                "id": "math_002", 
                "category": "mathematical_computation",
                "subcategory": "sequences",
                "difficulty": "intermediate",
                "input": "What's the 10th Fibonacci number?",
                "bytelogic_code": "CALC fib_mental\n  INPUT $position\n  LET $a = 0\n  LET $b = 1\n  LET $count = 0\n  FOR WHILE $count < $position\n    LET $next = $a + $b\n    LET $a = $b\n    LET $b = $next\n    LET $count = $count + 1\n  END\n  RESULT $a\nEND\nRESULT CALC fib_mental(10)",
                "expected_output": "I'll calculate the 10th Fibonacci number: <computation>\nCALC fib_mental\n  INPUT $position\n  LET $a = 0\n  LET $b = 1\n  LET $count = 0\n  FOR WHILE $count < $position\n    LET $next = $a + $b\n    LET $a = $b\n    LET $b = $next\n    LET $count = $count + 1\n  END\n  RESULT $a\nEND\nRESULT CALC fib_mental(10)\n</computation> → 55",
                "expected_result": [55]
            }
        ])
        
        # String processing examples
        examples.extend([
            {
                "id": "string_001",
                "category": "string_processing",
                "subcategory": "character_counting",
                "difficulty": "intermediate",
                "input": "How many R's are in the word 'strawberry'?",
                "bytelogic_code": "CALC count_letter_r\n  INPUT $word_string\n  LET $count = 0\n  FOR $i IN RANGE(0, LENGTH($word_string))\n    LET $char = CHAR_AT($word_string, $i)\n    IF $char == \"r\" THEN\n      LET $count = $count + 1\n    END\n  END\n  RESULT $count\nEND\nRESULT CALC count_letter_r(\"strawberry\")",
                "expected_output": "I'll count the R's: <computation>\nCALC count_letter_r\n  INPUT $word_string\n  LET $count = 0\n  FOR $i IN RANGE(0, LENGTH($word_string))\n    LET $char = CHAR_AT($word_string, $i)\n    IF $char == \"r\" THEN\n      LET $count = $count + 1\n    END\n  END\n  RESULT $count\nEND\nRESULT CALC count_letter_r(\"strawberry\")\n</computation> → 3",
                "expected_result": [3]
            }
        ])
        
        # Graph algorithms
        examples.extend([
            {
                "id": "graph_001",
                "category": "logic_programming",
                "subcategory": "graph_algorithms",
                "difficulty": "intermediate",
                "input": "Can you reach node 4 from node 0 in this graph?",
                "bytelogic_code": "REL edge\nREL reachable\nFACT edge 0 1\nFACT edge 0 2\nFACT edge 1 3\nFACT edge 2 3\nFACT edge 3 4\nRULE reachable: SCAN edge, EMIT reachable $0 $1\nRULE reachable: SCAN edge, JOIN reachable $1, EMIT reachable $0 $2\nSOLVE\nQUERY reachable 0 4",
                "expected_output": "I'll check graph reachability: <computation>\nREL edge\nREL reachable\nFACT edge 0 1\nFACT edge 0 2\nFACT edge 1 3\nFACT edge 2 3\nFACT edge 3 4\nRULE reachable: SCAN edge, EMIT reachable $0 $1\nRULE reachable: SCAN edge, JOIN reachable $1, EMIT reachable $0 $2\nSOLVE\nQUERY reachable 0 4\n</computation> → Yes, you can reach node 4 from node 0.",
                "expected_result": [1]
            }
        ])
        
        return examples
    
    def generate_variations(self, base_example: Dict, num_variations: int = 5) -> List[Dict]:
        """Generate variations of a base example."""
        variations = []
        
        for i in range(num_variations):
            variation = self._create_variation(base_example, i)
            if variation:
                variations.append(variation)
        
        return variations
    
    def _create_variation(self, base_example: Dict, variation_id: int) -> Optional[Dict]:
        """Create a single variation of a base example."""
        try:
            category = base_example["category"]
            
            if category == "logic_programming":
                return self._vary_logic_example(base_example, variation_id)
            elif category == "mathematical_computation":
                return self._vary_math_example(base_example, variation_id)
            elif category == "string_processing":
                return self._vary_string_example(base_example, variation_id)
            else:
                return None
                
        except Exception as e:
            logger.warning(f"Failed to create variation {variation_id}: {e}")
            return None
    
    def _vary_logic_example(self, base: Dict, var_id: int) -> Dict:
        """Create variations for logic programming examples."""
        # Name variations for family relationships
        name_sets = [
            ["alice", "bob", "charlie", "david", "eve"],
            ["john", "mary", "peter", "susan", "tom"],
            ["anna", "ben", "clara", "dan", "ella"],
            ["alex", "beth", "chris", "diana", "eric"],
            ["amy", "brad", "cathy", "dean", "faye"]
        ]
        
        names = name_sets[var_id % len(name_sets)]
        
        # Replace names in the example
        variation = base.copy()
        variation["id"] = f"{base['id']}_var_{var_id}"
        
        # Replace in input text
        old_names = ["alice", "bob", "charlie", "david", "eve"]
        new_input = variation["input"]
        new_code = variation["bytelogic_code"]
        new_output = variation["expected_output"]
        
        for old_name, new_name in zip(old_names, names):
            new_input = new_input.replace(old_name.capitalize(), new_name.capitalize())
            new_code = new_code.replace(old_name, new_name)
            new_output = new_output.replace(old_name.capitalize(), new_name.capitalize())
            new_output = new_output.replace(old_name, new_name)
        
        variation["input"] = new_input
        variation["bytelogic_code"] = new_code
        variation["expected_output"] = new_output
        
        # Update expected results
        if "expected_result" in variation:
            new_results = []
            for result in variation["expected_result"]:
                if isinstance(result, str) and result in old_names:
                    idx = old_names.index(result)
                    if idx < len(names):
                        new_results.append(names[idx])
                    else:
                        new_results.append(result)
                else:
                    new_results.append(result)
            variation["expected_result"] = new_results
        
        return variation
    
    def _vary_math_example(self, base: Dict, var_id: int) -> Dict:
        """Create variations for mathematical examples."""
        variation = base.copy()
        variation["id"] = f"{base['id']}_var_{var_id}"
        
        # Number variations for percentage calculation
        if "percentage" in base["id"]:
            values = [(200, 20), (300, 25), (150, 30), (400, 15), (500, 12)]
            value, percent = values[var_id % len(values)]
            expected = value * percent / 100
            
            variation["input"] = f"What's {percent}% of {value}?"
            variation["bytelogic_code"] = base["bytelogic_code"].replace("240", str(value)).replace("15", str(percent))
            variation["expected_output"] = variation["expected_output"].replace("240", str(value)).replace("15", str(percent)).replace("36", str(int(expected)))
            variation["expected_result"] = [int(expected)]
        
        # Fibonacci variations
        elif "fib" in base["id"]:
            positions = [5, 7, 8, 9, 12]
            results = [5, 13, 21, 34, 144]  # Corresponding Fibonacci numbers
            
            pos = positions[var_id % len(positions)]
            result = results[var_id % len(results)]
            
            variation["input"] = f"What's the {pos}th Fibonacci number?"
            variation["bytelogic_code"] = base["bytelogic_code"].replace("10", str(pos))
            variation["expected_output"] = variation["expected_output"].replace("10th", f"{pos}th").replace("10", str(pos)).replace("55", str(result))
            variation["expected_result"] = [result]
        
        return variation
    
    def _vary_string_example(self, base: Dict, var_id: int) -> Dict:
        """Create variations for string processing examples."""
        variation = base.copy()
        variation["id"] = f"{base['id']}_var_{var_id}"
        
        # Letter counting variations
        if "count_letter" in base["id"]:
            words_letters = [
                ("programming", "r", 2),
                ("hello", "l", 2),
                ("computer", "e", 1),
                ("algorithm", "o", 1),
                ("intelligence", "i", 2)
            ]
            
            word, letter, count = words_letters[var_id % len(words_letters)]
            
            variation["input"] = f"How many {letter.upper()}'s are in the word '{word}'?"
            variation["bytelogic_code"] = base["bytelogic_code"].replace("strawberry", word).replace('"r"', f'"{letter}"')
            variation["expected_output"] = variation["expected_output"].replace("strawberry", word).replace("R's", f"{letter.upper()}'s").replace("3", str(count))
            variation["expected_result"] = [count]
        
        return variation
    
    def validate_example(self, example: Dict) -> Dict:
        """Validate an example by checking syntax and optionally executing."""
        validation = {
            "valid": True,
            "syntax_valid": True,
            "compiles": True,
            "executes": True,
            "result_matches": True,
            "errors": []
        }
        
        try:
            # Check syntax
            code = example.get("bytelogic_code", "")
            is_valid, error = self.tokenizer.validate_bytelogic_syntax(code)
            
            if not is_valid:
                validation["syntax_valid"] = False
                validation["valid"] = False
                validation["errors"].append(f"Syntax error: {error}")
                return validation
            
            # Test compilation if executor available
            if self.validate_examples and self.executor:
                try:
                    result = self.executor.execute_bytelogic(code)
                    
                    if not result["success"]:
                        validation["compiles"] = False
                        validation["valid"] = False
                        validation["errors"].append(f"Compilation failed: {result.get('error', 'unknown')}")
                        return validation
                    
                    # Check if result matches expected
                    if "expected_result" in example:
                        expected = example["expected_result"]
                        actual = result.get("result")
                        
                        if isinstance(expected, list) and isinstance(actual, list):
                            # Compare lists (order may matter for some queries)
                            if sorted(expected) != sorted(actual):
                                validation["result_matches"] = False
                                validation["errors"].append(f"Result mismatch: expected {expected}, got {actual}")
                        elif expected != actual:
                            validation["result_matches"] = False
                            validation["errors"].append(f"Result mismatch: expected {expected}, got {actual}")
                
                except Exception as e:
                    validation["executes"] = False
                    validation["valid"] = False
                    validation["errors"].append(f"Execution failed: {str(e)}")
            
        except Exception as e:
            validation["valid"] = False
            validation["errors"].append(f"Validation error: {str(e)}")
        
        return validation
    
    def generate_training_dataset(self, 
                                 output_file: str,
                                 num_variations_per_example: int = 5,
                                 train_split: float = 0.8,
                                 val_split: float = 0.1) -> Dict:
        """
        Generate complete training dataset.
        
        Args:
            output_file: Path to save the dataset
            num_variations_per_example: Number of variations per base example
            train_split: Fraction for training set
            val_split: Fraction for validation set (rest goes to test)
            
        Returns:
            Dataset statistics
        """
        logger.info("Generating ByteLogic training dataset...")
        
        all_examples = []
        
        # Generate variations for each base example
        for base_example in self.base_examples:
            # Add original example
            if self.validate_examples:
                validation = self.validate_example(base_example)
                if validation["valid"]:
                    all_examples.append(base_example)
                    self.generation_stats["valid_examples"] += 1
                else:
                    logger.warning(f"Base example {base_example['id']} failed validation: {validation['errors']}")
                    self.generation_stats["invalid_examples"] += 1
            else:
                all_examples.append(base_example)
                self.generation_stats["valid_examples"] += 1
            
            # Generate variations
            variations = self.generate_variations(base_example, num_variations_per_example)
            for variation in variations:
                if self.validate_examples:
                    validation = self.validate_example(variation)
                    if validation["valid"]:
                        all_examples.append(variation)
                        self.generation_stats["valid_examples"] += 1
                    else:
                        self.generation_stats["invalid_examples"] += 1
                else:
                    all_examples.append(variation)
                    self.generation_stats["valid_examples"] += 1
        
        # Split dataset
        random.shuffle(all_examples)
        total = len(all_examples)
        
        train_end = int(total * train_split)
        val_end = train_end + int(total * val_split)
        
        train_examples = all_examples[:train_end]
        val_examples = all_examples[train_end:val_end]
        test_examples = all_examples[val_end:]
        
        # Create dataset format
        dataset = {
            "metadata": {
                "version": "1.0",
                "generator": "ByteLogicDatasetGenerator",
                "total_examples": total,
                "train_examples": len(train_examples),
                "val_examples": len(val_examples),
                "test_examples": len(test_examples),
                "categories": list(set(ex["category"] for ex in all_examples)),
                "difficulty_levels": list(set(ex["difficulty"] for ex in all_examples)),
                "validation_enabled": self.validate_examples,
                "generation_stats": self.generation_stats
            },
            "train": self._format_examples_for_training(train_examples),
            "validation": self._format_examples_for_training(val_examples),
            "test": self._format_examples_for_training(test_examples)
        }
        
        # Save dataset
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(dataset, f, indent=2)
        
        logger.info(f"Dataset saved to {output_file}")
        logger.info(f"  Total examples: {total}")
        logger.info(f"  Training: {len(train_examples)}")
        logger.info(f"  Validation: {len(val_examples)}")
        logger.info(f"  Test: {len(test_examples)}")
        
        return dataset["metadata"]
    
    def _format_examples_for_training(self, examples: List[Dict]) -> List[Dict]:
        """Format examples for model training."""
        formatted = []
        
        for example in examples:
            # Standard training format
            training_example = {
                "id": example["id"],
                "input": example["input"],
                "output": example["expected_output"],
                "metadata": {
                    "category": example["category"],
                    "subcategory": example["subcategory"],
                    "difficulty": example["difficulty"],
                    "bytelogic_code": example["bytelogic_code"],
                    "expected_result": example.get("expected_result")
                }
            }
            
            formatted.append(training_example)
        
        return formatted
    
    def export_examples_to_jsonl(self, examples: List[Dict], output_file: str):
        """Export examples to JSONL format for easy streaming."""
        with open(output_file, 'w') as f:
            for example in examples:
                f.write(json.dumps(example) + '\n')
    
    def get_statistics(self) -> Dict:
        """Get generation statistics."""
        return {
            "base_examples": len(self.base_examples),
            "generation_stats": self.generation_stats,
            "categories": list(set(ex["category"] for ex in self.base_examples)),
            "subcategories": list(set(ex["subcategory"] for ex in self.base_examples))
        }