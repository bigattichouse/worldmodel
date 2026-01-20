#!/usr/bin/env python3
"""
ByteLogic Error Handling Dataset Generator
==========================================

Generates training examples for:
1. Parse errors (invalid syntax)
2. Runtime errors (unbound variables, undefined relations)
3. Graceful error recovery 
4. WASM execution failures

This fills the gap in error handling training data.
"""

import json
import random
from typing import List, Dict, Any

class ErrorHandlingGenerator:
    """Generate error handling training examples."""
    
    def __init__(self):
        self.atoms = [
            "alice", "bob", "charlie", "david", "eve", "frank", "grace",
            "henry", "ivy", "jack", "kate", "luke", "mary", "nancy", 
            "oscar", "peter", "quinn", "ruth", "sam", "tina"
        ]
    
    def generate_parse_error_examples(self, count: int = 50) -> List[Dict[str, Any]]:
        """Generate examples with syntax errors."""
        examples = []
        
        error_patterns = [
            # Missing REL declaration
            {
                "error_type": "undefined_relation",
                "input": "Find parent relationships without declaring the relation",
                "output": "I notice there's an error in this ByteLogic code: <computation>\nFACT parent alice bob\nSOLVE\nQUERY parent ? ?\n</computation> ❌ Error: Relation 'parent' used but not declared. Need to add 'REL parent' first.",
                "expected_error": "undefined_relation"
            },
            
            # Unbound variables
            {
                "error_type": "unbound_variable", 
                "input": "Create a rule with an unbound variable",
                "output": "This ByteLogic code has a variable binding error: <computation>\nREL parent\nFACT parent alice bob\nRULE test: SCAN parent, EMIT result $2 $0\nSOLVE\nQUERY result ? ?\n</computation> ❌ Error: Variable $2 is not bound in the SCAN operation. Only $0 and $1 are available.",
                "expected_error": "unbound_variable"
            },
            
            # Missing SOLVE
            {
                "error_type": "missing_solve",
                "input": "Query without solving first", 
                "output": "This code is missing a required step: <computation>\nREL parent\nFACT parent alice bob\nQUERY parent alice ?\n</computation> ❌ Error: QUERY used without SOLVE. Must call SOLVE before querying results.",
                "expected_error": "missing_solve"
            },
            
            # Invalid syntax
            {
                "error_type": "syntax_error",
                "input": "Code with syntax errors",
                "output": "This ByteLogic code has syntax errors: <computation>\nREL parent\nFACTS parent alice bob\nSOLVE\nQUERY parent ? ?\n</computation> ❌ Error: Unknown keyword 'FACTS'. Did you mean 'FACT'?",
                "expected_error": "syntax_error"
            }
        ]
        
        for i in range(count):
            pattern = random.choice(error_patterns)
            atom1 = random.choice(self.atoms)
            atom2 = random.choice(self.atoms)
            
            example = {
                "input": pattern["input"],
                "output": pattern["output"].replace("alice", atom1).replace("bob", atom2),
                "metadata": {
                    "id": f"error_{pattern['error_type']}_{i}",
                    "category": "error_handling",
                    "subcategory": pattern["error_type"],
                    "difficulty": "intermediate",
                    "expected_error": pattern["expected_error"],
                    "error_type": pattern["error_type"]
                }
            }
            examples.append(example)
        
        return examples
    
    def generate_recovery_examples(self, count: int = 30) -> List[Dict[str, Any]]:
        """Generate examples showing error recovery."""
        examples = []
        
        recovery_patterns = [
            {
                "input": "Fix this broken ByteLogic code: FACT parent alice bob QUERY parent ? ?",
                "output": "I'll fix this step by step: <computation>\nREL parent\nFACT parent alice bob\nSOLVE\nQUERY parent ? ?\n</computation> → Fixed by adding missing REL declaration and SOLVE step. Result: alice→bob",
                "error_fixed": "missing_rel_and_solve"
            },
            
            {
                "input": "Correct this rule with unbound variables: RULE test: SCAN parent, EMIT result $2 $0",
                "output": "I'll fix the variable binding: <computation>\nREL parent\nFACT parent alice bob\nRULE test: SCAN parent, EMIT result $0 $1\nSOLVE\nQUERY result ? ?\n</computation> → Fixed by using bound variables $0 and $1 instead of unbound $2. Result: alice→bob",
                "error_fixed": "unbound_variable"
            }
        ]
        
        for i in range(count):
            pattern = random.choice(recovery_patterns)
            atom1 = random.choice(self.atoms)
            atom2 = random.choice(self.atoms)
            
            example = {
                "input": pattern["input"],
                "output": pattern["output"].replace("alice", atom1).replace("bob", atom2),
                "metadata": {
                    "id": f"recovery_{pattern['error_fixed']}_{i}",
                    "category": "error_handling", 
                    "subcategory": "error_recovery",
                    "difficulty": "advanced",
                    "error_fixed": pattern["error_fixed"]
                }
            }
            examples.append(example)
        
        return examples
    
    def generate_runtime_error_examples(self, count: int = 20) -> List[Dict[str, Any]]:
        """Generate examples with runtime/execution errors."""
        examples = []
        
        for i in range(count):
            atom1 = random.choice(self.atoms)
            atom2 = random.choice(self.atoms)
            
            example = {
                "input": f"What happens if WASM execution fails?",
                "output": f"If ByteLogic compilation or WASM execution fails: <computation>\nREL parent\nFACT parent {atom1} {atom2}\nSOLVE\nQUERY parent ? ?\n</computation> ⚠️ If compilation/execution fails, I'll explain the error and suggest fixes rather than returning invalid results.",
                "metadata": {
                    "id": f"runtime_error_{i}",
                    "category": "error_handling",
                    "subcategory": "runtime_errors", 
                    "difficulty": "advanced",
                    "handles_execution_failure": True
                }
            }
            examples.append(example)
        
        return examples

def main():
    """Generate complete error handling dataset."""
    print("🔧 Generating ByteLogic Error Handling Dataset...")
    
    generator = ErrorHandlingGenerator()
    
    # Generate different types of error examples
    parse_errors = generator.generate_parse_error_examples(50)
    recovery_examples = generator.generate_recovery_examples(30) 
    runtime_errors = generator.generate_runtime_error_examples(20)
    
    all_examples = parse_errors + recovery_examples + runtime_errors
    total = len(all_examples)
    
    # Split into train/val/test
    random.shuffle(all_examples)
    train_split = int(0.8 * total)
    val_split = int(0.9 * total)
    
    dataset = {
        "metadata": {
            "version": "1.0",
            "generator": "Error Handling Dataset Generator",
            "total_examples": total,
            "train_examples": train_split,
            "val_examples": val_split - train_split,
            "test_examples": total - val_split,
            "categories": ["parse_errors", "error_recovery", "runtime_errors"],
            "features": ["error_detection", "error_explanation", "error_recovery"]
        },
        "train": all_examples[:train_split],
        "validation": all_examples[train_split:val_split],
        "test": all_examples[val_split:]
    }
    
    # Save dataset
    output_file = "training/datasets/bytelogic_error_handling_dataset.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Generated {total} error handling examples")
    print(f"   Parse errors: {len(parse_errors)}")
    print(f"   Recovery examples: {len(recovery_examples)}")
    print(f"   Runtime errors: {len(runtime_errors)}")
    print(f"   Saved to: {output_file}")
    
    return output_file

if __name__ == "__main__":
    main()