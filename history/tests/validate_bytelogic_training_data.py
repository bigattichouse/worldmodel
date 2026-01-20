#!/usr/bin/env python3
"""
ByteLogic Training Data Validator
=================================

Validates the quality and syntax correctness of ByteLogic training data.
"""

import json
import re
import subprocess
import tempfile
import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple


class ByteLogicValidator:
    """Validator for ByteLogic training examples."""
    
    def __init__(self):
        self.bytelogic_compiler = "../bytelogic/build/bytelogic"
        self.valid_examples = 0
        self.invalid_examples = 0
        self.syntax_errors = []
        self.compilation_errors = []
        
    def validate_syntax(self, bytelogic_code: str) -> Tuple[bool, str]:
        """Validate ByteLogic syntax using the compiler."""
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.bl', delete=False) as f:
                f.write(bytelogic_code)
                temp_file = f.name
            
            # Test syntax by trying to compile
            result = subprocess.run([
                self.bytelogic_compiler, 
                "-c", "wat", "-o", "/dev/null",
                temp_file
            ], capture_output=True, text=True, timeout=5)
            
            os.unlink(temp_file)
            
            if result.returncode == 0:
                return True, ""
            else:
                return False, result.stderr.strip()
                
        except Exception as e:
            if 'temp_file' in locals():
                try:
                    os.unlink(temp_file)
                except:
                    pass
            return False, f"Validation error: {e}"
    
    def extract_computation_tokens(self, text: str) -> List[str]:
        """Extract ByteLogic code from computation tokens."""
        pattern = re.compile(r'<computation>\s*(.*?)\s*</computation>', re.DOTALL | re.IGNORECASE)
        matches = pattern.findall(text)
        return [match.strip() for match in matches]
    
    def validate_example(self, example: Dict[str, Any], idx: int) -> Dict[str, Any]:
        """Validate a single training example."""
        validation_result = {
            "index": idx,
            "valid": True,
            "errors": [],
            "warnings": [],
            "bytelogic_code_count": 0
        }
        
        # Check required fields
        if 'input' not in example:
            validation_result["valid"] = False
            validation_result["errors"].append("Missing 'input' field")
        
        if 'output' not in example:
            validation_result["valid"] = False
            validation_result["errors"].append("Missing 'output' field")
            return validation_result
        
        # Extract and validate ByteLogic code
        computation_codes = self.extract_computation_tokens(example['output'])
        validation_result["bytelogic_code_count"] = len(computation_codes)
        
        if len(computation_codes) == 0:
            validation_result["warnings"].append("No computation tokens found")
        elif len(computation_codes) > 1:
            validation_result["warnings"].append(f"Multiple computation tokens found: {len(computation_codes)}")
        
        # Validate each ByteLogic code block
        for i, code in enumerate(computation_codes):
            if not code.strip():
                validation_result["errors"].append(f"Empty ByteLogic code block {i}")
                validation_result["valid"] = False
                continue
            
            # Syntax validation
            is_valid, error_msg = self.validate_syntax(code)
            if not is_valid:
                validation_result["valid"] = False
                validation_result["errors"].append(f"Syntax error in block {i}: {error_msg}")
                self.syntax_errors.append({
                    "example_idx": idx,
                    "block_idx": i,
                    "code": code[:100] + "...",
                    "error": error_msg
                })
        
        # Check metadata if available
        metadata = example.get('metadata', {})
        if metadata:
            if 'category' not in metadata:
                validation_result["warnings"].append("Missing category in metadata")
            if 'difficulty' not in metadata:
                validation_result["warnings"].append("Missing difficulty in metadata")
        
        return validation_result
    
    def validate_dataset(self, dataset_file: str) -> Dict[str, Any]:
        """Validate entire dataset."""
        print(f"🔍 Validating ByteLogic dataset: {dataset_file}")
        
        # Load dataset
        try:
            with open(dataset_file, 'r') as f:
                if dataset_file.endswith('.jsonl'):
                    examples = []
                    for line in f:
                        if line.strip():
                            examples.append(json.loads(line))
                else:
                    data = json.load(f)
                    if isinstance(data, dict):
                        if 'train' in data:
                            examples = data['train'] + data.get('validation', []) + data.get('test', [])
                        else:
                            examples = [data]
                    elif isinstance(data, list):
                        examples = data
                    else:
                        examples = [data]
        except Exception as e:
            return {"error": f"Failed to load dataset: {e}"}
        
        print(f"   📊 Total examples: {len(examples)}")
        
        # Validate examples
        validation_results = []
        categories = {}
        difficulties = {}
        
        for idx, example in enumerate(examples):
            if idx % 100 == 0 and idx > 0:
                print(f"   📝 Validated {idx}/{len(examples)} examples...")
            
            result = self.validate_example(example, idx)
            validation_results.append(result)
            
            if result["valid"]:
                self.valid_examples += 1
            else:
                self.invalid_examples += 1
            
            # Track categories and difficulties
            metadata = example.get('metadata', {})
            category = metadata.get('category', 'unknown')
            difficulty = metadata.get('difficulty', 'unknown')
            
            categories[category] = categories.get(category, 0) + 1
            difficulties[difficulty] = difficulties.get(difficulty, 0) + 1
        
        # Calculate statistics
        total_examples = len(examples)
        valid_percentage = (self.valid_examples / total_examples * 100) if total_examples > 0 else 0
        
        # Count examples with computation tokens
        examples_with_computation = sum(1 for r in validation_results if r["bytelogic_code_count"] > 0)
        computation_percentage = (examples_with_computation / total_examples * 100) if total_examples > 0 else 0
        
        summary = {
            "total_examples": total_examples,
            "valid_examples": self.valid_examples,
            "invalid_examples": self.invalid_examples,
            "valid_percentage": valid_percentage,
            "examples_with_computation": examples_with_computation,
            "computation_percentage": computation_percentage,
            "categories": categories,
            "difficulties": difficulties,
            "syntax_errors": len(self.syntax_errors),
            "compilation_errors": len(self.compilation_errors)
        }
        
        return {
            "summary": summary,
            "validation_results": validation_results,
            "syntax_errors": self.syntax_errors[:10],  # First 10 errors
            "compilation_errors": self.compilation_errors[:10]
        }
    
    def print_validation_summary(self, results: Dict[str, Any]):
        """Print validation results summary."""
        summary = results["summary"]
        
        print(f"\n📊 Validation Summary:")
        print(f"=" * 60)
        print(f"   Total examples: {summary['total_examples']}")
        print(f"   Valid examples: {summary['valid_examples']} ({summary['valid_percentage']:.1f}%)")
        print(f"   Invalid examples: {summary['invalid_examples']}")
        print(f"   Examples with computation: {summary['examples_with_computation']} ({summary['computation_percentage']:.1f}%)")
        print(f"   Syntax errors: {summary['syntax_errors']}")
        print(f"   Compilation errors: {summary['compilation_errors']}")
        
        print(f"\n📈 Categories:")
        for category, count in sorted(summary['categories'].items()):
            print(f"   {category}: {count}")
        
        print(f"\n📊 Difficulty levels:")
        for difficulty, count in sorted(summary['difficulties'].items()):
            print(f"   {difficulty}: {count}")
        
        # Show sample errors
        if results["syntax_errors"]:
            print(f"\n❌ Sample Syntax Errors:")
            for i, error in enumerate(results["syntax_errors"][:5]):
                print(f"   {i+1}. Example {error['example_idx']}: {error['error']}")
                print(f"      Code: {error['code']}")
        
        # Overall assessment
        if summary['valid_percentage'] >= 95:
            print(f"\n✅ Dataset quality: EXCELLENT ({summary['valid_percentage']:.1f}% valid)")
        elif summary['valid_percentage'] >= 85:
            print(f"\n⚠️  Dataset quality: GOOD ({summary['valid_percentage']:.1f}% valid)")
        elif summary['valid_percentage'] >= 70:
            print(f"\n⚠️  Dataset quality: FAIR ({summary['valid_percentage']:.1f}% valid)")
        else:
            print(f"\n❌ Dataset quality: POOR ({summary['valid_percentage']:.1f}% valid)")


def main():
    """Main validation function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate ByteLogic training data")
    parser.add_argument("dataset", help="Path to dataset file (JSON or JSONL)")
    parser.add_argument("--output", help="Output validation report to file")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    print("🚀 ByteLogic Training Data Validation")
    print("=" * 60)
    
    # Check if ByteLogic compiler exists
    validator = ByteLogicValidator()
    if not os.path.exists(validator.bytelogic_compiler):
        print(f"❌ ByteLogic compiler not found at {validator.bytelogic_compiler}")
        print("   Run: cd bytelogic && make")
        return False
    
    # Validate dataset
    results = validator.validate_dataset(args.dataset)
    
    if "error" in results:
        print(f"❌ {results['error']}")
        return False
    
    # Print summary
    validator.print_validation_summary(results)
    
    # Save detailed report if requested
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n💾 Detailed validation report saved to {args.output}")
    
    # Return success if validation quality is acceptable
    success = results["summary"]["valid_percentage"] >= 85
    
    if success:
        print(f"\n🎉 Dataset validation PASSED! Ready for training.")
    else:
        print(f"\n❌ Dataset validation FAILED! Please fix errors before training.")
    
    return success


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)