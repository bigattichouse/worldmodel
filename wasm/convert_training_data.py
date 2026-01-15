#!/usr/bin/env python3
"""
Training Data Conversion Script
==============================

Converts existing WorldModel training data to WASM format with curriculum stages.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.data.auto_converter import PythonToWATConverter, ConversionResult
from typing import List, Dict
import json
import re


class TrainingDataConverter:
    """Converts WorldModel training data to WASM format."""
    
    def __init__(self):
        self.converter = PythonToWATConverter()
        self.curriculum_stages = {
            "basic_arithmetic": [],
            "system_operations": [],
            "complex_logic": []
        }
    
    def load_training_file(self, filepath: str) -> List[str]:
        """Load training file and split into examples."""
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Split by double newlines to get individual examples
        examples = []
        raw_examples = content.split('\n\n')
        
        for raw_example in raw_examples:
            if raw_example.strip() and 'User:' in raw_example and 'Assistant:' in raw_example:
                examples.append(raw_example.strip())
        
        return examples
    
    def categorize_example(self, example: str) -> str:
        """Categorize example into curriculum stage."""
        user_text = self._extract_user_text(example)
        
        # Basic arithmetic patterns
        basic_patterns = [
            r'\d+\s*[+\-*/]\s*\d+',
            r'\d+%\s+of\s+\d+',
            r'calculate.*\d+.*\d+',
            r'multiply.*\d+.*\d+',
            r'add.*\d+.*\d+'
        ]
        
        # System operation patterns
        system_patterns = [
            r'current date',
            r'current time', 
            r'datetime',
            r'list.*file',
            r'platform',
            r'system.*info'
        ]
        
        # Check patterns
        for pattern in basic_patterns:
            if re.search(pattern, user_text.lower()):
                return "basic_arithmetic"
        
        for pattern in system_patterns:
            if re.search(pattern, user_text.lower()):
                return "system_operations"
        
        return "complex_logic"
    
    def _extract_user_text(self, example: str) -> str:
        """Extract user text from example."""
        match = re.search(r'User:\s*(.*?)(?=Assistant:|$)', example, re.DOTALL)
        return match.group(1).strip() if match else ""
    
    def convert_examples(self, examples: List[str]) -> Dict[str, List[Dict]]:
        """Convert examples to WASM format and categorize."""
        converted = {
            "basic_arithmetic": [],
            "system_operations": [],
            "complex_logic": []
        }
        
        stats = {
            "total": len(examples),
            "successful": 0,
            "failed": 0,
            "by_category": {cat: 0 for cat in converted.keys()}
        }
        
        print(f"Converting {len(examples)} training examples...")
        
        for i, example in enumerate(examples):
            if i % 100 == 0:
                print(f"  Progress: {i}/{len(examples)}")
            
            # Categorize example
            category = self.categorize_example(example)
            stats["by_category"][category] += 1
            
            # Convert to WASM
            conversion_result = self.converter.convert_training_example(example)
            
            # Create enhanced example
            enhanced_example = {
                "original": example,
                "category": category,
                "wat_code": conversion_result.wat_code if conversion_result.success else None,
                "success": conversion_result.success,
                "error": conversion_result.error,
                "inputs": conversion_result.inputs,
                "expected_output": conversion_result.expected_output,
                "enhanced_format": self._create_enhanced_format(example, conversion_result)
            }
            
            converted[category].append(enhanced_example)
            
            if conversion_result.success:
                stats["successful"] += 1
            else:
                stats["failed"] += 1
        
        print(f"\n✅ Conversion completed!")
        print(f"  Total: {stats['total']}")
        print(f"  Successful: {stats['successful']}")
        print(f"  Failed: {stats['failed']}")
        print(f"  By category:")
        for cat, count in stats["by_category"].items():
            success_count = len([ex for ex in converted[cat] if ex["success"]])
            print(f"    {cat}: {count} total, {success_count} converted")
        
        return converted
    
    def _create_enhanced_format(self, original_example: str, result: ConversionResult) -> str:
        """Create enhanced training format with WASM and <computed> tokens."""
        if not result.success:
            return original_example
        
        # Extract parts of original example
        user_match = re.search(r'User:\s*(.*?)(?=Assistant:|$)', original_example, re.DOTALL)
        assistant_match = re.search(r'Assistant:\s*(.*?)(?=$)', original_example, re.DOTALL)
        
        if not user_match or not assistant_match:
            return original_example
        
        user_text = user_match.group(1).strip()
        assistant_text = assistant_match.group(1).strip()
        
        # Extract <think> and <requires> if present
        think_match = re.search(r'<think>(.*?)</think>', assistant_text, re.DOTALL)
        requires_match = re.search(r'<requires>(.*?)</requires>', assistant_text, re.DOTALL)
        
        think_text = think_match.group(1).strip() if think_match else "I need to solve this problem."
        requires_text = requires_match.group(1).strip() if requires_match else "wasm:computation"
        
        # Create enhanced format
        enhanced = f"""User: {user_text}
Assistant: <think>{think_text}</think>
<wat_model>
{result.wat_code}
</wat_model>"""
        
        # Add computed token if we have expected output
        if result.expected_output is not None:
            enhanced += f"\n<computed>{result.expected_output}</computed>"
        
        # Add requires
        enhanced += f"\n<requires>{requires_text}</requires>"
        
        # Add final answer (extract from original)
        final_answer = self._extract_final_answer(assistant_text)
        if final_answer:
            enhanced += f"\n\n{final_answer}"
        
        return enhanced
    
    def _extract_final_answer(self, assistant_text: str) -> str:
        """Extract final answer from assistant text."""
        # Remove <think>, <model>, <requires> blocks
        cleaned = re.sub(r'<think>.*?</think>', '', assistant_text, flags=re.DOTALL)
        cleaned = re.sub(r'<model>.*?</model>', '', cleaned, flags=re.DOTALL)
        cleaned = re.sub(r'<requires>.*?</requires>', '', cleaned, flags=re.DOTALL)
        
        # Clean up whitespace
        lines = [line.strip() for line in cleaned.split('\n') if line.strip()]
        return '\n'.join(lines) if lines else ""
    
    def save_converted_data(self, converted_data: Dict, output_dir: str):
        """Save converted data to files."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save by curriculum stage
        for category, examples in converted_data.items():
            output_file = os.path.join(output_dir, f"{category}.jsonl")
            
            with open(output_file, 'w', encoding='utf-8') as f:
                for example in examples:
                    json.dump(example, f, ensure_ascii=False)
                    f.write('\n')
            
            print(f"Saved {len(examples)} examples to {output_file}")
        
        # Save training-ready format
        for category, examples in converted_data.items():
            successful_examples = [ex for ex in examples if ex["success"]]
            
            if successful_examples:
                training_file = os.path.join(output_dir, f"{category}_training.txt")
                
                with open(training_file, 'w', encoding='utf-8') as f:
                    for example in successful_examples:
                        f.write(example["enhanced_format"])
                        f.write('\n\n')
                
                print(f"Saved {len(successful_examples)} training examples to {training_file}")


def main():
    """Convert training data."""
    print("🚀 WorldModel Training Data → WASM Conversion")
    print("=" * 60)
    
    converter = TrainingDataConverter()
    
    # Load training data
    input_file = "data/original/worldmodel_training_combined.txt"
    print(f"Loading training data from {input_file}")
    
    examples = converter.load_training_file(input_file)
    print(f"✅ Loaded {len(examples)} examples")
    
    # Convert examples
    converted_data = converter.convert_examples(examples)
    
    # Save converted data
    output_dir = "data/converted"
    converter.save_converted_data(converted_data, output_dir)
    
    print(f"\n🎉 Conversion complete! Check {output_dir}/ for results.")
    
    # Show sample conversions
    print(f"\n📝 Sample conversions:")
    for category, examples in converted_data.items():
        successful = [ex for ex in examples if ex["success"]]
        if successful:
            sample = successful[0]
            print(f"\n--- {category.upper()} SAMPLE ---")
            print("Original:")
            print("  " + sample["original"][:100] + "...")
            print("Enhanced:")
            print("  " + sample["enhanced_format"][:200] + "...")


if __name__ == "__main__":
    main()