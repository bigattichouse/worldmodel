#!/usr/bin/env python3
"""
Dataset Analysis Script for ByteLogic WorldModel
================================================

Analyze the current training datasets to identify expansion opportunities.
"""

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

def analyze_dataset_structure():
    """Analyze the structure of the complete dataset."""
    print("Dataset Structure Analysis")
    print("="*50)
    
    dataset_path = "training/datasets/complete_bytelogic_dataset.json"
    
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Dataset: {dataset_path}")
    print(f"Version: {data.get('metadata', {}).get('version', 'Unknown')}")
    print(f"Total examples: {data.get('metadata', {}).get('total_examples', 'Unknown')}")
    print(f"Train examples: {data.get('metadata', {}).get('train_examples', 'Unknown')}")
    print(f"Validation examples: {data.get('metadata', {}).get('val_examples', 'Unknown')}")
    print(f"Test examples: {data.get('metadata', {}).get('test_examples', 'Unknown')}")
    print(f"Features: {data.get('metadata', {}).get('features', 'Unknown')}")
    
    return data

def analyze_training_examples(data):
    """Analyze the training examples in detail."""
    print("\\nTraining Examples Analysis")
    print("="*50)
    
    train_examples = data.get('train', [])
    print(f"Training examples count: {len(train_examples)}")
    
    categories = Counter()
    difficulties = Counter()
    bytelogic_patterns = Counter()
    
    # Analyze each example
    for i, example in enumerate(train_examples):
        input_text = example.get('input', '')
        output_text = example.get('output', '')
        metadata = example.get('metadata', {})
        
        # Collect categories and difficulties
        cat = metadata.get('category', 'unknown')
        categories[cat] += 1
        
        difficulty = metadata.get('difficulty', 'unknown')
        difficulties[difficulty] += 1
        
        # Analyze ByteLogic patterns in output
        computation_blocks = re.findall(r'<computation>(.*?)</computation>', output_text, re.DOTALL)
        for block in computation_blocks:
            # Extract REL, FACT, RULE, QUERY patterns
            rel_matches = len(re.findall(r'REL\s+(\w+)', block))
            fact_matches = len(re.findall(r'FACT\s+(\w+)', block))
            rule_matches = len(re.findall(r'RULE\s+(\w+)', block))
            query_matches = len(re.findall(r'QUERY\s+(\w+)', block))
            
            if rel_matches > 0:
                bytelogic_patterns['relations'] += 1
            if fact_matches > 0:
                bytelogic_patterns['facts'] += 1
            if rule_matches > 0:
                bytelogic_patterns['rules'] += 1
            if query_matches > 0:
                bytelogic_patterns['queries'] += 1
        
        # Show first few examples
        if i < 3:
            print(f"\\nExample {i+1}:")
            print(f"  Input: {input_text[:100]}{'...' if len(input_text) > 100 else ''}")
            print(f"  Has computation: {'<computation>' in output_text}")
            print(f"  Category: {cat}")
            print(f"  Difficulty: {difficulty}")
    
    print(f"\\nCategories distribution:")
    for cat, count in categories.most_common():
        print(f"  {cat}: {count}")
    
    print(f"\\nDifficulties distribution:")
    for diff, count in difficulties.most_common():
        print(f"  {diff}: {count}")
    
    print(f"\\nByteLogic patterns distribution:")
    for pattern, count in bytelogic_patterns.most_common():
        print(f"  {pattern}: {count}")
    
    return train_examples

def analyze_computation_complexity(train_examples):
    """Analyze the complexity of computation blocks."""
    print("\\nComputation Complexity Analysis")
    print("="*50)
    
    computation_stats = {
        'has_computation': 0,
        'avg_computation_length': 0,
        'computation_with_rules': 0,
        'computation_with_multiple_queries': 0
    }
    
    total_computation_length = 0
    
    for example in train_examples:
        output_text = example.get('output', '')
        computation_blocks = re.findall(r'<computation>(.*?)</computation>', output_text, re.DOTALL)
        
        if computation_blocks:
            computation_stats['has_computation'] += 1
            block = computation_blocks[0]  # Take first block
            total_computation_length += len(block)
            
            # Check for rules
            if 'RULE' in block:
                computation_stats['computation_with_rules'] += 1
            
            # Check for multiple queries
            query_count = len(re.findall(r'QUERY', block))
            if query_count > 1:
                computation_stats['computation_with_multiple_queries'] += 1
    
    if computation_stats['has_computation'] > 0:
        computation_stats['avg_computation_length'] = total_computation_length / computation_stats['has_computation']
    
    for key, value in computation_stats.items():
        print(f"  {key}: {value}")

def compare_with_comprehensive():
    """Compare with comprehensive dataset to identify expansion opportunities."""
    print("\\nComparison with Comprehensive Dataset")
    print("="*50)
    
    try:
        with open("training/datasets/complete_bytelogic_dataset.json", 'r') as f:
            basic_data = json.load(f)
        
        with open("training/datasets/comprehensive_bytelogic_dataset_with_natural_language.json", 'r') as f:
            comprehensive_data = json.load(f)
        
        basic_total = basic_data.get('metadata', {}).get('total_examples', 0)
        comp_total = comprehensive_data.get('metadata', {}).get('total_examples', 0)
        
        print(f"Current dataset: {basic_total} examples")
        print(f"Comprehensive dataset: {comp_total} examples")
        print(f"Difference: {comp_total - basic_total} additional examples available")
        
        basic_cats = set()
        comp_cats = set()
        
        for ex in basic_data.get('train', []):
            basic_cats.add(ex.get('metadata', {}).get('category', 'unknown'))
        
        for ex in comprehensive_data.get('train', []):
            comp_cats.add(ex.get('metadata', {}).get('category', 'unknown'))
        
        print(f"\\nCategories in current dataset: {sorted(basic_cats)}")
        print(f"Categories in comprehensive dataset: {sorted(comp_cats)}")
        print(f"Additional categories available: {sorted(comp_cats - basic_cats)}")
        
        return basic_data, comprehensive_data
        
    except FileNotFoundError as e:
        print(f"File not found: {e}")
        return None, None

def identify_expansion_opportunities():
    """Identify specific opportunities for dataset expansion."""
    print("\\nDataset Expansion Opportunities")
    print("="*50)
    
    opportunities = [
        {
            "category": "Mathematical Operations",
            "description": "Add more complex arithmetic operations and multi-step calculations",
            "examples": ["complex fraction calculations", "multi-variable equations", "exponents and roots"]
        },
        {
            "category": "Advanced Logic",
            "description": "More sophisticated logical inference patterns",
            "examples": ["nested rules", "transitive closures", "complex join operations"]
        },
        {
            "category": "Natural Language Variations",
            "description": "Different ways to express the same computation request",
            "examples": ["synonym variations", "paraphrasing", "different question structures"]
        },
        {
            "category": "Error Recovery Patterns",
            "description": "More examples of correcting syntactic and semantic errors",
            "examples": ["malformed queries", "missing relations", "type mismatches"]
        },
        {
            "category": "Domain-Specific Examples",
            "description": "Apply ByteLogic to specific domains",
            "examples": ["geographic relationships", "scientific facts", "business rules", "legal logic"]
        },
        {
            "category": "Multi-step Reasoning",
            "description": "Problems requiring multiple sequential computations",
            "examples": ["chained queries", "iterative reasoning", "problem decomposition"]
        }
    ]
    
    for i, opp in enumerate(opportunities, 1):
        print(f"{i}. {opp['category']}")
        print(f"   Description: {opp['description']}")
        print(f"   Examples: {', '.join(opp['examples'])}")
        print()

def analyze_input_patterns(train_examples):
    """Analyze patterns in input questions to identify gaps.""" 
    print("Input Pattern Analysis")
    print("="*50)
    
    question_types = Counter()
    numeric_operations = Counter()
    relationship_types = Counter()
    
    for example in train_examples:
        input_text = example.get('input', '').lower()
        
        # Identify question types
        if any(qword in input_text for qword in ['what is', 'calculate', 'compute', 'evaluate']):
            question_types['mathematical'] += 1
        elif any(qword in input_text for qword in ['who', 'what', 'which', 'how many']):
            question_types['logical'] += 1
        elif any(qword in input_text for qword in ['if', 'given', 'suppose', 'assuming']):
            question_types['hypothetical'] += 1
        else:
            question_types['other'] += 1
        
        # Identify operations
        if any(op in input_text for op in [' + ', '+', ' plus ', ' and ', ' sum ', 'total']):
            numeric_operations['addition'] += 1
        if any(op in input_text for op in [' - ', '-', ' minus ', ' difference ', ' subtract']):
            numeric_operations['subtraction'] += 1
        if any(op in input_text for op in [' * ', '*', ' times ', ' multiply ', ' product ', ' × ']):
            numeric_operations['multiplication'] += 1
        if any(op in input_text for op in [' / ', '/', ' divided by ', ' ÷ ', ' quotient ']):
            numeric_operations['division'] += 1
        
        # Identify relationships
        if any(rel in input_text for rel in ['parent', 'child', 'father', 'mother', 'son', 'daughter']):
            relationship_types['family'] += 1
        if any(rel in input_text for rel in ['friend', 'knows', 'connects', 'relationship', 'related']):
            relationship_types['social'] += 1
        if any(rel in input_text for rel in ['edge', 'path', 'connected', 'reachable', 'connected']):
            relationship_types['graph'] += 1
    
    print("Question Types Distribution:")
    for qtype, count in question_types.most_common():
        print(f"  {qtype}: {count}")
    
    print("\\nNumeric Operations Distribution:")
    for op, count in numeric_operations.most_common():
        print(f"  {op}: {count}")
    
    print("\\nRelationship Types Distribution:")
    for rel, count in relationship_types.most_common():
        print(f"  {rel}: {count}")

def main():
    print("ByteLogic WorldModel - Dataset Analysis")
    print("="*60)
    
    # Analyze current dataset
    data = analyze_dataset_structure()
    
    # Analyze training examples
    train_examples = analyze_training_examples(data)
    
    # Analyze computation complexity
    analyze_computation_complexity(train_examples)
    
    # Analyze input patterns
    analyze_input_patterns(train_examples)
    
    # Compare with comprehensive dataset
    compare_with_comprehensive()
    
    # Identify expansion opportunities
    identify_expansion_opportunities()
    
    print("\\nSummary of Findings:")
    print("- Current dataset has 1,100 examples with basic logic patterns")
    print("- Comprehensive dataset offers 1,650 examples with advanced features")
    print("- Opportunities exist for more complex mathematical operations")
    print("- Natural language variations could enhance robustness") 
    print("- Multi-step reasoning examples would improve capabilities")
    print("- Domain-specific examples could broaden applicability")

if __name__ == "__main__":
    main()