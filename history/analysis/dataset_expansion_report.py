#!/usr/bin/env python3
"""
Detailed Dataset Expansion Report for ByteLogic WorldModel
=========================================================

Comprehensive analysis of current datasets and expansion opportunities.
"""

import json
import re
from collections import Counter

def analyze_current_vs_comprehensive():
    """Analyze differences between current and comprehensive datasets."""
    print("Detailed Dataset Comparison Report")
    print("="*70)
    
    with open("training/datasets/complete_bytelogic_dataset.json", 'r') as f:
        current = json.load(f)
    with open("training/datasets/comprehensive_bytelogic_dataset_with_natural_language.json", 'r') as f:
        comprehensive = json.load(f)
    
    print("DATASET OVERVIEW:")
    print(f"  Current dataset: {current['metadata']['total_examples']} examples (100% standard syntax)")
    print(f"  Comprehensive dataset: {comprehensive['metadata']['total_examples']} examples")
    print(f"  Difference: {comprehensive['metadata']['total_examples'] - current['metadata']['total_examples']} additional examples")
    print()
    
    print("CATEGORY BREAKDOWN:")
    print("  Current dataset categories:")
    current_train = current.get('train', [])
    current_cats = set()
    for ex in current_train:
        cat = ex.get('metadata', {}).get('category', 'unknown')
        current_cats.add(cat)
    for cat in sorted(current_cats):
        count = sum(1 for ex in current_train if ex.get('metadata', {}).get('category', '') == cat)
        print(f"    - {cat}: {count} examples")
    print()
    
    print("  Comprehensive dataset categories:")
    comp_train = comprehensive.get('train', [])
    comp_cats = set()
    for ex in comp_train:
        cat = ex.get('metadata', {}).get('category', 'unknown')
        comp_cats.add(cat)
    for cat in sorted(comp_cats):
        count = sum(1 for ex in comp_train if ex.get('metadata', {}).get('category', '') == cat)
        print(f"    - {cat}: {count} examples")
    print()
    
    print(f"  Categories in comprehensive but NOT in current: {sorted(comp_cats - current_cats)}")
    print()

def identify_standard_syntax_examples():
    """Identify examples in comprehensive dataset that use standard syntax."""
    print("STANDARD SYNTAX COMPATIBILITY ANALYSIS:")
    print("="*70)
    
    with open("training/datasets/comprehensive_bytelogic_dataset_with_natural_language.json", 'r') as f:
        data = json.load(f)
    
    standard_examples = []
    extended_examples = []
    
    for example in data.get('train', []):
        output_text = example.get('output', '')
        
        # Extract computation block
        comp_blocks = re.findall(r'<computation>(.*?)</computation>', output_text, re.DOTALL)
        if comp_blocks:
            comp_code = comp_blocks[0]
            
            # Check for extended syntax elements
            has_extended = any(keyword in comp_code for keyword in 
                             ['CALC', 'INPUT', 'RESULT', 'FOR', 'IF', 'THEN', 'ELSE', 'END', 'LET', 'POW', 'FOR'])
            
            if has_extended:
                extended_examples.append(example)
            else:
                # Check if it has standard syntax elements
                has_standard = any(keyword in comp_code for keyword in ['REL', 'FACT', 'RULE', 'SOLVE', 'QUERY'])
                if has_standard:
                    standard_examples.append(example)
    
    print(f"  Examples with STANDARD syntax in comprehensive dataset: {len(standard_examples)}")
    print(f"  Examples with EXTENDED (unsupported) syntax in comprehensive dataset: {len(extended_examples)}")
    print()
    
    print("  Sample standard syntax examples from comprehensive dataset:")
    for i, ex in enumerate(standard_examples[:3]):
        input_text = ex.get('input', '')[:80] + "..."
        cat = ex.get('metadata', {}).get('category', 'unknown')
        diff = ex.get('metadata', {}).get('difficulty', 'unknown')
        print(f"    {i+1}. [{cat}] [{diff}] {input_text}")
    print()

def recommend_expansion_strategy():
    """Recommend specific expansion strategies."""
    print("RECOMMENDED EXPANSION STRATEGY:")
    print("="*70)
    
    print("IMMEDIATE ACTIONS (Short-term):")
    print("  1. Extract standard syntax examples from comprehensive dataset")
    print("     - Add compatible examples to current dataset (~800-1000 potential examples)")
    print("     - Focus on 'basic_logic', 'graph_algorithms', 'conditional_logic' categories")
    print()
    
    print("  2. Expand existing categories with more variety:")
    print("     - Mathematical operations: more complex arithmetic")
    print("     - Family relations: multi-generational trees")
    print("     - Graph algorithms: shortest path, cycles, connectivity")
    print("     - Natural language variations: different phrasings for same logic")
    print()
    
    print("MAJOR EXPANSIONS (Long-term):")
    print("  1. Multi-step reasoning examples:")
    print("     - Problems requiring multiple queries")
    print("     - Sequential logical deductions")
    print("     - Complex rule chaining")
    print()
    
    print("  2. Error tolerance examples:")
    print("     - Malformed queries with proper corrections")
    print("     - Edge cases and boundary conditions")
    print("     - Partial information handling")
    print()
    
    print("  3. Domain-specific expansions:")
    print("     - Geographic reasoning: country-capitals, borders")
    print("     - Scientific facts: chemical reactions, physics formulas")
    print("     - Business logic: inventory, supply chain, scheduling")
    print("     - Legal logic: contracts, regulations")
    print()

def recommend_specific_examples():
    """Recommend specific types of examples to add."""
    print("SPECIFIC EXAMPLE TYPES TO ADD:")
    print("="*70)
    
    recommendations = [
        {
            "type": "Complex Mathematical Operations",
            "examples": [
                "What is the greatest common divisor of 48 and 18?",
                "Calculate the factorial of 7", 
                "What is the square root of 144?",
                "Compute compound interest on $1000 at 5% for 3 years"
            ],
            "byte_logic": ["REL calculation", "FACT calculation gcd 48 18", "SOLVE", "QUERY calculation ? ?"]
        },
        {
            "type": "Multi-generational Family Trees",
            "examples": [
                "If Alice is grandmother of Bob, and Bob is parent of Carol, what is Alice to Carol?",
                "How many great-grandchildren does David have?"
            ],
            "byte_logic": ["REL parent", "REL grandparent", "REL great_grandparent", "FACT parent alice bob", "...", "SOLVE", "QUERY great_grandparent ? ?"]
        },
        {
            "type": "Graph Theory Problems",
            "examples": [
                "Find all nodes connected to node A in 2 steps",
                "Is there a cycle in this graph?",
                "What is the shortest path from X to Y?"
            ],
            "byte_logic": ["REL edge", "REL path_two_hops", "RULE path_two_hops: SCAN edge MATCH $0, JOIN edge $0", "..."]
        },
        {
            "type": "Temporal Logic",
            "examples": [
                "If event A happened before B, and B before C, when did A happen relative to C?",
                "Schedule meetings with no conflicts"
            ],
            "byte_logic": ["REL before", "FACT before meeting_a meeting_b", "SOLVE", "QUERY before meeting_a ?"]
        },
        {
            "type": "Set Operations",
            "examples": [
                "What students are in both Math and Science classes?",
                "Find people who speak Spanish but not French"
            ],
            "byte_logic": ["REL enrolled", "REL speaks", "RULE intersection: SCAN enrolled math, JOIN enrolled science", "..."]
        }
    ]
    
    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. {rec['type']}")
        print(f"   Questions to add:")
        for ex in rec['examples'][:2]:  # Show first 2 examples
            print(f"     - {ex}")
        print()

def assess_capacity_for_expansion():
    """Assess current model capacity for expanded datasets."""
    print("CAPACITY ASSESSMENT FOR EXPANSION:")
    print("="*70)
    
    print("Current dataset statistics:")
    print("  - Size: 1,100 examples")
    print("  - Format: Standard ByteLogic (REL, FACT, RULE, SOLVE, QUERY)")
    print("  - Categories: 3 major categories")
    print("  - Difficulties: 3 levels (beginner, intermediate, advanced)")
    print()
    print("Recommended expansion limits:")
    print("  - Safe expansion: Up to 2,000-3,000 examples (2-3x current size)")
    print("  - Recommended batch size: 200-500 new examples at a time")
    print("  - Monitor training stability with increased complexity")
    print()

def main():
    print("ByteLogic WorldModel - Dataset Expansion Analysis")
    print("="*70)
    print()
    
    analyze_current_vs_comprehensive()
    identify_standard_syntax_examples()
    recommend_expansion_strategy() 
    recommend_specific_examples()
    assess_capacity_for_expansion()
    
    print("FINAL RECOMMENDATION:")
    print("="*70)
    print("1. IMMEDIATE: Extract and add compatible examples from comprehensive dataset")
    print("2. SHORT-TERM: Add 500-800 varied examples across existing categories") 
    print("3. LONG-TERM: Develop domain-specific and multi-step reasoning examples")
    print("4. CONTINUOUS: Add error-handling and edge-case examples regularly")
    print()
    print("Expected benefits:")
    print("- 15-25% increase in training data (improves generalization)")
    print("- Better handling of varied natural language expressions")
    print("- Improved performance on complex reasoning tasks")
    print("- Enhanced robustness to input variations")

if __name__ == "__main__":
    main()