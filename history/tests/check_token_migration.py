#!/usr/bin/env python3
"""
Check ByteLogic Dataset Token Migration
======================================

Verifies that all WAT tokens have been replaced with computation tokens
and that the dataset uses the new ByteLogic format consistently.
"""

import json
import re
from typing import Dict, List, Any, Tuple


def check_token_migration(dataset_file: str) -> Dict[str, Any]:
    """Check for WAT token remnants and verify computation token usage."""
    
    print(f"🔍 Checking token migration in: {dataset_file}")
    
    # Load dataset
    with open(dataset_file, 'r') as f:
        data = json.load(f)
    
    # Collect all examples
    all_examples = []
    if isinstance(data, dict):
        for split in ['train', 'validation', 'test']:
            if split in data:
                all_examples.extend(data[split])
    elif isinstance(data, list):
        all_examples = data
    
    print(f"   📊 Total examples to check: {len(all_examples)}")
    
    # Token patterns to check
    wat_patterns = [
        r'<wat_model>.*?</wat_model>',
        r'<wat_start>.*?<wat_end>',
        r'<computed>.*?</computed>',
        r'\(module',
        r'\(func.*export',
        r'i32\.add|i32\.sub|i32\.mul|i32\.div',
        r'local\.get|local\.set',
        r'f64\.add|f64\.sub|f64\.mul|f64\.div'
    ]
    
    computation_patterns = [
        r'<computation>.*?</computation>',
        r'REL\s+\w+',
        r'FACT\s+\w+\s+\w+\s+\w+',
        r'RULE\s+\w+:',
        r'SCAN\s+\w+',
        r'JOIN\s+\w+',
        r'EMIT\s+\w+',
        r'SOLVE',
        r'QUERY\s+\w+'
    ]
    
    # Check results
    results = {
        'wat_token_found': False,
        'computation_token_found': False,
        'wat_examples': [],
        'computation_examples': 0,
        'missing_computation_examples': [],
        'mixed_format_examples': [],
        'wat_pattern_matches': {},
        'computation_pattern_matches': {},
        'summary': {}
    }
    
    # Check each example
    for idx, example in enumerate(all_examples):
        output_text = example.get('output', '')
        
        # Check for WAT patterns
        wat_found = False
        for pattern in wat_patterns:
            matches = re.findall(pattern, output_text, re.IGNORECASE | re.DOTALL)
            if matches:
                wat_found = True
                results['wat_token_found'] = True
                if pattern not in results['wat_pattern_matches']:
                    results['wat_pattern_matches'][pattern] = []
                results['wat_pattern_matches'][pattern].append({
                    'example_idx': idx,
                    'matches': matches[:3]  # First 3 matches
                })
        
        if wat_found:
            results['wat_examples'].append({
                'idx': idx,
                'input': example.get('input', '')[:100],
                'output_preview': output_text[:200]
            })
        
        # Check for computation patterns
        computation_found = False
        for pattern in computation_patterns:
            matches = re.findall(pattern, output_text, re.IGNORECASE | re.DOTALL)
            if matches:
                computation_found = True
                results['computation_token_found'] = True
                if pattern not in results['computation_pattern_matches']:
                    results['computation_pattern_matches'][pattern] = 0
                results['computation_pattern_matches'][pattern] += len(matches)
        
        if computation_found:
            results['computation_examples'] += 1
        else:
            results['missing_computation_examples'].append({
                'idx': idx,
                'input': example.get('input', '')[:100],
                'output_preview': output_text[:200]
            })
        
        # Check for mixed format
        if wat_found and computation_found:
            results['mixed_format_examples'].append({
                'idx': idx,
                'input': example.get('input', '')[:100]
            })
    
    # Generate summary
    results['summary'] = {
        'total_examples': len(all_examples),
        'examples_with_computation': results['computation_examples'],
        'examples_with_wat': len(results['wat_examples']),
        'examples_missing_computation': len(results['missing_computation_examples']),
        'examples_with_mixed_format': len(results['mixed_format_examples']),
        'migration_complete': not results['wat_token_found'] and results['computation_token_found'],
        'computation_coverage': (results['computation_examples'] / len(all_examples) * 100) if all_examples else 0
    }
    
    return results


def print_migration_report(results: Dict[str, Any]):
    """Print detailed migration report."""
    
    summary = results['summary']
    
    print(f"\n📊 Token Migration Report")
    print("=" * 60)
    print(f"   Total examples: {summary['total_examples']}")
    print(f"   Examples with computation tokens: {summary['examples_with_computation']} ({summary['computation_coverage']:.1f}%)")
    print(f"   Examples with WAT tokens: {summary['examples_with_wat']}")
    print(f"   Examples missing computation: {summary['examples_missing_computation']}")
    print(f"   Examples with mixed format: {summary['examples_with_mixed_format']}")
    
    # Migration status
    if summary['migration_complete']:
        print(f"\n✅ Migration Status: COMPLETE")
        print(f"   ✅ No WAT tokens found")
        print(f"   ✅ All examples use computation tokens")
    else:
        print(f"\n⚠️  Migration Status: INCOMPLETE")
        if results['wat_token_found']:
            print(f"   ❌ WAT tokens still present")
        if not results['computation_token_found']:
            print(f"   ❌ No computation tokens found")
    
    # WAT token details
    if results['wat_examples']:
        print(f"\n❌ WAT Token Remnants Found:")
        for i, example in enumerate(results['wat_examples'][:5]):
            print(f"   {i+1}. Example {example['idx']}: {example['input']}")
            print(f"      Output preview: {example['output_preview']}...")
    
    # WAT pattern details
    if results['wat_pattern_matches']:
        print(f"\n🔍 WAT Patterns Found:")
        for pattern, matches in results['wat_pattern_matches'].items():
            print(f"   Pattern: {pattern}")
            print(f"   Found in {len(matches)} examples")
    
    # Computation pattern details
    if results['computation_pattern_matches']:
        print(f"\n✅ ByteLogic Patterns Found:")
        for pattern, count in sorted(results['computation_pattern_matches'].items()):
            print(f"   {pattern}: {count} occurrences")
    
    # Missing computation examples
    if results['missing_computation_examples']:
        print(f"\n⚠️  Examples Missing Computation Tokens:")
        for i, example in enumerate(results['missing_computation_examples'][:5]):
            print(f"   {i+1}. Example {example['idx']}: {example['input']}")
            print(f"      Output preview: {example['output_preview']}...")


def check_specific_datasets():
    """Check our specific ByteLogic datasets."""
    
    datasets_to_check = [
        "../training/datasets/corrected_bytelogic_dataset.json",
        "../training/datasets/comprehensive_bytelogic_dataset.json"
    ]
    
    all_results = {}
    
    for dataset in datasets_to_check:
        try:
            print(f"\n🔍 Checking: {dataset}")
            results = check_token_migration(dataset)
            all_results[dataset] = results
            print_migration_report(results)
            
        except FileNotFoundError:
            print(f"   ⚠️  Dataset not found: {dataset}")
        except Exception as e:
            print(f"   ❌ Error checking {dataset}: {e}")
    
    return all_results


def main():
    """Main function to check token migration."""
    
    print("🚀 ByteLogic Token Migration Check")
    print("=" * 60)
    
    # Check our datasets
    results = check_specific_datasets()
    
    # Overall summary
    print(f"\n📋 Overall Migration Summary:")
    print("=" * 60)
    
    all_complete = True
    for dataset, result in results.items():
        if result and 'summary' in result:
            status = "✅ COMPLETE" if result['summary']['migration_complete'] else "❌ INCOMPLETE"
            coverage = result['summary']['computation_coverage']
            print(f"   {dataset}: {status} ({coverage:.1f}% coverage)")
            
            if not result['summary']['migration_complete']:
                all_complete = False
    
    if all_complete and results:
        print(f"\n🎉 All datasets successfully migrated to ByteLogic!")
        print(f"   Ready for ByteLogic-only training")
        return True
    elif results:
        print(f"\n⚠️  Some datasets still contain WAT tokens")
        print(f"   Migration needs to be completed")
        return False
    else:
        print(f"\n❌ No datasets found to check")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)