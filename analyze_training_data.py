#!/usr/bin/env python3
"""
Analyze Training Data
====================

Check what training data we have and if it's all consolidated.
"""

import json
import os

def analyze_dataset(file_path):
    """Analyze a single dataset file."""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        if isinstance(data, dict) and 'metadata' in data:
            # Main dataset format
            metadata = data['metadata']
            total = metadata.get('total_examples', 0)
            train = len(data.get('train', []))
            val = len(data.get('validation', []))
            test = len(data.get('test', []))
            
            return {
                'format': 'consolidated_json',
                'total_examples': total,
                'train_examples': train,
                'val_examples': val,
                'test_examples': test,
                'categories': metadata.get('categories', []),
                'version': metadata.get('version', 'unknown')
            }
        elif isinstance(data, list):
            # Direct list format
            return {
                'format': 'list_json',
                'total_examples': len(data),
                'train_examples': len(data),
                'val_examples': 0,
                'test_examples': 0
            }
        else:
            return {'format': 'unknown', 'error': 'Unrecognized format'}
            
    except Exception as e:
        return {'format': 'error', 'error': str(e)}

def analyze_jsonl_file(file_path):
    """Analyze a JSONL file."""
    try:
        count = 0
        with open(file_path, 'r') as f:
            for line in f:
                if line.strip():
                    count += 1
        return {'format': 'jsonl', 'examples': count}
    except Exception as e:
        return {'format': 'error', 'error': str(e)}

def main():
    """Analyze all training data."""
    print("🔍 Training Data Analysis")
    print("=" * 60)
    
    datasets_dir = "training/datasets"
    
    # Analyze JSON datasets (main consolidated files)
    json_files = [f for f in os.listdir(datasets_dir) if f.endswith('.json')]
    jsonl_files = [f for f in os.listdir(datasets_dir) if f.endswith('.jsonl')]
    
    print(f"📊 Found {len(json_files)} JSON datasets and {len(jsonl_files)} JSONL files\n")
    
    total_examples = 0
    
    print("📚 Main Dataset Files (JSON):")
    print("-" * 40)
    for json_file in sorted(json_files):
        file_path = os.path.join(datasets_dir, json_file)
        analysis = analyze_dataset(file_path)
        
        if analysis['format'] == 'consolidated_json':
            print(f"✅ {json_file}")
            print(f"   Total examples: {analysis['total_examples']}")
            print(f"   Train/Val/Test: {analysis['train_examples']}/{analysis['val_examples']}/{analysis['test_examples']}")
            print(f"   Version: {analysis['version']}")
            print(f"   Categories: {', '.join(analysis['categories'][:3])}{'...' if len(analysis['categories']) > 3 else ''}")
            total_examples += analysis['total_examples']
        else:
            print(f"⚠️  {json_file}: {analysis.get('error', 'Unknown format')}")
        print()
    
    print("📄 JSONL Split Files:")
    print("-" * 40)
    jsonl_totals = {'train': 0, 'validation': 0, 'test': 0}
    
    for jsonl_file in sorted(jsonl_files):
        file_path = os.path.join(datasets_dir, jsonl_file)
        analysis = analyze_jsonl_file(file_path)
        
        if analysis['format'] == 'jsonl':
            split_type = 'unknown'
            if 'train' in jsonl_file:
                split_type = 'train'
                jsonl_totals['train'] += analysis['examples']
            elif 'validation' in jsonl_file:
                split_type = 'validation' 
                jsonl_totals['validation'] += analysis['examples']
            elif 'test' in jsonl_file:
                split_type = 'test'
                jsonl_totals['test'] += analysis['examples']
                
            print(f"   {jsonl_file}: {analysis['examples']} examples ({split_type})")
        else:
            print(f"   ❌ {jsonl_file}: {analysis.get('error')}")
    
    print(f"\n📊 JSONL Totals: Train={jsonl_totals['train']}, Val={jsonl_totals['validation']}, Test={jsonl_totals['test']}")
    
    print(f"\n🎯 Summary:")
    print(f"   JSON datasets: {total_examples} total examples")
    print(f"   JSONL files: {sum(jsonl_totals.values())} total examples")
    
    # Check for consolidation
    if sum(jsonl_totals.values()) == total_examples:
        print(f"✅ JSONL files are exports of JSON datasets (no additional data)")
    else:
        print(f"⚠️  JSONL files may contain different data than JSON datasets")
    
    print(f"\n💡 Recommendation:")
    if total_examples > 0:
        largest_dataset = max(json_files, key=lambda f: analyze_dataset(os.path.join(datasets_dir, f))['total_examples'])
        largest_analysis = analyze_dataset(os.path.join(datasets_dir, largest_dataset))
        print(f"   Use: training/datasets/{largest_dataset}")
        print(f"   Contains: {largest_analysis['total_examples']} examples")
        print(f"   Quality: {'Validated' if 'corrected' in largest_dataset else 'Comprehensive'}")
    else:
        print(f"   No valid datasets found!")

if __name__ == "__main__":
    main()