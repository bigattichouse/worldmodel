#!/usr/bin/env python3
"""
Extract standard syntax examples from comprehensive dataset
"""
import json
import re

def extract_standard_syntax_examples():
    """Extract only examples that use standard ByteLogic syntax."""
    
    # Load comprehensive dataset
    with open("training/datasets/comprehensive_bytelogic_dataset_with_natural_language.json", 'r', encoding='utf-8') as f:
        comp_data = json.load(f)
    
    # Load current dataset to compare with
    with open("training/datasets/complete_bytelogic_dataset.json", 'r', encoding='utf-8') as f:
        current_data = json.load(f)
    
    print(f"Comprehensive dataset has {len(comp_data.get('train', []))} training examples")
    print(f"Current dataset has {len(current_data.get('train', []))} training examples")
    
    # Extract examples that use only standard syntax
    standard_examples = []
    extended_examples = []
    
    for example in comp_data.get('train', []):
        output_text = example.get('output', '')
        
        # Extract computation block
        comp_blocks = re.findall(r'<computation>(.*?)</computation>', output_text, re.DOTALL)
        
        if not comp_blocks:
            # If no computation block, it might not be for ByteLogic execution
            extended_examples.append(example)
            continue
            
        comp_code = comp_blocks[0]
        
        # Check for extended syntax elements that are NOT supported by current compiler
        has_extended = any(keyword in comp_code for keyword in 
                         ['CALC', 'INPUT', 'RESULT', 'FOR ', 'FOR$', 'IF ', 'THEN', 'ELSE', 'END', 'LET ', 'POW(', 'WHILE', 'FOR EACH', 'MAP', 'FILTER'])
        
        # Also check for standard syntax elements to ensure it's not just text without ByteLogic
        has_standard = any(keyword in comp_code for keyword in ['REL ', 'FACT ', 'RULE ', 'SOLVE', 'QUERY '])
        
        if has_extended:
            extended_examples.append(example)
        elif has_standard:
            # Double check that it's truly standard syntax by checking for proper REL/FACT/RULE/QUERY structure
            lines = [line.strip() for line in comp_code.split('\n') if line.strip()]
            standard_elements = [line for line in lines if any(line.startswith(kw) for kw in ['REL', 'FACT', 'RULE', 'SOLVE', 'QUERY'])]
            if len(standard_elements) > 0:
                standard_examples.append(example)
            else:
                extended_examples.append(example)
        else:
            # Doesn't seem to have proper ByteLogic syntax
            extended_examples.append(example)
    
    print(f"Standard syntax examples found: {len(standard_examples)}")
    print(f"Extended syntax examples found: {len(extended_examples)}")
    
    # Create combined dataset
    all_train_examples = current_data.get('train', []) + standard_examples
    
    # Create expanded dataset with same structure as original
    expanded_dataset = {
        "metadata": {
            "version": "2.2-expanded",
            "generator": "Merged Original + Standard Syntax from Comprehensive",
            "total_examples": len(all_train_examples) + 110 + 110,  # train + val + test
            "train_examples": len(all_train_examples),
            "val_examples": current_data["metadata"]["val_examples"],  # Keep original validation
            "test_examples": current_data["metadata"]["test_examples"],  # Keep original test
            "features": current_data["metadata"]["features"] + ["expanded_variety"],
            "compatibility": "ByteLogic 2.0 Standard Syntax + Error Handling",
            "original_dataset": "training/datasets/complete_bytelogic_dataset.json",
            "added_from": "training/datasets/comprehensive_bytelogic_dataset_with_natural_language.json"
        },
        "train": all_train_examples,
        "val": current_data.get("val", []),  # Keep original validation
        "test": current_data.get("test", [])  # Keep original test
    }
    
    print(f"Expanded dataset will have: {len(all_train_examples)} training examples")
    print(f"Total examples after combining: {expanded_dataset['metadata']['total_examples']}")
    
    # Save the expanded dataset
    output_path = "training/datasets/expanded_bytelogic_dataset.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(expanded_dataset, f, ensure_ascii=False, indent=2)
    
    print(f"Expanded dataset saved to: {output_path}")
    
    # Show sample of added examples
    print(f"\\nSample of added standard syntax examples:")
    for i, ex in enumerate(standard_examples[:3]):
        print(f"\\n{i+1}. Input: {ex['input']}")
        print(f"   Category: {ex['metadata']['category']}")
        print(f"   Difficulty: {ex['metadata']['difficulty']}")
        # Show computation block preview
        output = ex['output']
        comp_matches = re.findall(r'<computation>(.*?)</computation>', output, re.DOTALL)
        if comp_matches:
            preview = comp_matches[0][:150] + "..." if len(comp_matches[0]) > 150 else comp_matches[0]
            print(f"   ByteLogic: {preview}")
    
    return len(standard_examples)

if __name__ == "__main__":
    count = extract_standard_syntax_examples()
    print(f"\\nSuccessfully extracted and added {count} standard syntax examples to create expanded dataset!")