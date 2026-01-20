#!/usr/bin/env python3
"""
Simple ByteLogic Dataset Generator
=================================

Generates ByteLogic training examples without complex imports.
"""

import json
import os
import random

def generate_family_examples():
    """Generate family relationship examples."""
    examples = []
    
    # Name variations
    name_sets = [
        ["alice", "bob", "charlie", "david", "eve"],
        ["john", "mary", "peter", "susan", "tom"],
        ["anna", "ben", "clara", "dan", "ella"],
        ["alex", "beth", "chris", "diana", "eric"]
    ]
    
    for i, names in enumerate(name_sets):
        a, b, c, d, e = names
        
        # Basic parent query
        examples.append({
            "id": f"family_basic_{i}",
            "category": "logic_programming",
            "subcategory": "family_relationships",
            "difficulty": "beginner",
            "input": f"Who are {a.capitalize()}'s children?",
            "output": f"I'll check {a.capitalize()}'s children: <computation>\nREL parent\nFACT parent {a} {b}\nFACT parent {a} {c}\nSOLVE\nQUERY parent {a} ?\n</computation> → {b.capitalize()} and {c.capitalize()} are {a.capitalize()}'s children.",
            "bytelogic_code": f"REL parent\nFACT parent {a} {b}\nFACT parent {a} {c}\nSOLVE\nQUERY parent {a} ?",
            "expected_result": [b, c]
        })
        
        # Grandparent relationships
        examples.append({
            "id": f"family_grandparent_{i}",
            "category": "logic_programming", 
            "subcategory": "family_relationships",
            "difficulty": "intermediate",
            "input": f"Who are {a.capitalize()}'s grandchildren?",
            "output": f"I'll find {a.capitalize()}'s grandchildren: <computation>\nREL parent\nREL grandparent\nFACT parent {a} {b}\nFACT parent {a} {c}\nFACT parent {b} {d}\nFACT parent {c} {e}\nRULE grandparent: SCAN parent MATCH $0, JOIN parent $0, EMIT grandparent $1 $2\nSOLVE\nQUERY grandparent {a} ?\n</computation> → {d.capitalize()} and {e.capitalize()} are {a.capitalize()}'s grandchildren.",
            "bytelogic_code": f"REL parent\nREL grandparent\nFACT parent {a} {b}\nFACT parent {a} {c}\nFACT parent {b} {d}\nFACT parent {c} {e}\nRULE grandparent: SCAN parent MATCH $0, JOIN parent $0, EMIT grandparent $1 $2\nSOLVE\nQUERY grandparent {a} ?",
            "expected_result": [d, e]
        })
    
    return examples

def generate_math_examples():
    """Generate mathematical computation examples."""
    examples = []
    
    # Percentage calculations
    test_cases = [
        (200, 20, 40), (300, 25, 75), (150, 30, 45), (400, 15, 60), (500, 12, 60)
    ]
    
    for i, (value, percent, result) in enumerate(test_cases):
        examples.append({
            "id": f"math_percentage_{i}",
            "category": "mathematical_computation",
            "subcategory": "percentage_calculations", 
            "difficulty": "beginner",
            "input": f"What's {percent}% of {value}?",
            "output": f"I'll calculate the percentage: <computation>\nCALC percentage\n  INPUT $value $percent\n  LET $decimal = $percent / 100\n  RESULT $value * $decimal\nEND\nRESULT CALC percentage({value}, {percent})\n</computation> → {result}",
            "bytelogic_code": f"CALC percentage\n  INPUT $value $percent\n  LET $decimal = $percent / 100\n  RESULT $value * $decimal\nEND\nRESULT CALC percentage({value}, {percent})",
            "expected_result": [result]
        })
    
    # Fibonacci numbers
    fib_cases = [(5, 5), (7, 13), (8, 21), (9, 34), (10, 55)]
    
    for i, (n, fib_n) in enumerate(fib_cases):
        examples.append({
            "id": f"math_fibonacci_{i}",
            "category": "mathematical_computation",
            "subcategory": "sequences",
            "difficulty": "intermediate", 
            "input": f"What's the {n}th Fibonacci number?",
            "output": f"I'll calculate the {n}th Fibonacci number: <computation>\nCALC fibonacci\n  INPUT $n\n  IF $n <= 1 THEN\n    RESULT $n\n  ELSE\n    LET $a = CALC fibonacci($n - 1)\n    LET $b = CALC fibonacci($n - 2)\n    RESULT $a + $b\n  END\nEND\nRESULT CALC fibonacci({n})\n</computation> → {fib_n}",
            "bytelogic_code": f"CALC fibonacci\n  INPUT $n\n  IF $n <= 1 THEN\n    RESULT $n\n  ELSE\n    LET $a = CALC fibonacci($n - 1)\n    LET $b = CALC fibonacci($n - 2)\n    RESULT $a + $b\n  END\nEND\nRESULT CALC fibonacci({n})",
            "expected_result": [fib_n]
        })
    
    return examples

def generate_string_examples():
    """Generate string processing examples."""
    examples = []
    
    # Character counting
    test_words = [
        ("strawberry", "r", 3),
        ("programming", "m", 2), 
        ("hello", "l", 2),
        ("computer", "e", 1),
        ("algorithm", "i", 1)
    ]
    
    for i, (word, char, count) in enumerate(test_words):
        examples.append({
            "id": f"string_count_{i}",
            "category": "string_processing",
            "subcategory": "character_counting",
            "difficulty": "intermediate",
            "input": f"How many '{char}' characters are in '{word}'?",
            "output": f"I'll count the '{char}' characters: <computation>\nCALC count_char\n  INPUT $word $char\n  LET $count = 0\n  FOR $i IN RANGE(0, LENGTH($word))\n    LET $letter = CHAR_AT($word, $i)\n    IF $letter == $char THEN\n      LET $count = $count + 1\n    END\n  END\n  RESULT $count\nEND\nRESULT CALC count_char(\"{word}\", \"{char}\")\n</computation> → {count}",
            "bytelogic_code": f"CALC count_char\n  INPUT $word $char\n  LET $count = 0\n  FOR $i IN RANGE(0, LENGTH($word))\n    LET $letter = CHAR_AT($word, $i)\n    IF $letter == $char THEN\n      LET $count = $count + 1\n    END\n  END\n  RESULT $count\nEND\nRESULT CALC count_char(\"{word}\", \"{char}\")",
            "expected_result": [count]
        })
    
    return examples

def generate_graph_examples():
    """Generate graph algorithm examples."""
    examples = []
    
    # Graph reachability with different node configurations using atoms
    graph_configs = [
        {
            "edges": [("start", "middle"), ("start", "branch"), ("middle", "target"), ("branch", "target"), ("target", "end")],
            "query": ("start", "end"),
            "result": 1,
            "description": "Can you reach the end node from the start node?"
        },
        {
            "edges": [("a", "b"), ("b", "c"), ("c", "d"), ("d", "e")],
            "query": ("a", "e"),
            "result": 1,
            "description": "Is node e reachable from node a?"
        },
        {
            "edges": [("x", "y"), ("y", "z"), ("p", "q")],
            "query": ("x", "q"),
            "result": 0,
            "description": "Can you reach node q from node x?"
        }
    ]
    
    for i, config in enumerate(graph_configs):
        edges_facts = "\n".join([f"FACT edge {a} {b}" for a, b in config["edges"]])
        
        examples.append({
            "id": f"graph_reach_{i}",
            "category": "logic_programming",
            "subcategory": "graph_algorithms",
            "difficulty": "intermediate",
            "input": config["description"],
            "output": f"I'll check graph reachability: <computation>\nREL edge\nREL reachable\n{edges_facts}\nRULE reachable: SCAN edge, EMIT reachable $0 $1\nRULE reachable: SCAN edge, JOIN reachable $1, EMIT reachable $0 $2\nSOLVE\nQUERY reachable {config['query'][0]} {config['query'][1]}\n</computation> → {'Yes' if config['result'] else 'No'}, {'you can' if config['result'] else 'you cannot'} reach node {config['query'][1]} from node {config['query'][0]}.",
            "bytelogic_code": f"REL edge\nREL reachable\n{edges_facts}\nRULE reachable: SCAN edge, EMIT reachable $0 $1\nRULE reachable: SCAN edge, JOIN reachable $1, EMIT reachable $0 $2\nSOLVE\nQUERY reachable {config['query'][0]} {config['query'][1]}",
            "expected_result": [config["result"]]
        })
    
    return examples

def generate_dataset():
    """Generate complete training dataset."""
    print("📚 Generating ByteLogic training examples...")
    
    all_examples = []
    
    # Generate different categories
    categories = [
        ("Family Relationships", generate_family_examples()),
        ("Mathematical Computation", generate_math_examples()),
        ("String Processing", generate_string_examples()), 
        ("Graph Algorithms", generate_graph_examples())
    ]
    
    for category_name, examples in categories:
        print(f"  ✅ Generated {len(examples)} {category_name} examples")
        all_examples.extend(examples)
    
    print(f"  📊 Total examples: {len(all_examples)}")
    
    # Split dataset
    random.shuffle(all_examples)
    total = len(all_examples)
    
    train_end = int(total * 0.8)
    val_end = train_end + int(total * 0.1)
    
    train_examples = all_examples[:train_end]
    val_examples = all_examples[train_end:val_end]
    test_examples = all_examples[val_end:]
    
    # Create dataset
    dataset = {
        "metadata": {
            "version": "1.0",
            "generator": "Simple ByteLogic Dataset Generator",
            "total_examples": total,
            "train_examples": len(train_examples),
            "val_examples": len(val_examples), 
            "test_examples": len(test_examples),
            "categories": list(set(ex["category"] for ex in all_examples)),
            "difficulty_levels": list(set(ex["difficulty"] for ex in all_examples))
        },
        "train": [{"input": ex["input"], "output": ex["output"], "metadata": ex} for ex in train_examples],
        "validation": [{"input": ex["input"], "output": ex["output"], "metadata": ex} for ex in val_examples],
        "test": [{"input": ex["input"], "output": ex["output"], "metadata": ex} for ex in test_examples]
    }
    
    return dataset

def main():
    """Generate and save the dataset."""
    print("🚀 ByteLogic Dataset Generation")
    print("=" * 50)
    
    # Generate dataset
    dataset = generate_dataset()
    
    # Create output directory
    os.makedirs("training/datasets", exist_ok=True)
    
    # Save full dataset
    output_file = "training/datasets/bytelogic_dataset.json"
    with open(output_file, 'w') as f:
        json.dump(dataset, f, indent=2)
    
    print(f"\n💾 Dataset saved to {output_file}")
    
    # Save JSONL files
    for split in ["train", "validation", "test"]:
        jsonl_file = f"training/datasets/bytelogic_{split}.jsonl"
        with open(jsonl_file, 'w') as f:
            for example in dataset[split]:
                f.write(json.dumps(example) + '\n')
        print(f"   📄 {split.capitalize()}: {jsonl_file} ({len(dataset[split])} examples)")
    
    # Show sample
    print(f"\n📝 Sample Training Examples:")
    print("=" * 60)
    
    for i, example in enumerate(dataset["train"][:3]):
        print(f"\nExample {i+1} ({example['metadata']['category']}):")
        print(f"Input: {example['input']}")
        print(f"Output: {example['output'][:100]}...")
    
    # Show statistics
    metadata = dataset["metadata"]
    print(f"\n📊 Dataset Statistics:")
    print(f"   Total: {metadata['total_examples']} examples")
    print(f"   Training: {metadata['train_examples']} examples")  
    print(f"   Validation: {metadata['val_examples']} examples")
    print(f"   Test: {metadata['test_examples']} examples")
    print(f"   Categories: {', '.join(metadata['categories'])}")
    print(f"   Difficulty: {', '.join(metadata['difficulty_levels'])}")
    
    print(f"\n🎉 ByteLogic training dataset generation complete!")
    return True

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)