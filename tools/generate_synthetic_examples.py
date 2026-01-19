#!/usr/bin/env python3
"""
Generate Synthetic Examples for ByteLogic Dataset
=================================================

Create additional mathematical and logical examples to expand the dataset.
"""

import json
import random
from pathlib import Path

def generate_mathematical_examples():
    """Generate mathematical calculation examples."""
    examples = []
    
    # Basic arithmetic
    for i in range(100):
        a = random.randint(1, 100)
        b = random.randint(1, 50)
        operation = random.choice(['add', 'subtract', 'multiply', 'divide'])
        
        if operation == 'add':
            question = f"What is {a} + {b}?"
            result = a + b
            byte_logic = f"""REL calculation
FACT calculation add {a}
FACT calculation add2 {b}
SOLVE
QUERY calculation ? ?
"""
        elif operation == 'subtract':
            # Ensure positive result
            if a < b:
                a, b = b, a
            question = f"What is {a} - {b}?"
            result = a - b
            byte_logic = f"""REL calculation
FACT calculation minus {a}
FACT calculation subtrahend {b}
SOLVE
QUERY calculation ? ?
"""
        elif operation == 'multiply':
            question = f"What is {a} * {b}?"
            result = a * b
            byte_logic = f"""REL operation
REL number
FACT operation multiply 1
FACT number operand1 {a}
FACT number operand2 {b}
SOLVE
QUERY operation ? ?
"""
        else:  # divide
            # Make sure division works evenly
            b = random.randint(1, min(10, b))
            result = random.randint(1, 20)
            a = result * b  # Ensure integer division
            question = f"What is {a} / {b}?"
            byte_logic = f"""REL division
REL operands
FACT division dividend {a}
FACT operands divisor {b}
SOLVE
QUERY division ? ?
"""
        
        example = {
            "input": question,
            "output": f"I'll calculate: <computation>\\n{byte_logic.strip()}\\n</computation> → {result}",
            "metadata": {
                "id": f"math_gen_{len(examples)}",
                "category": "mathematical_computation",
                "subcategory": "basic_arithmetic",
                "difficulty": "beginner"
            }
        }
        examples.append(example)
    
    # More complex calculations
    for i in range(50):
        a, b, c = random.randint(2, 20), random.randint(2, 20), random.randint(1, 10)
        operation = random.choice(['sum_three', 'product_three', 'mixed'])
        
        if operation == 'sum_three':
            question = f"What is {a} + {b} + {c}?"
            result = a + b + c
            byte_logic = f"""REL calculation
REL values
FACT calculation add1 {a}
FACT values add2 {b}
FACT values add3 {c}
SOLVE
QUERY calculation ? ?
"""
        elif operation == 'product_three':
            question = f"What is {a} * {b} * {c}?"
            result = a * b * c
            byte_logic = f"""REL multiplication
REL factors
FACT multiplication factor1 {a}
FACT factors factor2 {b}
FACT factors factor3 {c}
SOLVE
QUERY multiplication ? ?
"""
        else:  # mixed operation
            question = f"If I have {a} apples, buy {b} more, then eat {c}, how many remain?"
            result = a + b - c
            byte_logic = f"""REL apples
REL transactions
FACT apples initial {a}
FACT transactions bought {b}
FACT transactions eaten {c}
SOLVE
QUERY apples ? ?
"""
        
        example = {
            "input": question,
            "output": f"I'll calculate: <computation>\\n{byte_logic.strip()}\\n</computation> → {result}",
            "metadata": {
                "id": f"math_compound_{len(examples)}",
                "category": "mathematical_computation",
                "subcategory": "compound_operations",
                "difficulty": "intermediate"
            }
        }
        examples.append(example)
    
    return examples

def generate_logical_examples():
    """Generate logical reasoning examples."""
    examples = []
    
    # Family relations
    names = ['alice', 'bob', 'charlie', 'diana', 'eve', 'frank', 'grace', 'henry', 'irene', 'jack']
    
    for i in range(100):
        # Pick 3-4 different names
        selected_names = random.sample(names, 3)
        
        # Create family tree
        father, mother, child = selected_names[0], selected_names[1], selected_names[2]
        
        question_type = random.choice(['parent', 'child', 'sibling', 'grandparent'])
        
        if question_type == 'parent':
            question = f"Who are {child}'s parents?"
            byte_logic = f"""REL parent
FACT parent {father} {child}
FACT parent {mother} {child}
SOLVE
QUERY parent ? {child}
"""
            output = f"The parents of {child} are: <computation>\\n{byte_logic.strip()}\\n</computation> → {father} and {mother}"
        elif question_type == 'child':
            question = f"Who are {father}'s children?"
            byte_logic = f"""REL parent
FACT parent {father} {child}
SOLVE
QUERY parent {father} ?
"""
            output = f"{father}'s children are: <computation>\\n{byte_logic.strip()}\\n</computation> → {child}"
        elif question_type == 'sibling':
            # Add another child to make siblings
            other_child = random.choice([name for name in names if name not in selected_names])
            question = f"Who are {child}'s siblings?"
            byte_logic = f"""REL parent
REL sibling
FACT parent {father} {child}
FACT parent {father} {other_child}
RULE sibling: SCAN parent MATCH $0, JOIN parent $0, EMIT sibling $1 $2
SOLVE
QUERY sibling {child} ?
"""
            output = f"{child}'s siblings are: <computation>\\n{byte_logic.strip()}\\n</computation> → {other_child}"
        else:  # grandparent
            grandparent = random.choice([name for name in names if name not in selected_names[:3]])
            question = f"Who are {child}'s grandparents?"
            byte_logic = f"""REL parent
REL grandparent
FACT parent {father} {child}
FACT parent {grandparent} {father}
RULE grandparent: SCAN parent MATCH $0, JOIN parent $0, EMIT grandparent $1 $2
SOLVE
QUERY grandparent {child} ?
"""
            output = f"{child}'s grandparents are: <computation>\\n{byte_logic.strip()}\\n</computation> → {grandparent}"
        
        example = {
            "input": question,
            "output": output,
            "metadata": {
                "id": f"logic_family_{len(examples)}",
                "category": "basic_logic",
                "subcategory": "family_relations",
                "difficulty": "intermediate"
            }
        }
        examples.append(example)
    
    # Friend/social relations
    for i in range(50):
        name1, name2, name3 = random.sample(names, 3)
        
        question_type = random.choice(['friend', 'common_friend', 'connection'])
        
        if question_type == 'friend':
            question = f"Who does {name1} know?"
            byte_logic = f"""REL knows
FACT knows {name1} {name2}
SOLVE
QUERY knows {name1} ?
"""
            output = f"{name1} knows: <computation>\\n{byte_logic.strip()}\\n</computation> → {name2}"
        elif question_type == 'common_friend':
            question = f"Do {name1} and {name2} share a friend?"
            byte_logic = f"""REL friend
REL common
FACT friend {name1} {name3}
FACT friend {name2} {name3}
RULE common: SCAN friend MATCH $0, JOIN friend $0, EMIT common $1 $2
SOLVE
QUERY common {name1} {name2}
"""
            output = f"Do {name1} and {name2} share a friend?: <computation>\\n{byte_logic.strip()}\\n</computation> → Yes, {name3}"
        else:  # connection
            question = f"Are {name1} and {name2} connected?"
            byte_logic = f"""REL connected
FACT connected {name1} {name2}
SOLVE
QUERY connected {name1} {name2} ?
"""
            output = f"Connection status: <computation>\\n{byte_logic.strip()}\\n</computation> → Yes"
        
        example = {
            "input": question,
            "output": output,
            "metadata": {
                "id": f"logic_social_{len(examples)}",
                "category": "basic_logic",
                "subcategory": "social_networks",
                "difficulty": "beginner"
            }
        }
        examples.append(example)
    
    return examples

def generate_graph_examples():
    """Generate graph theory examples."""
    examples = []
    
    for i in range(50):
        nodes = random.sample(['a', 'b', 'c', 'd', 'e', 'f'], random.randint(4, 6))
        
        # Create a connected graph
        connections = []
        for j in range(len(nodes)-1):
            connections.append((nodes[j], nodes[j+1]))
        
        # Add a few more random connections
        extra_edges = random.randint(1, 3)
        for _ in range(extra_edges):
            node1, node2 = random.sample(nodes, 2)
            if (node1, node2) not in connections and (node2, node1) not in connections:
                connections.append((node1, node2))
        
        start_node = nodes[0]
        end_node = nodes[-1]
        
        question = f"Can you reach {end_node} from {start_node}?"
        byte_logic_parts = ["REL edge", "REL reachable"]
        
        for src, dst in connections:
            byte_logic_parts.append(f"FACT edge {src} {dst}")
        
        byte_logic_parts.append("RULE reachable: SCAN edge, EMIT reachable $0 $1")
        byte_logic_parts.append("RULE reachable: SCAN edge, JOIN reachable $1, EMIT reachable $0 $2")
        byte_logic_parts.append("SOLVE")
        byte_logic_parts.append(f"QUERY reachable {start_node} {end_node}")
        
        byte_logic = "\\n".join(byte_logic_parts)
        
        # Simple path existence check (most generated graphs will be connected)
        output = f"Reachability check: <computation>\\n{byte_logic}\\n</computation> → Yes"
        
        example = {
            "input": question,
            "output": output,
            "metadata": {
                "id": f"graph_gen_{len(examples)}",
                "category": "graph_algorithms",
                "subcategory": "reachability",
                "difficulty": "intermediate"
            }
        }
        examples.append(example)
    
    return examples

def combine_and_save_datasets():
    """Combine generated examples with existing dataset."""
    
    # Load existing expanded dataset
    with open("training/datasets/expanded_bytelogic_dataset.json", 'r', encoding='utf-8') as f:
        existing_data = json.load(f)
    
    print(f"Existing dataset has {len(existing_data['train'])} training examples")
    
    # Generate new examples
    print("Generating mathematical examples...")
    math_examples = generate_mathematical_examples()
    print(f"Generated {len(math_examples)} mathematical examples")
    
    print("Generating logical examples...")
    logic_examples = generate_logical_examples()
    print(f"Generated {len(logic_examples)} logical examples")
    
    print("Generating graph examples...")
    graph_examples = generate_graph_examples()
    print(f"Generated {len(graph_examples)} graph examples")
    
    # Combine all examples
    all_new_examples = math_examples + logic_examples + graph_examples
    
    # Combine with existing examples
    combined_train_examples = existing_data['train'] + all_new_examples
    
    # Create new dataset with all examples
    combined_dataset = {
        "metadata": {
            "version": "2.3-comprehensive",
            "generator": "Original + Expanded + Synthetic Examples",
            "total_examples": len(combined_train_examples) + existing_data['metadata']['val_examples'] + existing_data['metadata']['test_examples'],
            "train_examples": len(combined_train_examples),
            "val_examples": existing_data['metadata']['val_examples'],
            "test_examples": existing_data['metadata']['test_examples'],
            "features": existing_data['metadata']['features'] + ["synthetic_generation", "additional_variety"],
            "compatibility": existing_data['metadata']['compatibility'],
            "original_dataset": existing_data['metadata']['original_dataset'],
            "added_synthetic": len(all_new_examples)
        },
        "train": combined_train_examples,
        "val": existing_data.get('val', []),
        "test": existing_data.get('test', [])
    }
    
    print(f"Combined dataset will have: {len(combined_train_examples)} training examples")
    print(f"Total examples: {combined_dataset['metadata']['total_examples']}")
    
    # Save the combined dataset
    output_path = "training/datasets/comprehensive_bytelogic_dataset.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(combined_dataset, f, ensure_ascii=False, indent=2)
    
    print(f"Comprehensive dataset saved to: {output_path}")
    
    return len(all_new_examples)

if __name__ == "__main__":
    print("Generating Synthetic Examples for ByteLogic Dataset Expansion")
    print("="*70)
    
    total_added = combine_and_save_datasets()
    
    print(f"\\nSUCCESS: Added {total_added} synthetic examples!")
    print(f"New comprehensive dataset includes:")
    print(f"- {len(generate_mathematical_examples())} mathematical calculation examples")
    print(f"- {len(generate_logical_examples())} logical reasoning examples") 
    print(f"- {len(generate_graph_examples())} graph theory examples")
    print(f"\\nUse --dataset training/datasets/comprehensive_bytelogic_dataset.json for training")