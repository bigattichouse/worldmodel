#!/usr/bin/env python3
"""
Corrected ByteLogic Dataset Generator
====================================

Generates training examples using only ByteLogic 2.0 features that are actually supported.
Based on the working examples from bytelogic/examples/
"""

import json
import random
import os
from typing import List, Dict, Any


class CorrectedByteLogicGenerator:
    """Generator that only uses supported ByteLogic 2.0 features."""
    
    def __init__(self):
        # Use atoms (symbols) instead of strings for compatibility
        self.atoms = [
            "alice", "bob", "charlie", "david", "eve", "frank", "grace", 
            "henry", "iris", "jack", "kate", "luke", "nina", "oscar", "paul",
            "start", "middle", "end", "node_a", "node_b", "node_c", "node_d",
            "red", "blue", "green", "yellow", "large", "small", "fast", "slow"
        ]
        
        self.relations = [
            "parent", "child", "sibling", "grandparent", "ancestor", "descendant",
            "friend", "knows", "likes", "teaches", "manages", "works_for",
            "edge", "path", "reachable", "connected", "neighbor",
            "color", "size", "speed", "type", "category", "belongs_to"
        ]
    
    def generate_basic_family_examples(self, count: int) -> List[Dict]:
        """Generate basic family relationship examples."""
        examples = []
        
        for i in range(count):
            # Pick random atoms
            parent = random.choice(self.atoms[:10])
            child1 = random.choice(self.atoms[10:15])
            child2 = random.choice(self.atoms[15:20])
            
            # Simple parent-child query
            code = f"""REL parent
FACT parent {parent} {child1}
FACT parent {parent} {child2}
SOLVE
QUERY parent {parent} ?"""
            
            examples.append({
                "id": f"family_basic_{i}",
                "category": "basic_logic",
                "subcategory": "family_relationships",
                "difficulty": "beginner",
                "input": f"Who are {parent}'s children?",
                "output": f"I'll find {parent}'s children: <computation>\n{code}\n</computation> → {child1} and {child2}",
                "bytelogic_code": code,
                "expected_result": [child1, child2]
            })
        
        return examples
    
    def generate_grandparent_examples(self, count: int) -> List[Dict]:
        """Generate grandparent relationship examples.""" 
        examples = []
        
        for i in range(count):
            grandparent = random.choice(self.atoms[:5])
            parent1 = random.choice(self.atoms[5:10])
            parent2 = random.choice(self.atoms[10:12])
            grandchild1 = random.choice(self.atoms[12:15])
            grandchild2 = random.choice(self.atoms[15:18])
            
            code = f"""REL parent
REL grandparent
FACT parent {grandparent} {parent1}
FACT parent {grandparent} {parent2}
FACT parent {parent1} {grandchild1}
FACT parent {parent2} {grandchild2}
RULE grandparent: SCAN parent MATCH $0, JOIN parent $0, EMIT grandparent $1 $2
SOLVE
QUERY grandparent {grandparent} ?"""
            
            examples.append({
                "id": f"grandparent_{i}",
                "category": "basic_logic",
                "subcategory": "family_relationships",
                "difficulty": "intermediate",
                "input": f"Who are {grandparent}'s grandchildren?",
                "output": f"I'll find {grandparent}'s grandchildren: <computation>\n{code}\n</computation> → {grandchild1} and {grandchild2}",
                "bytelogic_code": code,
                "expected_result": [grandchild1, grandchild2]
            })
        
        return examples
    
    def generate_symmetric_examples(self, count: int) -> List[Dict]:
        """Generate symmetric relationship examples."""
        examples = []
        
        for i in range(count):
            person1 = random.choice(self.atoms[:8])
            person2 = random.choice(self.atoms[8:16])
            person3 = random.choice(self.atoms[16:20])
            
            code = f"""REL friend_directed
REL friend
FACT friend_directed {person1} {person2}
FACT friend_directed {person2} {person3}
RULE friend: SCAN friend_directed, EMIT friend $0 $1
RULE friend: SCAN friend_directed, EMIT friend $1 $0
SOLVE
QUERY friend {person3} ?"""
            
            examples.append({
                "id": f"symmetric_{i}",
                "category": "basic_logic",
                "subcategory": "symmetric_relations",
                "difficulty": "intermediate",
                "input": f"Who are {person3}'s friends?",
                "output": f"I'll find {person3}'s friends: <computation>\n{code}\n</computation> → {person2}",
                "bytelogic_code": code,
                "expected_result": [person2]
            })
        
        return examples
    
    def generate_transitive_examples(self, count: int) -> List[Dict]:
        """Generate transitive closure examples."""
        examples = []
        
        for i in range(count):
            node_a = random.choice(self.atoms[-10:-7])
            node_b = random.choice(self.atoms[-7:-4])  
            node_c = random.choice(self.atoms[-4:-1])
            
            code = f"""REL edge
REL reachable
FACT edge {node_a} {node_b}
FACT edge {node_b} {node_c}
RULE reachable: SCAN edge, EMIT reachable $0 $1
RULE reachable: SCAN edge, JOIN reachable $1, EMIT reachable $0 $2
SOLVE
QUERY reachable {node_a} {node_c}"""
            
            examples.append({
                "id": f"transitive_{i}",
                "category": "graph_algorithms",
                "subcategory": "reachability",
                "difficulty": "intermediate",
                "input": f"Can you reach {node_c} from {node_a}?",
                "output": f"I'll check reachability: <computation>\n{code}\n</computation> → Yes, {node_c} is reachable from {node_a}",
                "bytelogic_code": code,
                "expected_result": [1]
            })
        
        return examples
    
    def generate_classification_examples(self, count: int) -> List[Dict]:
        """Generate classification hierarchy examples."""
        examples = []
        
        for i in range(count):
            item = random.choice(self.atoms[:5])
            category1 = random.choice(self.atoms[5:10])
            category2 = random.choice(self.atoms[10:15])
            property1 = random.choice(self.atoms[15:20])
            property2 = random.choice(self.atoms[20:25])
            
            code = f"""REL isa
REL has_property
FACT isa {item} {category1}
FACT isa {category1} {category2}
FACT has_property {category1} {property1}
FACT has_property {category2} {property2}
RULE isa: SCAN isa, JOIN isa $1, EMIT isa $0 $2
RULE has_property: SCAN isa, JOIN has_property $1, EMIT has_property $0 $2
SOLVE
QUERY has_property {item} ?"""
            
            examples.append({
                "id": f"classification_{i}",
                "category": "basic_logic",
                "subcategory": "classification",
                "difficulty": "advanced",
                "input": f"What properties does {item} have?",
                "output": f"I'll find {item}'s properties: <computation>\n{code}\n</computation> → {property1} and {property2}",
                "bytelogic_code": code,
                "expected_result": [property1, property2]
            })
        
        return examples
    
    def generate_simple_queries(self, count: int) -> List[Dict]:
        """Generate simple fact queries."""
        examples = []
        
        for i in range(count):
            relation = random.choice(self.relations[:5])
            entity1 = random.choice(self.atoms[:10])
            entity2 = random.choice(self.atoms[10:20])
            entity3 = random.choice(self.atoms[20:25])
            
            code = f"""REL {relation}
FACT {relation} {entity1} {entity2}
FACT {relation} {entity1} {entity3}
SOLVE
QUERY {relation} {entity1} ?"""
            
            examples.append({
                "id": f"simple_{i}",
                "category": "basic_logic",
                "subcategory": "simple_facts",
                "difficulty": "beginner",
                "input": f"What does {entity1} {relation}?",
                "output": f"I'll check what {entity1} {relation}: <computation>\n{code}\n</computation> → {entity2} and {entity3}",
                "bytelogic_code": code,
                "expected_result": [entity2, entity3]
            })
        
        return examples
    
    def generate_multi_relation_examples(self, count: int) -> List[Dict]:
        """Generate examples with multiple relations."""
        examples = []
        
        for i in range(count):
            relation1 = random.choice(self.relations[:3])
            relation2 = random.choice(self.relations[3:6])
            entity1 = random.choice(self.atoms[:8])
            entity2 = random.choice(self.atoms[8:12])
            entity3 = random.choice(self.atoms[12:16])
            
            code = f"""REL {relation1}
REL {relation2}
FACT {relation1} {entity1} {entity2}
FACT {relation2} {entity2} {entity3}
SOLVE
QUERY {relation1} ? ?
QUERY {relation2} ? ?"""
            
            examples.append({
                "id": f"multi_rel_{i}",
                "category": "basic_logic",
                "subcategory": "multiple_relations",
                "difficulty": "intermediate",
                "input": f"What are all the {relation1} and {relation2} relationships?",
                "output": f"I'll find all relationships: <computation>\n{code}\n</computation> → {relation1}: {entity1}→{entity2}, {relation2}: {entity2}→{entity3}",
                "bytelogic_code": code,
                "expected_result": [(entity1, entity2), (entity2, entity3)]
            })
        
        return examples
    
    def generate_complex_graph_examples(self, count: int) -> List[Dict]:
        """Generate complex graph examples."""
        examples = []
        
        for i in range(count):
            # Create a more complex graph
            nodes = random.sample(self.atoms[-8:], 4)
            a, b, c, d = nodes
            
            code = f"""REL edge
REL path
FACT edge {a} {b}
FACT edge {b} {c}
FACT edge {c} {d}
FACT edge {a} {c}
RULE path: SCAN edge, EMIT path $0 $1
RULE path: SCAN edge, JOIN path $1, EMIT path $0 $2
SOLVE
QUERY path {a} ?"""
            
            examples.append({
                "id": f"complex_graph_{i}",
                "category": "graph_algorithms",
                "subcategory": "path_finding",
                "difficulty": "advanced",
                "input": f"What nodes can be reached from {a}?",
                "output": f"I'll find all reachable nodes: <computation>\n{code}\n</computation> → {b}, {c}, and {d}",
                "bytelogic_code": code,
                "expected_result": [b, c, d]
            })
        
        return examples
    
    def generate_corrected_dataset(self, total_examples: int = 1000) -> Dict:
        """Generate a corrected dataset with only supported features."""
        print("🚀 Generating corrected ByteLogic dataset...")
        
        all_examples = []
        
        # Distribution of example types
        generators = [
            ("Basic Family", self.generate_basic_family_examples, 200),
            ("Grandparent Relationships", self.generate_grandparent_examples, 150),
            ("Symmetric Relations", self.generate_symmetric_examples, 150),
            ("Transitive Closure", self.generate_transitive_examples, 150),
            ("Classification", self.generate_classification_examples, 100),
            ("Simple Queries", self.generate_simple_queries, 150),
            ("Multiple Relations", self.generate_multi_relation_examples, 100)
        ]
        
        for name, generator, count in generators:
            print(f"  📝 Generating {count} {name} examples...")
            examples = generator(count)
            all_examples.extend(examples)
            print(f"    ✅ Generated {len(examples)} examples")
        
        # Shuffle and split
        random.shuffle(all_examples)
        total = len(all_examples)
        
        train_end = int(total * 0.8)
        val_end = train_end + int(total * 0.1)
        
        train_examples = all_examples[:train_end]
        val_examples = all_examples[train_end:val_end]
        test_examples = all_examples[val_end:]
        
        # Create dataset structure
        dataset = {
            "metadata": {
                "version": "2.0-corrected",
                "generator": "Corrected ByteLogic Dataset Generator",
                "total_examples": total,
                "train_examples": len(train_examples),
                "val_examples": len(val_examples),
                "test_examples": len(test_examples),
                "features": ["relations", "facts", "rules", "queries", "basic_logic_programming"],
                "compatibility": "ByteLogic 2.0 only"
            },
            "train": [self._format_example(ex) for ex in train_examples],
            "validation": [self._format_example(ex) for ex in val_examples],
            "test": [self._format_example(ex) for ex in test_examples]
        }
        
        print(f"\n📊 Corrected Dataset Statistics:")
        print(f"   Total examples: {total}")
        print(f"   Training: {len(train_examples)}")
        print(f"   Validation: {len(val_examples)}")
        print(f"   Test: {len(test_examples)}")
        
        return dataset
    
    def _format_example(self, example: Dict) -> Dict:
        """Format example for training."""
        return {
            "input": example["input"],
            "output": example["output"],
            "metadata": example
        }


def main():
    """Generate corrected ByteLogic dataset."""
    print("🚀 Corrected ByteLogic Dataset Generation")
    print("=" * 60)
    
    generator = CorrectedByteLogicGenerator()
    
    # Generate dataset
    dataset = generator.generate_corrected_dataset(1000)
    
    # Save main dataset
    os.makedirs("training/datasets", exist_ok=True)
    output_file = "training/datasets/corrected_bytelogic_dataset.json"
    
    with open(output_file, 'w') as f:
        json.dump(dataset, f, indent=2)
    
    print(f"\n💾 Corrected dataset saved to {output_file}")
    
    # Save JSONL files
    for split in ["train", "validation", "test"]:
        jsonl_file = f"training/datasets/bytelogic_{split}_corrected.jsonl"
        with open(jsonl_file, 'w') as f:
            for example in dataset[split]:
                f.write(json.dumps(example) + '\n')
        print(f"   📄 {split.capitalize()}: {jsonl_file} ({len(dataset[split])} examples)")
    
    # Show sample examples
    print(f"\n📝 Sample Training Examples:")
    print("=" * 60)
    
    for i, example in enumerate(dataset["train"][:3]):
        print(f"\nExample {i+1} ({example['metadata']['category']}):")
        print(f"Input: {example['input']}")
        print(f"Code Preview:")
        code_lines = example['metadata']['bytelogic_code'].split('\n')
        for line in code_lines[:5]:
            print(f"  {line}")
        if len(code_lines) > 5:
            print(f"  ... ({len(code_lines)-5} more lines)")
    
    print(f"\n🎉 Corrected ByteLogic dataset generation complete!")
    print(f"   Compatible with ByteLogic 2.0")
    print(f"   Ready for syntax validation and training")
    return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)