#!/usr/bin/env python3
"""
Comprehensive ByteLogic Dataset Generator
=========================================

Generates training examples covering all ByteLogic language features:
1. Core Logic Programming (Relations, Facts, Rules, Queries)
2. Mathematical Calculations (CALC blocks, expressions)
3. Loop Constructs (FOR-RANGE, FOR-WHILE, FOR-EACH)
4. String Processing (LENGTH, CHAR_AT)
5. Conditional Logic (IF/THEN/ELSE)
6. Complex Expressions (arithmetic, comparisons)
7. Hybrid Logic+Math scenarios
"""

import json
import random
import os
import itertools
from typing import List, Dict, Any, Tuple


class ByteLogicDatasetGenerator:
    """Comprehensive generator for ByteLogic training examples."""
    
    def __init__(self):
        self.categories = [
            "basic_logic",
            "mathematical_computation", 
            "string_processing",
            "loop_constructs",
            "conditional_logic",
            "hybrid_reasoning",
            "graph_algorithms",
            "family_relationships",
            "classification_hierarchies",
            "data_analysis"
        ]
        
        # Common name sets for variety
        self.name_sets = [
            ["alice", "bob", "charlie", "david", "eve"],
            ["john", "mary", "peter", "susan", "tom"],
            ["anna", "ben", "clara", "dan", "ella"],
            ["alex", "beth", "chris", "diana", "eric"],
            ["frank", "grace", "henry", "iris", "jack"],
            ["kate", "luke", "nina", "oscar", "paul"]
        ]
        
        # Words for string processing
        self.test_words = [
            "hello", "world", "strawberry", "programming", "computer",
            "algorithm", "function", "variable", "calculation", "structure",
            "artificial", "intelligence", "machine", "learning", "neural",
            "network", "database", "system", "software", "hardware"
        ]
        
    def generate_basic_logic_examples(self, count: int = 200) -> List[Dict]:
        """Generate basic logic programming examples."""
        examples = []
        
        for i in range(count):
            names = random.choice(self.name_sets)
            category_type = random.choice([
                "simple_facts", "parent_child", "friendships", 
                "symmetric_relations", "transitive_closure"
            ])
            
            if category_type == "simple_facts":
                examples.append(self._generate_simple_facts_example(names, i))
            elif category_type == "parent_child":
                examples.append(self._generate_parent_child_example(names, i))
            elif category_type == "friendships":
                examples.append(self._generate_friendship_example(names, i))
            elif category_type == "symmetric_relations":
                examples.append(self._generate_symmetric_example(names, i))
            elif category_type == "transitive_closure":
                examples.append(self._generate_transitive_example(names, i))
        
        return examples
    
    def generate_mathematical_examples(self, count: int = 150) -> List[Dict]:
        """Generate mathematical computation examples."""
        examples = []
        
        for i in range(count):
            math_type = random.choice([
                "basic_arithmetic", "percentage", "factorial", "fibonacci",
                "power_calculations", "trigonometry", "mathematical_functions", "natural_language_math"
            ])
            
            if math_type == "basic_arithmetic":
                examples.append(self._generate_basic_arithmetic_example(i))
            elif math_type == "percentage":
                examples.append(self._generate_percentage_example(i))
            elif math_type == "factorial":
                examples.append(self._generate_factorial_example(i))
            elif math_type == "fibonacci":
                examples.append(self._generate_fibonacci_example(i))
            elif math_type == "power_calculations":
                examples.append(self._generate_power_example(i))
            elif math_type == "trigonometry":
                examples.append(self._generate_trigonometry_example(i))
            elif math_type == "mathematical_functions":
                examples.append(self._generate_math_functions_example(i))
            elif math_type == "natural_language_math":
                examples.append(self._generate_natural_language_math_example(i))
        
        return examples

    def _generate_natural_language_math_example(self, idx: int) -> Dict:
        """Generate math examples from natural language questions."""
        question_type = random.choice([
            "half", "double", "add", "subtract", "multiply", "divide"
        ])
        
        a = random.randint(10, 100)
        b = random.randint(2, 10)
        
        if question_type == "half":
            a = a * 2 # Ensure it's even
            question = f"what's half of {a}"
            code = f"CALC half\nINPUT $0\nRESULT $0 / 2\nEND\nRESULT CALC half({a})"
            result = a // 2
        elif question_type == "double":
            question = f"what's double {a}"
            code = f"CALC double\nINPUT $0\nRESULT $0 * 2\nEND\nRESULT CALC double({a})"
            result = a * 2
        elif question_type == "add":
            question = f"what's {a} plus {b}"
            code = f"CALC add\nINPUT $0 $1\nRESULT $0 + $1\nEND\nRESULT CALC add({a}, {b})"
            result = a + b
        elif question_type == "subtract":
            question = f"what's {a} minus {b}"
            code = f"CALC subtract\nINPUT $0 $1\nRESULT $0 - $1\nEND\nRESULT CALC subtract({a}, {b})"
            result = a - b
        elif question_type == "multiply":
            question = f"what's {a} times {b}"
            code = f"CALC multiply\nINPUT $0 $1\nRESULT $0 * $1\nEND\nRESULT CALC multiply({a}, {b})"
            result = a * b
        else: # divide
            a = a * b # Ensure it's a clean division
            question = f"what's {a} divided by {b}"
            code = f"CALC divide\nINPUT $0 $1\nRESULT $0 / $1\nEND\nRESULT CALC divide({a}, {b})"
            result = a // b
            
        return {
            "id": f"natural_math_{idx}",
            "category": "mathematical_computation",
            "subcategory": "natural_language_math",
            "difficulty": "beginner",
            "input": question,
            "output": f"I'll calculate that: <computation>\n{code}\n</computation> → {result}",
            "bytelogic_code": code,
            "expected_result": [result]
        }
    
    def generate_string_processing_examples(self, count: int = 100) -> List[Dict]:
        """Generate string processing examples."""
        examples = []
        
        for i in range(count):
            string_type = random.choice([
                "character_counting", "vowel_counting", "word_analysis",
                "string_comparison", "pattern_detection"
            ])
            
            word = random.choice(self.test_words)
            
            if string_type == "character_counting":
                examples.append(self._generate_character_counting_example(word, i))
            elif string_type == "vowel_counting":
                examples.append(self._generate_vowel_counting_example(word, i))
            elif string_type == "word_analysis":
                examples.append(self._generate_word_analysis_example(word, i))
            elif string_type == "string_comparison":
                examples.append(self._generate_string_comparison_example(i))
            elif string_type == "pattern_detection":
                examples.append(self._generate_pattern_detection_example(word, i))
        
        return examples
    
    def generate_loop_examples(self, count: int = 120) -> List[Dict]:
        """Generate loop construct examples."""
        examples = []
        
        for i in range(count):
            loop_type = random.choice([
                "for_range", "for_while", "nested_loops", "loop_with_conditions",
                "accumulation_loops", "iteration_patterns"
            ])
            
            if loop_type == "for_range":
                examples.append(self._generate_for_range_example(i))
            elif loop_type == "for_while":
                examples.append(self._generate_for_while_example(i))
            elif loop_type == "nested_loops":
                examples.append(self._generate_nested_loops_example(i))
            elif loop_type == "loop_with_conditions":
                examples.append(self._generate_conditional_loop_example(i))
            elif loop_type == "accumulation_loops":
                examples.append(self._generate_accumulation_loop_example(i))
            elif loop_type == "iteration_patterns":
                examples.append(self._generate_iteration_pattern_example(i))
        
        return examples
    
    def generate_conditional_logic_examples(self, count: int = 100) -> List[Dict]:
        """Generate conditional logic examples."""
        examples = []
        
        for i in range(count):
            conditional_type = random.choice([
                "if_then_else", "nested_conditions", "comparison_operations",
                "logical_decisions", "value_classification"
            ])
            
            if conditional_type == "if_then_else":
                examples.append(self._generate_if_then_else_example(i))
            elif conditional_type == "nested_conditions":
                examples.append(self._generate_nested_conditions_example(i))
            elif conditional_type == "comparison_operations":
                examples.append(self._generate_comparison_example(i))
            elif conditional_type == "logical_decisions":
                examples.append(self._generate_logical_decision_example(i))
            elif conditional_type == "value_classification":
                examples.append(self._generate_classification_example(i))
        
        return examples
    
    def generate_hybrid_reasoning_examples(self, count: int = 150) -> List[Dict]:
        """Generate hybrid logic + computation examples."""
        examples = []
        
        for i in range(count):
            hybrid_type = random.choice([
                "logic_with_calculations", "data_analysis", "business_rules",
                "scientific_calculations", "optimization_problems"
            ])
            
            names = random.choice(self.name_sets)
            
            if hybrid_type == "logic_with_calculations":
                examples.append(self._generate_logic_calc_hybrid_example(names, i))
            elif hybrid_type == "data_analysis":
                examples.append(self._generate_data_analysis_example(names, i))
            elif hybrid_type == "business_rules":
                examples.append(self._generate_business_rules_example(names, i))
            elif hybrid_type == "scientific_calculations":
                examples.append(self._generate_scientific_calc_example(i))
            elif hybrid_type == "optimization_problems":
                examples.append(self._generate_optimization_example(i))
        
        return examples
    
    def generate_graph_algorithm_examples(self, count: int = 80) -> List[Dict]:
        """Generate graph algorithm examples."""
        examples = []
        
        for i in range(count):
            graph_type = random.choice([
                "reachability", "shortest_path", "connected_components",
                "graph_properties", "network_analysis"
            ])
            
            if graph_type == "reachability":
                examples.append(self._generate_reachability_example(i))
            elif graph_type == "shortest_path":
                examples.append(self._generate_shortest_path_example(i))
            elif graph_type == "connected_components":
                examples.append(self._generate_connected_components_example(i))
            elif graph_type == "graph_properties":
                examples.append(self._generate_graph_properties_example(i))
            elif graph_type == "network_analysis":
                examples.append(self._generate_network_analysis_example(i))
        
        return examples
    
    # Individual example generators
    def _generate_simple_facts_example(self, names: List[str], idx: int) -> Dict:
        """Generate simple facts and queries example."""
        a, b, c = names[:3]
        relation = random.choice(["likes", "knows", "teaches", "helps"])
        
        code = f"""REL {relation}
FACT {relation} {a} {b}
FACT {relation} {b} {c}
FACT {relation} {a} {c}
SOLVE
QUERY {relation} {a} ?"""
        
        return {
            "id": f"basic_facts_{idx}",
            "category": "basic_logic",
            "subcategory": "simple_facts",
            "difficulty": "beginner",
            "input": f"Who does {a.capitalize()} {relation}?",
            "output": f"I'll check who {a.capitalize()} {relation}: <computation>\n{code}\n</computation> → {b.capitalize()} and {c.capitalize()}",
            "bytelogic_code": code,
            "expected_result": [b, c]
        }
    
    def _generate_parent_child_example(self, names: List[str], idx: int) -> Dict:
        """Generate parent-child relationship example."""
        a, b, c, d = names[:4]
        
        code = f"""REL parent
REL child
FACT parent {a} {b}
FACT parent {a} {c}
RULE child: SCAN parent, EMIT child $1 $0
SOLVE
QUERY child ? {a}"""
        
        return {
            "id": f"parent_child_{idx}",
            "category": "basic_logic", 
            "subcategory": "family_relationships",
            "difficulty": "intermediate",
            "input": f"Who are {a.capitalize()}'s children?",
            "output": f"I'll find {a.capitalize()}'s children: <computation>\n{code}\n</computation> → {b.capitalize()} and {c.capitalize()}",
            "bytelogic_code": code,
            "expected_result": [b, c]
        }
    
    def _generate_basic_arithmetic_example(self, idx: int) -> Dict:
        """Generate basic arithmetic calculation example."""
        a = random.randint(10, 100)
        b = random.randint(5, 20)
        result = a + b
        
        code = f"""CALC add_numbers
INPUT $0 $1
LET $2 = $0 + $1
RESULT $2
END
RESULT CALC add_numbers({a}, {b})"""
        
        return {
            "id": f"basic_arithmetic_{idx}",
            "category": "mathematical_computation",
            "subcategory": "basic_arithmetic",
            "difficulty": "beginner",
            "input": f"What is {a} + {b}?",
            "output": f"I'll calculate the sum: <computation>\n{code}\n</computation> → {result}",
            "bytelogic_code": code,
            "expected_result": [result]
        }
    
    def _generate_character_counting_example(self, word: str, idx: int) -> Dict:
        """Generate character counting example."""
        char = random.choice(['a', 'e', 'i', 'o', 'u', 'r', 's', 't', 'n'])
        count = word.count(char)
        
        code = f"""CALC count_char
INPUT $word $target_char
LET $count = 0
FOR $i IN RANGE(0, LENGTH($word))
  LET $letter = CHAR_AT($word, $i)
  IF $letter == $target_char THEN
    LET $count = $count + 1
  END
END
RESULT $count
END
RESULT CALC count_char("{word}", "{char}")"""
        
        return {
            "id": f"char_count_{idx}",
            "category": "string_processing",
            "subcategory": "character_counting", 
            "difficulty": "intermediate",
            "input": f"How many '{char}' characters are in '{word}'?",
            "output": f"I'll count the '{char}' characters: <computation>\n{code}\n</computation> → {count}",
            "bytelogic_code": code,
            "expected_result": [count]
        }
    
    def _generate_for_range_example(self, idx: int) -> Dict:
        """Generate FOR-RANGE loop example."""
        start = random.randint(1, 5)
        end = start + random.randint(3, 7)
        total = sum(range(start, end + 1))
        
        code = f"""CALC sum_range
LET $total = 0
FOR $i IN RANGE({start}, {end + 1})
  LET $total = $total + $i
END
RESULT $total
END
RESULT CALC sum_range()"""
        
        return {
            "id": f"for_range_{idx}",
            "category": "loop_constructs",
            "subcategory": "for_range",
            "difficulty": "intermediate", 
            "input": f"What's the sum of numbers from {start} to {end}?",
            "output": f"I'll calculate the sum using a loop: <computation>\n{code}\n</computation> → {total}",
            "bytelogic_code": code,
            "expected_result": [total]
        }
    
    def _generate_if_then_else_example(self, idx: int) -> Dict:
        """Generate IF-THEN-ELSE conditional example."""
        value = random.randint(-20, 20)
        result = abs(value)
        
        code = f"""CALC absolute_value
INPUT $0
IF $0 < 0 THEN
  RESULT -$0
ELSE
  RESULT $0
END
END
RESULT CALC absolute_value({value})"""
        
        return {
            "id": f"conditional_{idx}",
            "category": "conditional_logic",
            "subcategory": "if_then_else",
            "difficulty": "beginner",
            "input": f"What's the absolute value of {value}?",
            "output": f"I'll calculate the absolute value: <computation>\n{code}\n</computation> → {result}",
            "bytelogic_code": code,
            "expected_result": [result]
        }
    
    def _generate_logic_calc_hybrid_example(self, names: List[str], idx: int) -> Dict:
        """Generate hybrid logic + calculation example."""
        a, b, c = names[:3]
        salary_a = random.randint(50000, 100000)
        salary_b = random.randint(45000, 95000) 
        salary_c = random.randint(40000, 90000)
        avg_salary = (salary_a + salary_b + salary_c) // 3
        
        code = f"""REL employee
REL salary
REL department_avg
FACT employee {a} engineering
FACT employee {b} engineering  
FACT employee {c} engineering
FACT salary {a} {salary_a}
FACT salary {b} {salary_b}
FACT salary {c} {salary_c}

CALC calc_avg_salary
INPUT $dept
LET $total = 0
LET $count = 0
FOR emp IN (QUERY employee ? $dept)
  FOR sal IN (QUERY salary emp.a ?)
    LET $total = $total + sal.b
    LET $count = $count + 1
  END
END
IF $count > 0 THEN
  RESULT $total / $count
ELSE
  RESULT 0
END
END

LET $avg = CALC calc_avg_salary("engineering")
FACT department_avg engineering $avg
SOLVE
QUERY department_avg engineering ?"""
        
        return {
            "id": f"hybrid_{idx}",
            "category": "hybrid_reasoning",
            "subcategory": "logic_with_calculations",
            "difficulty": "advanced",
            "input": f"What's the average salary in the engineering department?",
            "output": f"I'll calculate the department average: <computation>\n{code}\n</computation> → {avg_salary}",
            "bytelogic_code": code,
            "expected_result": [avg_salary]
        }
    
    def _generate_reachability_example(self, idx: int) -> Dict:
        """Generate graph reachability example."""
        nodes = ['a', 'b', 'c', 'd', 'e']
        edges = [('a', 'b'), ('b', 'c'), ('c', 'd'), ('a', 'e')]
        start, end = 'a', 'd'
        
        edges_facts = '\n'.join([f"FACT edge {src} {dst}" for src, dst in edges])
        
        code = f"""REL edge
REL reachable
{edges_facts}
RULE reachable: SCAN edge, EMIT reachable $0 $1
RULE reachable: SCAN edge, JOIN reachable $1, EMIT reachable $0 $2
SOLVE
QUERY reachable {start} {end}"""
        
        return {
            "id": f"reachability_{idx}",
            "category": "graph_algorithms",
            "subcategory": "reachability",
            "difficulty": "intermediate",
            "input": f"Can you reach node {end} from node {start}?",
            "output": f"I'll check graph reachability: <computation>\n{code}\n</computation> → Yes, {end} is reachable from {start}",
            "bytelogic_code": code,
            "expected_result": [1]
        }
    
    # Additional helper methods for other example types...
    def _generate_friendship_example(self, names: List[str], idx: int) -> Dict:
        """Generate friendship example with symmetric relations."""
        a, b, c = names[:3]
        
        code = f"""REL friend_directed
REL friend
FACT friend_directed {a} {b}
FACT friend_directed {b} {c}
RULE friend: SCAN friend_directed, EMIT friend $0 $1
RULE friend: SCAN friend_directed, EMIT friend $1 $0
SOLVE
QUERY friend {c} ?"""
        
        return {
            "id": f"friendship_{idx}",
            "category": "basic_logic",
            "subcategory": "symmetric_relations", 
            "difficulty": "intermediate",
            "input": f"Who are {c.capitalize()}'s friends?",
            "output": f"I'll find {c.capitalize()}'s friends: <computation>\n{code}\n</computation> → {b.capitalize()}",
            "bytelogic_code": code,
            "expected_result": [b]
        }
    
    def _generate_symmetric_example(self, names: List[str], idx: int) -> Dict:
        """Generate symmetric relationship example."""
        return self._generate_friendship_example(names, idx)
    
    def _generate_transitive_example(self, names: List[str], idx: int) -> Dict:
        """Generate transitive closure example."""
        a, b, c, d = names[:4]
        
        code = f"""REL knows
REL connected
FACT knows {a} {b}
FACT knows {b} {c}  
FACT knows {c} {d}
RULE connected: SCAN knows, EMIT connected $0 $1
RULE connected: SCAN knows, JOIN connected $1, EMIT connected $0 $2
SOLVE
QUERY connected {a} {d}"""
        
        return {
            "id": f"transitive_{idx}",
            "category": "basic_logic", 
            "subcategory": "transitive_closure",
            "difficulty": "intermediate",
            "input": f"Is {a.capitalize()} connected to {d.capitalize()}?",
            "output": f"I'll check transitive connections: <computation>\n{code}\n</computation> → Yes, {a.capitalize()} is connected to {d.capitalize()}",
            "bytelogic_code": code,
            "expected_result": [1]
        }
    
    def _generate_percentage_example(self, idx: int) -> Dict:
        """Generate percentage calculation example."""
        value = random.randint(100, 1000)
        percent = random.randint(10, 50)
        result = (value * percent) // 100
        
        code = f"""CALC percentage
INPUT $value $percent
LET $decimal = $percent / 100
RESULT $value * $decimal
END
RESULT CALC percentage({value}, {percent})"""
        
        return {
            "id": f"percentage_{idx}",
            "category": "mathematical_computation",
            "subcategory": "percentage",
            "difficulty": "beginner", 
            "input": f"What is {percent}% of {value}?",
            "output": f"I'll calculate the percentage: <computation>\n{code}\n</computation> → {result}",
            "bytelogic_code": code,
            "expected_result": [result]
        }
    
    def _generate_factorial_example(self, idx: int) -> Dict:
        """Generate factorial calculation example."""
        n = random.randint(3, 6)
        result = 1
        for i in range(1, n + 1):
            result *= i
        
        code = f"""CALC factorial
INPUT $n
IF $n <= 1 THEN
  RESULT 1
ELSE
  LET $prev = CALC factorial($n - 1)
  RESULT $n * $prev
END
END
RESULT CALC factorial({n})"""
        
        return {
            "id": f"factorial_{idx}",
            "category": "mathematical_computation",
            "subcategory": "factorial",
            "difficulty": "intermediate",
            "input": f"What is {n}! (factorial of {n})?", 
            "output": f"I'll calculate the factorial: <computation>\n{code}\n</computation> → {result}",
            "bytelogic_code": code,
            "expected_result": [result]
        }
    
    def _generate_fibonacci_example(self, idx: int) -> Dict:
        """Generate Fibonacci sequence example."""
        n = random.randint(5, 10)
        
        # Calculate Fibonacci
        a, b = 0, 1
        for _ in range(n):
            a, b = b, a + b
        result = a
        
        code = f"""CALC fibonacci
INPUT $n
LET $a = 0
LET $b = 1
LET $i = 0
FOR WHILE $i < $n
  LET $temp = $a + $b
  LET $a = $b
  LET $b = $temp
  LET $i = $i + 1
END
RESULT $a
END
RESULT CALC fibonacci({n})"""
        
        return {
            "id": f"fibonacci_{idx}",
            "category": "mathematical_computation",
            "subcategory": "fibonacci",
            "difficulty": "intermediate",
            "input": f"What is the {n}th Fibonacci number?",
            "output": f"I'll calculate the Fibonacci number: <computation>\n{code}\n</computation> → {result}",
            "bytelogic_code": code, 
            "expected_result": [result]
        }
    
    # Stub implementations for remaining methods to avoid errors
    def _generate_power_example(self, idx: int) -> Dict:
        """Generate power calculation example."""
        base = random.randint(2, 5)
        exp = random.randint(2, 4)
        result = base ** exp
        
        code = f"""CALC power
INPUT $base $exp
RESULT POW($base, $exp)
END
RESULT CALC power({base}, {exp})"""
        
        return {
            "id": f"power_{idx}",
            "category": "mathematical_computation",
            "subcategory": "power_calculations",
            "difficulty": "beginner",
            "input": f"What is {base} to the power of {exp}?",
            "output": f"I'll calculate the power: <computation>\n{code}\n</computation> → {result}",
            "bytelogic_code": code,
            "expected_result": [result]
        }
    
    def _generate_trigonometry_example(self, idx: int) -> Dict:
        """Generate trigonometry example."""
        angle = random.choice([0, 30, 45, 60, 90])
        
        code = f"""CALC sine_calc
INPUT $angle_degrees
LET $radians = $angle_degrees * 3.14159 / 180
RESULT SIN($radians)
END
RESULT CALC sine_calc({angle})"""
        
        return {
            "id": f"trigonometry_{idx}",
            "category": "mathematical_computation", 
            "subcategory": "trigonometry",
            "difficulty": "advanced",
            "input": f"What is sin({angle}°)?",
            "output": f"I'll calculate the sine: <computation>\n{code}\n</computation> → {round(math.sin(math.radians(angle)), 4) if angle != 90 else 1}",
            "bytelogic_code": code,
            "expected_result": [round(math.sin(math.radians(angle)) if angle != 90 else 1.0, 4)]
        }
    
    def _generate_math_functions_example(self, idx: int) -> Dict:
        """Generate mathematical functions example."""
        value = random.randint(16, 100)
        result = int(value ** 0.5)
        
        code = f"""CALC square_root
INPUT $value
RESULT SQRT($value)
END
RESULT CALC square_root({value})"""
        
        return {
            "id": f"math_func_{idx}",
            "category": "mathematical_computation",
            "subcategory": "mathematical_functions", 
            "difficulty": "beginner",
            "input": f"What is the square root of {value}?",
            "output": f"I'll calculate the square root: <computation>\n{code}\n</computation> → {result}",
            "bytelogic_code": code,
            "expected_result": [result]
        }
    
    # Continue with stubs for remaining methods...
    def _generate_vowel_counting_example(self, word: str, idx: int) -> Dict:
        vowels = sum(1 for c in word if c in 'aeiou')
        code = f"""CALC count_vowels
INPUT $word
LET $count = 0
FOR $i IN RANGE(0, LENGTH($word))
  LET $char = CHAR_AT($word, $i)
  IF $char == "a" THEN LET $count = $count + 1 END
  IF $char == "e" THEN LET $count = $count + 1 END
  IF $char == "i" THEN LET $count = $count + 1 END
  IF $char == "o" THEN LET $count = $count + 1 END
  IF $char == "u" THEN LET $count = $count + 1 END
END
RESULT $count
END
RESULT CALC count_vowels("{word}")"""
        
        return {
            "id": f"vowel_count_{idx}",
            "category": "string_processing",
            "subcategory": "vowel_counting",
            "difficulty": "intermediate", 
            "input": f"How many vowels are in '{word}'?",
            "output": f"I'll count the vowels: <computation>\n{code}\n</computation> → {vowels}",
            "bytelogic_code": code,
            "expected_result": [vowels]
        }
    
    # Add stubs for all remaining methods to prevent errors
    def _generate_word_analysis_example(self, word: str, idx: int) -> Dict:
        return self._generate_character_counting_example(word, idx)
    
    def _generate_string_comparison_example(self, idx: int) -> Dict:
        word1 = random.choice(self.test_words)
        word2 = random.choice(self.test_words)
        result = 1 if len(word1) > len(word2) else 0
        
        code = f"""CALC compare_lengths
INPUT $word1 $word2  
LET $len1 = LENGTH($word1)
LET $len2 = LENGTH($word2)
IF $len1 > $len2 THEN
  RESULT 1
ELSE
  RESULT 0
END
END
RESULT CALC compare_lengths("{word1}", "{word2}")"""
        
        return {
            "id": f"string_compare_{idx}",
            "category": "string_processing",
            "subcategory": "string_comparison",
            "difficulty": "intermediate",
            "input": f"Is '{word1}' longer than '{word2}'?",
            "output": f"I'll compare the lengths: <computation>\n{code}\n</computation> → {'Yes' if result else 'No'}",
            "bytelogic_code": code,
            "expected_result": [result]
        }
    
    def _generate_pattern_detection_example(self, word: str, idx: int) -> Dict:
        return self._generate_character_counting_example(word, idx)
    
    def _generate_for_while_example(self, idx: int) -> Dict:
        return self._generate_fibonacci_example(idx)
    
    def _generate_nested_loops_example(self, idx: int) -> Dict:
        size = random.randint(2, 4)
        result = size * size
        
        code = f"""CALC multiplication_table
LET $count = 0
FOR $i IN RANGE(1, {size + 1})
  FOR $j IN RANGE(1, {size + 1})
    LET $count = $count + 1
  END
END  
RESULT $count
END
RESULT CALC multiplication_table()"""
        
        return {
            "id": f"nested_loops_{idx}",
            "category": "loop_constructs",
            "subcategory": "nested_loops",
            "difficulty": "advanced",
            "input": f"How many entries in a {size}x{size} multiplication table?",
            "output": f"I'll count using nested loops: <computation>\n{code}\n</computation> → {result}",
            "bytelogic_code": code,
            "expected_result": [result]
        }
    
    def _generate_conditional_loop_example(self, idx: int) -> Dict:
        return self._generate_for_range_example(idx)
    
    def _generate_accumulation_loop_example(self, idx: int) -> Dict:
        return self._generate_for_range_example(idx)
    
    def _generate_iteration_pattern_example(self, idx: int) -> Dict:
        return self._generate_for_range_example(idx)
    
    def _generate_nested_conditions_example(self, idx: int) -> Dict:
        value = random.randint(1, 100)
        if value > 50:
            if value > 75:
                category = "large"
            else:
                category = "medium"
        else:
            category = "small"
        
        code = f"""CALC categorize_number
INPUT $value
IF $value > 50 THEN
  IF $value > 75 THEN
    RESULT "large"
  ELSE
    RESULT "medium"  
  END
ELSE
  RESULT "small"
END
END
RESULT CALC categorize_number({value})"""
        
        return {
            "id": f"nested_cond_{idx}",
            "category": "conditional_logic",
            "subcategory": "nested_conditions",
            "difficulty": "intermediate",
            "input": f"How would you categorize the number {value}?",
            "output": f"I'll categorize the number: <computation>\n{code}\n</computation> → {category}",
            "bytelogic_code": code,
            "expected_result": [category]
        }
    
    def _generate_comparison_example(self, idx: int) -> Dict:
        return self._generate_if_then_else_example(idx)
    
    def _generate_logical_decision_example(self, idx: int) -> Dict:
        return self._generate_if_then_else_example(idx)
    
    def _generate_classification_example(self, idx: int) -> Dict:
        return self._generate_nested_conditions_example(idx)
    
    def _generate_data_analysis_example(self, names: List[str], idx: int) -> Dict:
        return self._generate_logic_calc_hybrid_example(names, idx)
    
    def _generate_business_rules_example(self, names: List[str], idx: int) -> Dict:
        return self._generate_logic_calc_hybrid_example(names, idx)
    
    def _generate_scientific_calc_example(self, idx: int) -> Dict:
        return self._generate_power_example(idx)
    
    def _generate_optimization_example(self, idx: int) -> Dict:
        return self._generate_nested_loops_example(idx)
    
    def _generate_shortest_path_example(self, idx: int) -> Dict:
        return self._generate_reachability_example(idx)
    
    def _generate_connected_components_example(self, idx: int) -> Dict:
        return self._generate_reachability_example(idx)
    
    def _generate_graph_properties_example(self, idx: int) -> Dict:
        return self._generate_reachability_example(idx)
    
    def _generate_network_analysis_example(self, idx: int) -> Dict:
        return self._generate_reachability_example(idx)
    
    def generate_comprehensive_dataset(self, total_examples: int = 2000) -> Dict:
        """Generate a comprehensive dataset covering all ByteLogic features."""
        print("🚀 Generating comprehensive ByteLogic dataset...")
        
        # Calculate distribution
        distribution = {
            "basic_logic": 300,
            "mathematical_computation": 250,
            "string_processing": 200,
            "loop_constructs": 250,
            "conditional_logic": 200,
            "hybrid_reasoning": 300,
            "graph_algorithms": 150
        }
        
        all_examples = []
        
        # Generate each category
        for category, count in distribution.items():
            print(f"  📝 Generating {count} {category} examples...")
            
            if category == "basic_logic":
                examples = self.generate_basic_logic_examples(count)
            elif category == "mathematical_computation":
                examples = self.generate_mathematical_examples(count)
            elif category == "string_processing":
                examples = self.generate_string_processing_examples(count)
            elif category == "loop_constructs":
                examples = self.generate_loop_examples(count)
            elif category == "conditional_logic":
                examples = self.generate_conditional_logic_examples(count)
            elif category == "hybrid_reasoning":
                examples = self.generate_hybrid_reasoning_examples(count)
            elif category == "graph_algorithms":
                examples = self.generate_graph_algorithm_examples(count)
            
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
                "version": "2.0",
                "generator": "Comprehensive ByteLogic Dataset Generator",
                "total_examples": total,
                "train_examples": len(train_examples),
                "val_examples": len(val_examples),
                "test_examples": len(test_examples),
                "categories": list(distribution.keys()),
                "language_features": [
                    "relations_facts_rules", "queries", "calculations", 
                    "loops", "conditionals", "string_processing",
                    "expressions", "hybrid_reasoning"
                ],
                "difficulty_levels": ["beginner", "intermediate", "advanced"]
            },
            "train": [self._format_example(ex) for ex in train_examples],
            "validation": [self._format_example(ex) for ex in val_examples],
            "test": [self._format_example(ex) for ex in test_examples]
        }
        
        print(f"\n📊 Dataset Statistics:")
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
    """Generate the comprehensive dataset."""
    print("🚀 Comprehensive ByteLogic Dataset Generation")
    print("=" * 60)
    
    generator = ByteLogicDatasetGenerator()
    
    # Generate comprehensive dataset
    dataset = generator.generate_comprehensive_dataset(2000)
    
    # Save to file
    os.makedirs("training/datasets", exist_ok=True)
    output_file = "training/datasets/comprehensive_bytelogic_dataset_with_natural_language.json"
    
    with open(output_file, 'w') as f:
        json.dump(dataset, f, indent=2)
    
    print(f"\n💾 Comprehensive dataset saved to {output_file}")
    
    # Save JSONL files for each split
    for split in ["train", "validation", "test"]:
        jsonl_file = f"training/datasets/bytelogic_{split}_comprehensive_natural_language.jsonl"
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
        print(f"Output: {example['output'][:100]}...")
    
    print(f"\n🎉 Comprehensive ByteLogic dataset generation complete!")
    print(f"   Ready for ByteLogic-only training with {dataset['metadata']['total_examples']} examples")
    return True


if __name__ == "__main__":
    import math  # Add missing import
    success = main()
    exit(0 if success else 1)