#!/usr/bin/env python3
"""
Build comprehensive training dataset for WorldModel fine-tuning.
Teaches the model when to use WorldModel format vs normal responses.
"""

import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Any

def create_training_example(input_text: str, target_output: str, category: str, 
                          difficulty: str = "medium", metadata: Dict = None) -> Dict:
    """Create a training example in the required format."""
    if metadata is None:
        metadata = {}
    
    return {
        "input_text": input_text,
        "target_output": target_output,
        "metadata": metadata,
        "difficulty": difficulty,
        "category": category,
        "created_at": datetime.now(timezone.utc).isoformat()
    }

def generate_computational_examples() -> List[Dict]:
    """Generate examples that SHOULD use WorldModel format."""
    examples = []
    
    # Math calculations
    math_problems = [
        ("Calculate the square root of 144", 
         "<think>I need to find the square root of 144.</think>\n<model>\nimport math\nresult = math.sqrt(144)\nprint(result)\n</model>\n<requires>python:math</requires>"),
        
        ("Find the factorial of 6",
         "<think>I need to calculate 6! = 6 × 5 × 4 × 3 × 2 × 1.</think>\n<model>\nimport math\nresult = math.factorial(6)\nprint(result)\n</model>\n<requires>python:math</requires>"),
         
        ("Convert 100 degrees Fahrenheit to Celsius",
         "<think>To convert Fahrenheit to Celsius: C = (F - 32) × 5/9</think>\n<model>\nfahrenheit = 100\ncelsius = (fahrenheit - 32) * 5/9\nprint(f\"{fahrenheit}°F = {celsius:.2f}°C\")\n</model>\n<requires>python:math</requires>"),
         
        ("Calculate compound interest: $1000 at 5% for 3 years",
         "<think>Compound interest formula: A = P(1 + r)^t where P=1000, r=0.05, t=3</think>\n<model>\nprincipal = 1000\nrate = 0.05\ntime = 3\namount = principal * (1 + rate) ** time\ninterest = amount - principal\nprint(f\"Final amount: ${amount:.2f}\")\nprint(f\"Interest earned: ${interest:.2f}\")\n</model>\n<requires>python:math</requires>")
    ]
    
    for problem, solution in math_problems:
        examples.append(create_training_example(
            problem, solution, "math", "medium",
            {"requires_computation": True, "pattern": "calculation"}
        ))
    
    # Data processing
    data_problems = [
        ("Sort this list of numbers: [64, 34, 25, 12, 22, 11, 90]",
         "<think>I need to sort the given list of numbers in ascending order.</think>\n<model>\nnumbers = [64, 34, 25, 12, 22, 11, 90]\nsorted_numbers = sorted(numbers)\nprint(sorted_numbers)\n</model>\n<requires>python:data_processing</requires>"),
         
        ("Find the median of: [1, 3, 5, 7, 9, 11, 13]",
         "<think>To find the median of an odd number of elements, I take the middle value.</think>\n<model>\ndata = [1, 3, 5, 7, 9, 11, 13]\ndata.sort()\nn = len(data)\nmedian = data[n // 2]\nprint(f\"Median: {median}\")\n</model>\n<requires>python:data_processing</requires>")
    ]
    
    for problem, solution in data_problems:
        examples.append(create_training_example(
            problem, solution, "data_processing", "medium",
            {"requires_computation": True, "pattern": "data_analysis"}
        ))
    
    # Algorithm implementation
    algorithm_problems = [
        ("Implement binary search to find 7 in [1, 3, 5, 7, 9, 11]",
         "<think>I need to implement binary search to efficiently find 7 in the sorted array.</think>\n<model>\ndef binary_search(arr, target):\n    left, right = 0, len(arr) - 1\n    \n    while left <= right:\n        mid = (left + right) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            left = mid + 1\n        else:\n            right = mid - 1\n    return -1\n\narr = [1, 3, 5, 7, 9, 11]\ntarget = 7\nresult = binary_search(arr, target)\nprint(f\"Found {target} at index {result}\")\n</model>\n<requires>python:algorithms</requires>"),
        
        ("Generate all permutations of [1, 2, 3]",
         "<think>I need to generate all possible permutations of the list [1, 2, 3].</think>\n<model>\nfrom itertools import permutations\n\nitems = [1, 2, 3]\nperms = list(permutations(items))\nfor perm in perms:\n    print(list(perm))\n</model>\n<requires>python:algorithms</requires>")
    ]
    
    for problem, solution in algorithm_problems:
        examples.append(create_training_example(
            problem, solution, "algorithms", "hard",
            {"requires_computation": True, "pattern": "algorithm"}
        ))
    
    return examples

def generate_non_computational_examples() -> List[Dict]:
    """Generate examples that should NOT use WorldModel format."""
    examples = []
    
    # Factual questions - just direct answers
    factual_qa = [
        ("What is the capital of Japan?", "Tokyo is the capital of Japan."),
        ("Who invented the telephone?", "Alexander Graham Bell invented the telephone in 1876."),
        ("What is the largest planet in our solar system?", "Jupiter is the largest planet in our solar system."),
        ("What does CPU stand for?", "CPU stands for Central Processing Unit."),
        ("What year did World War II end?", "World War II ended in 1945.")
    ]
    
    for question, answer in factual_qa:
        examples.append(create_training_example(
            question, answer, "factual", "easy",
            {"requires_computation": False, "pattern": "direct_answer"}
        ))
    
    # Explanations - detailed but no code
    explanations = [
        ("Explain what machine learning is",
         "Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed. It uses algorithms to analyze data, identify patterns, and make predictions or decisions based on that analysis."),
         
        ("What is the difference between HTTP and HTTPS?",
         "HTTP (HyperText Transfer Protocol) is the foundation of data communication on the web, while HTTPS (HTTP Secure) is the secure version that encrypts data transmission using SSL/TLS protocols. HTTPS protects against eavesdropping and man-in-the-middle attacks."),
         
        ("Describe what a database is",
         "A database is an organized collection of structured information or data, typically stored electronically in a computer system. It's managed by a database management system (DBMS) that allows users to create, read, update, and delete data efficiently.")
    ]
    
    for question, answer in explanations:
        examples.append(create_training_example(
            question, answer, "explanation", "medium",
            {"requires_computation": False, "pattern": "explanation"}
        ))
    
    # Creative/Opinion - no computation needed
    creative = [
        ("Write a short poem about coding",
         "Lines of code flow like a river,\\nLogic dancing in the night,\\nBugs and features, give and take,\\nUntil solutions shine so bright."),
         
        ("What are the benefits of learning Python?",
         "Learning Python offers many benefits: it has simple, readable syntax that's beginner-friendly; extensive libraries for data science, web development, and automation; strong community support; and high demand in the job market across many industries.")
    ]
    
    for question, answer in creative:
        examples.append(create_training_example(
            question, answer, "creative", "easy",
            {"requires_computation": False, "pattern": "creative"}
        ))
    
    return examples

def generate_mixed_examples() -> List[Dict]:
    """Generate examples that show the boundary between computational and non-computational."""
    examples = []
    
    # These show when to switch to WorldModel mode
    boundary_examples = [
        # Asking about a concept vs calculating it
        ("What is compound interest?",
         "Compound interest is interest calculated on the initial principal and the accumulated interest from previous periods. It grows exponentially over time, making it more powerful than simple interest."),
         
        # Follow-up that needs calculation would use WorldModel format
        ("How do I calculate the area of a circle?",
         "To calculate the area of a circle, use the formula A = πr², where A is the area, π (pi) is approximately 3.14159, and r is the radius of the circle."),
         
        # Vs asking for specific calculation (would use WorldModel)
        
        # Explaining vs implementing
        ("What is the bubble sort algorithm?",
         "Bubble sort is a simple sorting algorithm that repeatedly steps through a list, compares adjacent elements, and swaps them if they're in the wrong order. This process repeats until no more swaps are needed. While easy to understand, it's inefficient for large datasets with O(n²) time complexity."),
    ]
    
    for question, answer in boundary_examples:
        examples.append(create_training_example(
            question, answer, "boundary", "medium",
            {"requires_computation": False, "pattern": "conceptual"}
        ))
    
    return examples

def generate_reasoning_examples() -> List[Dict]:
    """Generate examples that require multi-step reasoning with WorldModel format."""
    examples = []
    
    reasoning_problems = [
        ("A train travels 120 miles in 2 hours. How fast was it going in mph? Then calculate how far it would travel in 5 hours at that speed.",
         "<think>First I need to find the speed: speed = distance/time = 120/2 = 60 mph. Then calculate distance in 5 hours: distance = speed × time = 60 × 5 = 300 miles.</think>\n<model>\n# Step 1: Calculate speed\ndistance1 = 120  # miles\ntime1 = 2  # hours\nspeed = distance1 / time1\nprint(f\"Speed: {speed} mph\")\n\n# Step 2: Calculate distance in 5 hours\ntime2 = 5  # hours\ndistance2 = speed * time2\nprint(f\"Distance in {time2} hours: {distance2} miles\")\n</model>\n<requires>python:math</requires>"),
        
        ("If I buy 3 books for $12 each and 2 notebooks for $4 each, how much change do I get from $50?",
         "<think>I need to calculate total cost: (3 × $12) + (2 × $4) = $36 + $8 = $44. Change from $50 = $50 - $44 = $6.</think>\n<model>\nbooks = 3 * 12\nnotebooks = 2 * 4\ntotal_cost = books + notebooks\nmoney_given = 50\nchange = money_given - total_cost\nprint(f\"Books: ${books}\")\nprint(f\"Notebooks: ${notebooks}\")\nprint(f\"Total cost: ${total_cost}\")\nprint(f\"Change: ${change}\")\n</model>\n<requires>python:math</requires>")
    ]
    
    for problem, solution in reasoning_problems:
        examples.append(create_training_example(
            problem, solution, "reasoning", "hard",
            {"requires_computation": True, "pattern": "multi_step_reasoning"}
        ))
    
    return examples

def main():
    """Build comprehensive training dataset."""
    print("Building comprehensive WorldModel training dataset...")
    
    # Generate different types of examples
    computational = generate_computational_examples()
    non_computational = generate_non_computational_examples()
    mixed = generate_mixed_examples()
    reasoning = generate_reasoning_examples()
    
    # Combine all examples
    all_examples = computational + non_computational + mixed + reasoning
    
    # Shuffle for better training
    random.shuffle(all_examples)
    
    # Create dataset metadata
    dataset = {
        "metadata": {
            "total_examples": len(all_examples),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "generator_version": "2.0",
            "description": "Comprehensive WorldModel training dataset with computational and non-computational examples",
            "categories": {
                "computational": len(computational),
                "non_computational": len(non_computational), 
                "boundary": len(mixed),
                "reasoning": len(reasoning)
            }
        },
        "examples": all_examples
    }
    
    # Save dataset
    output_file = Path("./data/worldmodel_comprehensive_training.json")
    output_file.parent.mkdir(exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(dataset, f, indent=2)
    
    print(f"✅ Dataset created with {len(all_examples)} examples")
    print(f"   - {len(computational)} computational examples")
    print(f"   - {len(non_computational)} non-computational examples") 
    print(f"   - {len(mixed)} boundary examples")
    print(f"   - {len(reasoning)} reasoning examples")
    print(f"   - Saved to {output_file}")
    
    # Print sample examples
    print("\n📝 Sample Examples:")
    for i, example in enumerate(all_examples[:3]):
        print(f"\n{i+1}. {example['category'].upper()}")
        print(f"   Input: {example['input_text']}")
        print(f"   Output: {example['target_output'][:100]}...")

if __name__ == "__main__":
    main()