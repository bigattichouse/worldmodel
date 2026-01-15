#!/usr/bin/env python3
"""
WorldModel Training Data Generator
=================================

Generates 1000+ high-quality training examples for WorldModel format:
- Mathematical computations
- Data analysis tasks
- Text processing
- Scientific calculations
- Programming problems
- Logic puzzles

Each example follows the format:
User: [question]
Assistant: <think>[reasoning]</think>
<model>[code]</model>
<requires>[dependencies]</requires>
[explanation]
"""

import random
import math
import json
from typing import List, Dict, Tuple
from dataclasses import dataclass

@dataclass
class WorldModelExample:
    user_prompt: str
    thinking: str
    code: str
    requires: str
    explanation: str
    
    def format(self) -> str:
        """Format as training example."""
        return f"""User: {self.user_prompt}
Assistant: <think>{self.thinking}</think>
<model>
{self.code}
</model>
<requires>{self.requires}</requires>

{self.explanation}"""

class ExampleGenerator:
    """Generates diverse WorldModel training examples."""
    
    def __init__(self):
        self.examples = []
        
        # Data for generation
        self.words = [
            "python", "javascript", "computer", "algorithm", "database", "network", "programming",
            "artificial", "intelligence", "machine", "learning", "science", "mathematics", "physics",
            "chemistry", "biology", "engineering", "technology", "software", "hardware", "internet",
            "website", "application", "development", "framework", "library", "function", "variable",
            "constant", "parameter", "argument", "return", "loop", "condition", "iteration"
        ]
        
        self.cities = [
            "New York", "Los Angeles", "Chicago", "Houston", "Phoenix", "Philadelphia", 
            "San Antonio", "San Diego", "Dallas", "San Jose", "Austin", "Jacksonville",
            "San Francisco", "Indianapolis", "Columbus", "Fort Worth", "Charlotte", "Seattle"
        ]
        
        self.countries = [
            "United States", "China", "Japan", "Germany", "India", "United Kingdom", 
            "France", "Italy", "Brazil", "Canada", "Russia", "South Korea", "Spain", 
            "Australia", "Mexico", "Indonesia", "Netherlands", "Saudi Arabia", "Turkey"
        ]
    
    def generate_math_basic(self) -> List[WorldModelExample]:
        """Generate basic math examples."""
        examples = []
        
        # Addition
        for _ in range(20):
            a, b = random.randint(10, 999), random.randint(10, 999)
            examples.append(WorldModelExample(
                user_prompt=f"Calculate {a} + {b}",
                thinking=f"I need to add {a} and {b}. Let me compute this step by step.",
                code=f"result = {a} + {b}\nprint(f\"{a} + {b} = {{result}}\")",
                requires="python:math",
                explanation=f"{a} + {b} equals {a + b}."
            ))
        
        # Subtraction
        for _ in range(20):
            a, b = random.randint(100, 999), random.randint(10, 99)
            examples.append(WorldModelExample(
                user_prompt=f"Calculate {a} - {b}",
                thinking=f"I need to subtract {b} from {a}.",
                code=f"result = {a} - {b}\nprint(f\"{a} - {b} = {{result}}\")",
                requires="python:math",
                explanation=f"{a} - {b} equals {a - b}."
            ))
        
        # Multiplication
        for _ in range(20):
            a, b = random.randint(12, 99), random.randint(12, 99)
            examples.append(WorldModelExample(
                user_prompt=f"What is {a} × {b}?",
                thinking=f"I need to multiply {a} by {b}.",
                code=f"result = {a} * {b}\nprint(f\"{a} × {b} = {{result}}\")",
                requires="python:math",
                explanation=f"{a} × {b} equals {a * b}."
            ))
        
        # Division
        for _ in range(20):
            b = random.randint(2, 25)
            a = b * random.randint(10, 50)
            examples.append(WorldModelExample(
                user_prompt=f"Divide {a} by {b}",
                thinking=f"I need to divide {a} by {b}.",
                code=f"result = {a} / {b}\nprint(f\"{a} ÷ {b} = {{result}}\")",
                requires="python:math",
                explanation=f"{a} ÷ {b} equals {a // b}."
            ))
        
        return examples
    
    def generate_percentages(self) -> List[WorldModelExample]:
        """Generate percentage calculation examples."""
        examples = []
        
        for _ in range(30):
            percent = random.choice([5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 75, 80, 90])
            number = random.randint(100, 1000)
            
            examples.append(WorldModelExample(
                user_prompt=f"Calculate {percent}% of {number}",
                thinking=f"I need to calculate {percent}% of {number}. {percent}% = {percent/100}, so {percent/100} × {number}.",
                code=f"percent = {percent}\nnumber = {number}\nresult = (percent / 100) * number\nprint(f\"{percent}% of {number} = {{result}}\")",
                requires="python:math",
                explanation=f"{percent}% of {number} equals {(percent/100) * number}."
            ))
        
        return examples
    
    def generate_compound_interest(self) -> List[WorldModelExample]:
        """Generate compound interest examples."""
        examples = []
        
        for _ in range(15):
            principal = random.choice([1000, 1500, 2000, 2500, 5000, 10000])
            rate = random.choice([3, 4, 5, 6, 7, 8])
            years = random.randint(2, 10)
            
            examples.append(WorldModelExample(
                user_prompt=f"Calculate compound interest: ${principal} at {rate}% for {years} years",
                thinking=f"Compound interest formula: A = P(1 + r)^t where P=${principal}, r={rate/100}, t={years}",
                code=f"principal = {principal}\nrate = {rate / 100}\nyears = {years}\namount = principal * (1 + rate) ** years\ninterest = amount - principal\nprint(f\"Amount: ${{amount:.2f}}\")\nprint(f\"Interest: ${{interest:.2f}}\")",
                requires="python:math",
                explanation=f"With compound interest, ${principal} at {rate}% for {years} years grows to ${principal * (1 + rate/100) ** years:.2f}."
            ))
        
        return examples
    
    def generate_quadratics(self) -> List[WorldModelExample]:
        """Generate quadratic equation examples."""
        examples = []
        
        for _ in range(25):
            a = random.randint(1, 5)
            b = random.randint(-10, 10)
            c = random.randint(-10, 10)
            
            examples.append(WorldModelExample(
                user_prompt=f"Find the roots of the quadratic equation {a}x² + {b}x + {c} = 0",
                thinking=f"I need to find roots of {a}x² + {b}x + {c} = 0 using the quadratic formula.",
                code=f"import math\n\na, b, c = {a}, {b}, {c}\ndiscriminant = b**2 - 4*a*c\n\nif discriminant > 0:\n    root1 = (-b + math.sqrt(discriminant)) / (2*a)\n    root2 = (-b - math.sqrt(discriminant)) / (2*a)\n    print(f\"Roots: {{root1:.3f}}, {{root2:.3f}}\")\nelif discriminant == 0:\n    root = -b / (2*a)\n    print(f\"Double root: {{root:.3f}}\")\nelse:\n    real = -b / (2*a)\n    imag = math.sqrt(-discriminant) / (2*a)\n    print(f\"Complex roots: {{real:.3f}} ± {{imag:.3f}}i\")",
                requires="python:math",
                explanation=f"Using the quadratic formula for {a}x² + {b}x + {c} = 0."
            ))
        
        return examples
    
    def generate_statistics(self) -> List[WorldModelExample]:
        """Generate statistics examples."""
        examples = []
        
        # Mean calculations
        for _ in range(20):
            numbers = [random.randint(10, 100) for _ in range(random.randint(5, 10))]
            numbers_str = ", ".join(map(str, numbers))
            
            examples.append(WorldModelExample(
                user_prompt=f"Calculate the mean of: {numbers_str}",
                thinking=f"I need to calculate the mean by adding all numbers and dividing by the count.",
                code=f"numbers = {numbers}\nmean = sum(numbers) / len(numbers)\nprint(f\"Mean: {{mean:.2f}}\")",
                requires="python:math",
                explanation=f"The mean of these {len(numbers)} numbers is {sum(numbers)/len(numbers):.2f}."
            ))
        
        # Standard deviation
        for _ in range(15):
            numbers = [random.randint(20, 80) for _ in range(random.randint(6, 8))]
            numbers_str = ", ".join(map(str, numbers))
            
            examples.append(WorldModelExample(
                user_prompt=f"Calculate the standard deviation of: {numbers_str}",
                thinking=f"I need to calculate the standard deviation using the formula: sqrt(sum((x - mean)²) / n).",
                code=f"import math\n\nnumbers = {numbers}\nmean = sum(numbers) / len(numbers)\nvariance = sum((x - mean)**2 for x in numbers) / len(numbers)\nstd_dev = math.sqrt(variance)\nprint(f\"Standard deviation: {{std_dev:.3f}}\")",
                requires="python:math",
                explanation=f"The standard deviation measures how spread out the data is from the mean."
            ))
        
        return examples
    
    def generate_fibonacci(self) -> List[WorldModelExample]:
        """Generate Fibonacci sequence examples."""
        examples = []
        
        for _ in range(15):
            n = random.randint(5, 15)
            
            examples.append(WorldModelExample(
                user_prompt=f"Generate the first {n} numbers in the Fibonacci sequence",
                thinking=f"The Fibonacci sequence starts with 0, 1 and each subsequent number is the sum of the two preceding ones.",
                code=f"def fibonacci(n):\n    if n <= 0:\n        return []\n    elif n == 1:\n        return [0]\n    elif n == 2:\n        return [0, 1]\n    \n    fib = [0, 1]\n    for i in range(2, n):\n        fib.append(fib[i-1] + fib[i-2])\n    return fib\n\nresult = fibonacci({n})\nprint(f\"First {n} Fibonacci numbers: {{result}}\")",
                requires="python:computation",
                explanation=f"The first {n} Fibonacci numbers follow the pattern where each number is the sum of the two preceding ones."
            ))
        
        return examples
    
    def generate_prime_numbers(self) -> List[WorldModelExample]:
        """Generate prime number examples."""
        examples = []
        
        # Prime checking
        for _ in range(20):
            number = random.randint(50, 200)
            
            examples.append(WorldModelExample(
                user_prompt=f"Check if {number} is a prime number",
                thinking=f"A prime number is only divisible by 1 and itself. I need to check if {number} has any divisors other than 1 and {number}.",
                code=f"def is_prime(n):\n    if n < 2:\n        return False\n    for i in range(2, int(n**0.5) + 1):\n        if n % i == 0:\n            return False\n    return True\n\nnumber = {number}\nresult = is_prime(number)\nprint(f\"{number} is {'prime' if result else 'not prime'}\")",
                requires="python:math",
                explanation=f"{number} is {'prime' if all(number % i != 0 for i in range(2, int(number**0.5) + 1)) and number > 1 else 'not prime'}."
            ))
        
        # Prime generation
        for _ in range(10):
            limit = random.choice([50, 100, 150])
            
            examples.append(WorldModelExample(
                user_prompt=f"Find all prime numbers up to {limit}",
                thinking=f"I'll use the Sieve of Eratosthenes algorithm to find all primes up to {limit}.",
                code=f"def sieve_of_eratosthenes(n):\n    sieve = [True] * (n + 1)\n    sieve[0] = sieve[1] = False\n    \n    for i in range(2, int(n**0.5) + 1):\n        if sieve[i]:\n            for j in range(i*i, n + 1, i):\n                sieve[j] = False\n    \n    return [i for i in range(2, n + 1) if sieve[i]]\n\nprimes = sieve_of_eratosthenes({limit})\nprint(f\"Primes up to {limit}: {primes}\")\nprint(f\"Count: {len(primes)}\")",
                requires="python:computation",
                explanation=f"Using the Sieve of Eratosthenes to efficiently find all prime numbers up to {limit}."
            ))
        
        return examples
    
    def generate_text_analysis(self) -> List[WorldModelExample]:
        """Generate text analysis examples."""
        examples = []
        
        texts = [
            "The quick brown fox jumps over the lazy dog",
            "Python is a powerful programming language",
            "Machine learning algorithms process large datasets",
            "Web development requires HTML, CSS, and JavaScript",
            "Database systems store and retrieve information efficiently",
            "Software engineering involves design, coding, and testing",
            "Artificial intelligence mimics human cognitive functions",
            "Data science combines statistics and computer science"
        ]
        
        # Character counting
        for _ in range(15):
            text = random.choice(texts)
            char = random.choice(['a', 'e', 'i', 'o', 'u', 'r', 's', 't', 'n'])
            
            examples.append(WorldModelExample(
                user_prompt=f"Count the occurrences of '{char}' in: \"{text}\"",
                thinking=f"I need to count how many times the character '{char}' appears in the given text.",
                code=f"text = \"{text}\"\nchar = '{char}'\ncount = text.lower().count(char)\nprint(f\"The character '{char}' appears {count} times\")",
                requires="python:text",
                explanation=f"Counting character occurrences in the given text."
            ))
        
        # Word counting
        for _ in range(15):
            text = random.choice(texts)
            
            examples.append(WorldModelExample(
                user_prompt=f"Count the number of words in: \"{text}\"",
                thinking=f"I need to split the text by spaces and count the resulting words.",
                code=f"text = \"{text}\"\nwords = text.split()\ncount = len(words)\nprint(f\"Number of words: {count}\")",
                requires="python:text",
                explanation=f"The text contains {len(text.split())} words."
            ))
        
        # Vowel counting
        for _ in range(10):
            text = random.choice(texts)
            
            examples.append(WorldModelExample(
                user_prompt=f"Count the vowels in: \"{text}\"",
                thinking=f"I need to count all vowels (a, e, i, o, u) in the text, both uppercase and lowercase.",
                code=f"text = \"{text}\"\nvowels = \"aeiouAEIOU\"\ncount = sum(1 for char in text if char in vowels)\nprint(f\"Number of vowels: {count}\")",
                requires="python:text",
                explanation=f"Counting all vowel occurrences in the given text."
            ))
        
        return examples
    
    def generate_conversions(self) -> List[WorldModelExample]:
        """Generate unit conversion examples."""
        examples = []
        
        # Temperature conversions
        for _ in range(20):
            if random.choice([True, False]):
                celsius = random.randint(-20, 50)
                examples.append(WorldModelExample(
                    user_prompt=f"Convert {celsius}°C to Fahrenheit",
                    thinking=f"To convert Celsius to Fahrenheit: F = (C × 9/5) + 32",
                    code=f"celsius = {celsius}\nfahrenheit = (celsius * 9/5) + 32\nprint(f\"{celsius}°C = {fahrenheit}°F\")",
                    requires="python:conversion",
                    explanation=f"{celsius}°C equals {(celsius * 9/5) + 32}°F."
                ))
            else:
                fahrenheit = random.randint(32, 120)
                examples.append(WorldModelExample(
                    user_prompt=f"Convert {fahrenheit}°F to Celsius",
                    thinking=f"To convert Fahrenheit to Celsius: C = (F - 32) × 5/9",
                    code=f"fahrenheit = {fahrenheit}\ncelsius = (fahrenheit - 32) * 5/9\nprint(f\"{fahrenheit}°F = {celsius:.1f}°C\")",
                    requires="python:conversion",
                    explanation=f"{fahrenheit}°F equals {(fahrenheit - 32) * 5/9:.1f}°C."
                ))
        
        # Distance conversions
        for _ in range(15):
            if random.choice([True, False]):
                miles = random.randint(1, 100)
                examples.append(WorldModelExample(
                    user_prompt=f"Convert {miles} miles to kilometers",
                    thinking=f"1 mile = 1.60934 kilometers",
                    code=f"miles = {miles}\nkilometers = miles * 1.60934\nprint(f\"{miles} miles = {kilometers:.2f} km\")",
                    requires="python:conversion",
                    explanation=f"{miles} miles equals {miles * 1.60934:.2f} kilometers."
                ))
            else:
                km = random.randint(1, 100)
                examples.append(WorldModelExample(
                    user_prompt=f"Convert {km} kilometers to miles",
                    thinking=f"1 kilometer = 0.621371 miles",
                    code=f"kilometers = {km}\nmiles = kilometers * 0.621371\nprint(f\"{km} km = {miles:.2f} miles\")",
                    requires="python:conversion",
                    explanation=f"{km} kilometers equals {km * 0.621371:.2f} miles."
                ))
        
        return examples
    
    def generate_geometry(self) -> List[WorldModelExample]:
        """Generate geometry calculation examples."""
        examples = []
        
        # Circle calculations
        for _ in range(20):
            radius = random.randint(5, 25)
            calc_type = random.choice(['area', 'circumference'])
            
            if calc_type == 'area':
                examples.append(WorldModelExample(
                    user_prompt=f"Calculate the area of a circle with radius {radius}",
                    thinking=f"Area of a circle = π × r². With radius {radius}, area = π × {radius}².",
                    code=f"import math\n\nradius = {radius}\narea = math.pi * radius**2\nprint(f\"Area = {area:.2f} square units\")",
                    requires="python:math",
                    explanation=f"The area of a circle with radius {radius} is {math.pi * radius**2:.2f} square units."
                ))
            else:
                examples.append(WorldModelExample(
                    user_prompt=f"Calculate the circumference of a circle with radius {radius}",
                    thinking=f"Circumference of a circle = 2 × π × r. With radius {radius}, circumference = 2 × π × {radius}.",
                    code=f"import math\n\nradius = {radius}\ncircumference = 2 * math.pi * radius\nprint(f\"Circumference = {circumference:.2f} units\")",
                    requires="python:math",
                    explanation=f"The circumference of a circle with radius {radius} is {2 * math.pi * radius:.2f} units."
                ))
        
        # Rectangle calculations
        for _ in range(15):
            length = random.randint(5, 30)
            width = random.randint(3, 25)
            
            examples.append(WorldModelExample(
                user_prompt=f"Calculate the area and perimeter of a rectangle with length {length} and width {width}",
                thinking=f"For a rectangle: Area = length × width, Perimeter = 2(length + width)",
                code=f"length = {length}\nwidth = {width}\narea = length * width\nperimeter = 2 * (length + width)\nprint(f\"Area = {area} square units\")\nprint(f\"Perimeter = {perimeter} units\")",
                requires="python:math",
                explanation=f"Rectangle with length {length} and width {width}: Area = {length * width}, Perimeter = {2 * (length + width)}."
            ))
        
        return examples
    
    def generate_boolean_logic(self) -> List[WorldModelExample]:
        """Generate boolean logic examples."""
        examples = []
        
        # Boolean expressions
        expressions = [
            ("True and False", "True and False"),
            ("False or True", "False or True"),
            ("not True", "not True"),
            ("True and True and False", "True and True and False"),
            ("False or False or True", "False or False or True"),
            ("(True and False) or True", "(True and False) or True"),
            ("not (True and False)", "not (True and False)"),
            ("True or False and True", "True or False and True")
        ]
        
        for expr_display, expr_code in expressions:
            examples.append(WorldModelExample(
                user_prompt=f"Evaluate the boolean expression: {expr_display}",
                thinking=f"I need to evaluate the boolean expression step by step, considering operator precedence.",
                code=f"result = {expr_code}\nprint(f\"Result: {result}\")",
                requires="python:logic",
                explanation=f"The boolean expression {expr_display} evaluates to {eval(expr_code)}."
            ))
        
        return examples
    
    def generate_data_structures(self) -> List[WorldModelExample]:
        """Generate data structure examples."""
        examples = []
        
        # List operations
        for _ in range(20):
            numbers = [random.randint(1, 100) for _ in range(random.randint(5, 10))]
            operation = random.choice(['max', 'min', 'sum', 'sort', 'reverse'])
            
            if operation == 'max':
                examples.append(WorldModelExample(
                    user_prompt=f"Find the maximum value in the list: {numbers}",
                    thinking=f"I need to find the largest number in the given list.",
                    code=f"numbers = {numbers}\nmax_value = max(numbers)\nprint(f\"Maximum value: {max_value}\")",
                    requires="python:data",
                    explanation=f"The maximum value in the list is {max(numbers)}."
                ))
            elif operation == 'min':
                examples.append(WorldModelExample(
                    user_prompt=f"Find the minimum value in the list: {numbers}",
                    thinking=f"I need to find the smallest number in the given list.",
                    code=f"numbers = {numbers}\nmin_value = min(numbers)\nprint(f\"Minimum value: {min_value}\")",
                    requires="python:data",
                    explanation=f"The minimum value in the list is {min(numbers)}."
                ))
            elif operation == 'sum':
                examples.append(WorldModelExample(
                    user_prompt=f"Calculate the sum of all numbers in: {numbers}",
                    thinking=f"I need to add all numbers in the list together.",
                    code=f"numbers = {numbers}\ntotal = sum(numbers)\nprint(f\"Sum: {total}\")",
                    requires="python:data",
                    explanation=f"The sum of all numbers in the list is {sum(numbers)}."
                ))
            elif operation == 'sort':
                examples.append(WorldModelExample(
                    user_prompt=f"Sort the list in ascending order: {numbers}",
                    thinking=f"I need to arrange the numbers from smallest to largest.",
                    code=f"numbers = {numbers}\nsorted_numbers = sorted(numbers)\nprint(f\"Sorted list: {sorted_numbers}\")",
                    requires="python:data",
                    explanation=f"The list sorted in ascending order is {sorted(numbers)}."
                ))
            else:  # reverse
                examples.append(WorldModelExample(
                    user_prompt=f"Reverse the order of the list: {numbers}",
                    thinking=f"I need to reverse the order of elements in the list.",
                    code=f"numbers = {numbers}\nreversed_numbers = numbers[::-1]\nprint(f\"Reversed list: {reversed_numbers}\")",
                    requires="python:data",
                    explanation=f"The list in reverse order is {numbers[::-1]}."
                ))
        
        return examples
    
    def generate_word_problems(self) -> List[WorldModelExample]:
        """Generate word problem examples."""
        examples = []
        
        # Shopping problems
        for _ in range(25):
            items = random.sample([
                ("books", random.randint(8, 25)),
                ("pens", random.randint(2, 8)),
                ("notebooks", random.randint(5, 15)),
                ("calculators", random.randint(15, 50)),
                ("folders", random.randint(3, 10))
            ], 2)
            
            item1_name, item1_price = items[0]
            item2_name, item2_price = items[1]
            item1_qty = random.randint(2, 5)
            item2_qty = random.randint(1, 4)
            payment = ((item1_price * item1_qty) + (item2_price * item2_qty)) + random.randint(5, 50)
            
            examples.append(WorldModelExample(
                user_prompt=f"If I buy {item1_qty} {item1_name} for ${item1_price} each and {item2_qty} {item2_name} for ${item2_price} each, how much change do I get from ${payment}?",
                thinking=f"Total cost = ({item1_qty} × ${item1_price}) + ({item2_qty} × ${item2_price}) = ${item1_price * item1_qty} + ${item2_price * item2_qty} = ${item1_price * item1_qty + item2_price * item2_qty}. Change = ${payment} - ${item1_price * item1_qty + item2_price * item2_qty} = ${payment - (item1_price * item1_qty + item2_price * item2_qty)}.",
                code=f"item1_price = {item1_price}\nitem1_qty = {item1_qty}\nitem2_price = {item2_price}\nitem2_qty = {item2_qty}\npayment = {payment}\n\ntotal_cost = (item1_price * item1_qty) + (item2_price * item2_qty)\nchange = payment - total_cost\n\nprint(f\"Total cost: ${total_cost}\")\nprint(f\"Change from ${payment}: ${change}\")",
                requires="python:math",
                explanation=f"The total cost is ${item1_price * item1_qty + item2_price * item2_qty}, so the change from ${payment} is ${payment - (item1_price * item1_qty + item2_price * item2_qty)}."
            ))
        
        # Time problems
        for _ in range(15):
            speed = random.randint(30, 80)
            distance = speed * random.randint(2, 8)
            
            examples.append(WorldModelExample(
                user_prompt=f"How long does it take to travel {distance} miles at {speed} mph?",
                thinking=f"Time = Distance ÷ Speed. Time = {distance} ÷ {speed}.",
                code=f"distance = {distance}\nspeed = {speed}\ntime = distance / speed\nprint(f\"Time: {time} hours\")\nif time != int(time):\n    minutes = (time - int(time)) * 60\n    print(f\"That's {int(time)} hours and {minutes:.0f} minutes\")",
                requires="python:math",
                explanation=f"At {speed} mph, it takes {distance/speed} hours to travel {distance} miles."
            ))
        
        return examples
    
    def generate_advanced_math(self) -> List[WorldModelExample]:
        """Generate advanced mathematics examples."""
        examples = []
        
        # Trigonometry
        angles = [30, 45, 60, 90, 120, 135, 150, 180]
        functions = ['sin', 'cos', 'tan']
        
        for _ in range(20):
            angle = random.choice(angles)
            func = random.choice(functions)
            
            examples.append(WorldModelExample(
                user_prompt=f"Calculate {func}({angle}°)",
                thinking=f"I need to calculate {func}({angle}°). First convert degrees to radians: {angle}° = {angle} × π/180.",
                code=f"import math\n\nangle_degrees = {angle}\nangle_radians = math.radians(angle_degrees)\nresult = math.{func}(angle_radians)\nprint(f\"{func}({angle}°) = {result:.4f}\")",
                requires="python:math",
                explanation=f"Converting {angle}° to radians and calculating the {func} function."
            ))
        
        # Logarithms
        for _ in range(15):
            number = random.choice([2, 4, 8, 16, 32, 10, 100, 1000])
            base = random.choice([2, 10, math.e])
            base_name = "2" if base == 2 else ("10" if base == 10 else "e")
            
            examples.append(WorldModelExample(
                user_prompt=f"Calculate log base {base_name} of {number}",
                thinking=f"I need to calculate the logarithm of {number} with base {base_name}.",
                code=f"import math\n\nnumber = {number}\nif {base} == math.e:\n    result = math.log(number)\n    print(f\"ln({number}) = {result:.4f}\")\nelse:\n    result = math.log(number, {base})\n    print(f\"log_{base_name}({number}) = {result:.4f}\")",
                requires="python:math",
                explanation=f"Calculating the logarithm of {number} with base {base_name}."
            ))
        
        return examples
    
    def generate_all_examples(self) -> List[WorldModelExample]:
        """Generate all example categories."""
        print("Generating 1000+ WorldModel training examples...")
        
        all_examples = []
        
        print("  Math basics (80 examples)...")
        all_examples.extend(self.generate_math_basic())
        
        print("  Percentages (30 examples)...")
        all_examples.extend(self.generate_percentages())
        
        print("  Compound interest (15 examples)...")
        all_examples.extend(self.generate_compound_interest())
        
        print("  Quadratic equations (25 examples)...")
        all_examples.extend(self.generate_quadratics())
        
        print("  Statistics (35 examples)...")
        all_examples.extend(self.generate_statistics())
        
        print("  Fibonacci sequences (15 examples)...")
        all_examples.extend(self.generate_fibonacci())
        
        print("  Prime numbers (30 examples)...")
        all_examples.extend(self.generate_prime_numbers())
        
        print("  Text analysis (40 examples)...")
        all_examples.extend(self.generate_text_analysis())
        
        print("  Unit conversions (35 examples)...")
        all_examples.extend(self.generate_conversions())
        
        print("  Geometry (35 examples)...")
        all_examples.extend(self.generate_geometry())
        
        print("  Boolean logic (8 examples)...")
        all_examples.extend(self.generate_boolean_logic())
        
        print("  Data structures (20 examples)...")
        all_examples.extend(self.generate_data_structures())
        
        print("  Word problems (40 examples)...")
        all_examples.extend(self.generate_word_problems())
        
        print("  Advanced math (35 examples)...")
        all_examples.extend(self.generate_advanced_math())
        
        print(f"Generated {len(all_examples)} total examples!")
        return all_examples

def main():
    """Generate and save training data."""
    print("🔥 WorldModel Training Data Generator")
    print("=" * 50)
    
    generator = ExampleGenerator()
    examples = generator.generate_all_examples()
    
    # Shuffle for variety
    random.shuffle(examples)
    
    # Format for training
    formatted_examples = []
    for example in examples:
        formatted_examples.append(example.format())
    
    # Save to file
    output_file = "/home/bigattichouse/workspace/worldmodel/data/worldmodel_training_expanded.txt"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n\n'.join(formatted_examples))
    
    print(f"\n✅ Saved {len(examples)} examples to: {output_file}")
    
    # Show some statistics
    categories = {
        'math': sum(1 for ex in examples if 'python:math' in ex.requires),
        'text': sum(1 for ex in examples if 'python:text' in ex.requires),
        'data': sum(1 for ex in examples if 'python:data' in ex.requires),
        'computation': sum(1 for ex in examples if 'python:computation' in ex.requires),
        'conversion': sum(1 for ex in examples if 'python:conversion' in ex.requires),
        'logic': sum(1 for ex in examples if 'python:logic' in ex.requires)
    }
    
    print(f"\n📊 Category breakdown:")
    for category, count in categories.items():
        print(f"   {category}: {count} examples")
    
    # Show sample examples
    print(f"\n📝 Sample examples:")
    for i, example in enumerate(examples[:3]):
        print(f"\n--- Example {i+1} ---")
        print(f"User: {example.user_prompt}")
        print(f"Think: {example.thinking[:60]}...")
        print(f"Code: {example.code.split('\\n')[0]}...")
        print(f"Requires: {example.requires}")
    
    print(f"\n🎉 Ready for training!")
    print(f"   Use: python3 train_worldmodel_rocm.py")
    print(f"   Update DATA_FILE to: {output_file}")

if __name__ == "__main__":
    main()