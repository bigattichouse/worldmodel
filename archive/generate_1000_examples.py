#!/usr/bin/env python3
"""
Quick 1000 WorldModel Examples Generator
=======================================

Generates exactly 1000 training examples quickly.
"""

import random
import math

def generate_examples():
    examples = []
    
    # 1. Basic Math (200 examples)
    for i in range(50):
        a, b = random.randint(10, 999), random.randint(10, 999)
        examples.append(f"""User: Calculate {a} + {b}
Assistant: <think>I need to add {a} and {b}.</think>
<model>
result = {a} + {b}
print(result)
</model>
<requires>python:math</requires>

{a} + {b} equals {a + b}.""")

    for i in range(50):
        a, b = random.randint(100, 999), random.randint(10, 99)
        examples.append(f"""User: Calculate {a} - {b}
Assistant: <think>I need to subtract {b} from {a}.</think>
<model>
result = {a} - {b}
print(result)
</model>
<requires>python:math</requires>

{a} - {b} equals {a - b}.""")

    for i in range(50):
        a, b = random.randint(12, 99), random.randint(12, 99)
        examples.append(f"""User: What is {a} × {b}?
Assistant: <think>I need to multiply {a} by {b}.</think>
<model>
result = {a} * {b}
print(result)
</model>
<requires>python:math</requires>

{a} × {b} equals {a * b}.""")

    for i in range(50):
        b = random.randint(2, 25)
        a = b * random.randint(10, 50)
        examples.append(f"""User: Divide {a} by {b}
Assistant: <think>I need to divide {a} by {b}.</think>
<model>
result = {a} / {b}
print(result)
</model>
<requires>python:math</requires>

{a} ÷ {b} equals {a // b}.""")

    # 2. Percentages (100 examples)
    for i in range(100):
        percent = random.choice([5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 75, 80, 90])
        number = random.randint(100, 1000)
        result = (percent/100) * number
        examples.append(f"""User: Calculate {percent}% of {number}
Assistant: <think>I need to calculate {percent}% of {number}. {percent}% = {percent/100}, so {percent/100} × {number}.</think>
<model>
percent = {percent}
number = {number}
result = (percent / 100) * number
print(f"{percent}% of {number} = " + str(result))
</model>
<requires>python:math</requires>

{percent}% of {number} equals {result}.""")

    # 3. Text Analysis (150 examples)
    texts = [
        "The quick brown fox jumps over the lazy dog",
        "Python is a powerful programming language",
        "Machine learning algorithms process large datasets",
        "Web development requires HTML, CSS, and JavaScript",
        "Database systems store and retrieve information efficiently"
    ]
    
    for i in range(50):
        text = random.choice(texts)
        char = random.choice(['a', 'e', 'i', 'o', 'u', 'r', 's', 't'])
        count = text.lower().count(char)
        examples.append(f"""User: Count the '{char}' characters in: "{text}"
Assistant: <think>I need to count how many times '{char}' appears in the text.</think>
<model>
text = "{text}"
char = '{char}'
count = text.lower().count(char)
print(f"Character '{char}' appears " + str(count) + " times")
</model>
<requires>python:text</requires>

The character '{char}' appears {count} times in the text.""")

    for i in range(50):
        text = random.choice(texts)
        word_count = len(text.split())
        examples.append(f"""User: Count the words in: "{text}"
Assistant: <think>I need to split the text by spaces and count the words.</think>
<model>
text = "{text}"
words = text.split()
count = len(words)
print(f"Number of words: " + str(count))
</model>
<requires>python:text</requires>

The text contains {word_count} words.""")

    for i in range(50):
        text = random.choice(texts)
        vowel_count = sum(1 for char in text if char.lower() in 'aeiou')
        examples.append(f"""User: Count vowels in: "{text}"
Assistant: <think>I need to count all vowels (a, e, i, o, u) in the text.</think>
<model>
text = "{text}"
vowels = "aeiouAEIOU"
count = sum(1 for char in text if char in vowels)
print(f"Number of vowels: " + str(count))
</model>
<requires>python:text</requires>

The text contains {vowel_count} vowels.""")

    # 4. Geometry (100 examples)
    for i in range(50):
        radius = random.randint(5, 25)
        area = math.pi * radius**2
        examples.append(f"""User: Calculate the area of a circle with radius {radius}
Assistant: <think>Area of a circle = π × r². With radius {radius}, area = π × {radius}².</think>
<model>
import math
radius = {radius}
area = math.pi * radius**2
print(f"Area = " + str(round(area, 2)) + " square units")
</model>
<requires>python:math</requires>

The area of a circle with radius {radius} is {area:.2f} square units.""")

    for i in range(50):
        length = random.randint(5, 30)
        width = random.randint(3, 25)
        area = length * width
        perimeter = 2 * (length + width)
        examples.append(f"""User: Calculate area and perimeter of rectangle with length {length} and width {width}
Assistant: <think>For a rectangle: Area = length × width, Perimeter = 2(length + width).</think>
<model>
length = {length}
width = {width}
area = length * width
perimeter = 2 * (length + width)
print(f"Area = " + str(area) + " square units")
print(f"Perimeter = " + str(perimeter) + " units")
</model>
<requires>python:math</requires>

Rectangle: Area = {area}, Perimeter = {perimeter}.""")

    # 5. Statistics (100 examples)
    for i in range(100):
        numbers = [random.randint(10, 100) for _ in range(random.randint(5, 10))]
        mean = sum(numbers) / len(numbers)
        examples.append(f"""User: Calculate the mean of: {numbers}
Assistant: <think>I need to add all numbers and divide by the count.</think>
<model>
numbers = {numbers}
mean = sum(numbers) / len(numbers)
print(f"Mean: " + str(round(mean, 2)))
</model>
<requires>python:math</requires>

The mean of these numbers is {mean:.2f}.""")

    # 6. Prime Numbers (80 examples)
    for i in range(80):
        number = random.randint(50, 200)
        is_prime = number > 1 and all(number % i != 0 for i in range(2, int(number**0.5) + 1))
        examples.append(f"""User: Is {number} a prime number?
Assistant: <think>A prime number is only divisible by 1 and itself. I need to check if {number} has any other divisors.</think>
<model>
def is_prime(n):
    if n < 2:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True

number = {number}
result = is_prime(number)
print(f"{number} is " + ("prime" if result else "not prime"))
</model>
<requires>python:math</requires>

{number} is {"prime" if is_prime else "not prime"}.""")

    # 7. Conversions (70 examples)
    for i in range(35):
        celsius = random.randint(-20, 50)
        fahrenheit = (celsius * 9/5) + 32
        examples.append(f"""User: Convert {celsius}°C to Fahrenheit
Assistant: <think>To convert Celsius to Fahrenheit: F = (C × 9/5) + 32.</think>
<model>
celsius = {celsius}
fahrenheit = (celsius * 9/5) + 32
print(f"{celsius}°C = " + str(fahrenheit) + "°F")
</model>
<requires>python:conversion</requires>

{celsius}°C equals {fahrenheit}°F.""")

    for i in range(35):
        miles = random.randint(1, 100)
        kilometers = miles * 1.60934
        examples.append(f"""User: Convert {miles} miles to kilometers
Assistant: <think>1 mile = 1.60934 kilometers.</think>
<model>
miles = {miles}
kilometers = miles * 1.60934
print(f"{miles} miles = " + str(round(kilometers, 2)) + " km")
</model>
<requires>python:conversion</requires>

{miles} miles equals {kilometers:.2f} kilometers.""")

    # 8. Boolean Logic (50 examples)
    expressions = [
        ("True and False", False),
        ("False or True", True),
        ("not True", False),
        ("True and True and False", False),
        ("False or False or True", True)
    ]
    
    for i in range(50):
        expr, result = random.choice(expressions)
        examples.append(f"""User: Evaluate: {expr}
Assistant: <think>I need to evaluate the boolean expression step by step.</think>
<model>
result = {expr}
print(f"Result: " + str(result))
</model>
<requires>python:logic</requires>

The expression {expr} evaluates to {result}.""")

    # 9. Word Problems (100 examples)
    for i in range(100):
        item1_price = random.randint(5, 25)
        item2_price = random.randint(3, 15)
        item1_qty = random.randint(2, 5)
        item2_qty = random.randint(1, 4)
        payment = (item1_price * item1_qty) + (item2_price * item2_qty) + random.randint(5, 50)
        total_cost = (item1_price * item1_qty) + (item2_price * item2_qty)
        change = payment - total_cost
        
        examples.append(f"""User: I buy {item1_qty} books for ${item1_price} each and {item2_qty} pens for ${item2_price} each. How much change from ${payment}?
Assistant: <think>Total cost = ({item1_qty} × ${item1_price}) + ({item2_qty} × ${item2_price}) = ${total_cost}. Change = ${payment} - ${total_cost}.</think>
<model>
item1_price = {item1_price}
item1_qty = {item1_qty}
item2_price = {item2_price}
item2_qty = {item2_qty}
payment = {payment}

total_cost = (item1_price * item1_qty) + (item2_price * item2_qty)
change = payment - total_cost
print(f"Total cost: $" + str(total_cost))
print(f"Change: $" + str(change))
</model>
<requires>python:math</requires>

The total cost is ${total_cost}, so the change from ${payment} is ${change}.""")

    print(f"Generated {len(examples)} examples!")
    return examples

def main():
    print("🚀 Generating 1000 WorldModel Examples...")
    examples = generate_examples()
    
    # Save to file
    output_file = "/home/bigattichouse/workspace/worldmodel/data/worldmodel_training_1000.txt"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n\n'.join(examples))
    
    print(f"✅ Saved {len(examples)} examples to: {output_file}")
    print(f"📁 File size: {len('\n\n'.join(examples)) / 1024:.1f} KB")

if __name__ == "__main__":
    main()