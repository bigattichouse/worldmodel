#!/usr/bin/env python3
"""
Science & Algorithm Examples Generator
=====================================

Generates hundreds of examples focusing on:
- String algorithms (counting characters, sorting)
- Temperature conversions
- Chemistry (molarity, stoichiometry)
- Physics (projectile motion, waves, electricity)
- RF/Electronics (wavelength, LC circuits)
- Statistical algorithms
- Computer science algorithms
"""

import random
import math
from typing import List

def generate_string_algorithms() -> List[str]:
    """Generate string algorithm examples."""
    examples = []
    
    # Character counting examples
    words = [
        "strawberry", "programming", "algorithm", "statistics", "chemistry",
        "physics", "mathematics", "engineering", "computer", "science",
        "frequency", "analysis", "laboratory", "experiment", "research",
        "university", "professor", "student", "learning", "knowledge"
    ]
    
    chars = ['r', 's', 't', 'a', 'e', 'i', 'o', 'n', 'm', 'l', 'g']
    
    for _ in range(30):
        word = random.choice(words)
        char = random.choice(chars)
        count = word.lower().count(char)
        
        examples.append(f"""User: Count the number of '{char}' characters in the word "{word}"
Assistant: <think>I need to count how many times the character '{char}' appears in "{word}". I'll iterate through each character and count matches.</think>
<model>
word = "{word}"
char = '{char}'
count = 0
for c in word.lower():
    if c == char:
        count += 1
print(f"The character '{char}' appears {{count}} times in '{word}'")
</model>
<requires>python:text</requires>

The character '{char}' appears {count} times in the word "{word}".""")
    
    # String sorting examples
    word_lists = [
        ["apple", "banana", "cherry", "date"],
        ["python", "java", "javascript", "rust"],
        ["red", "green", "blue", "yellow"],
        ["north", "south", "east", "west"],
        ["spring", "summer", "autumn", "winter"]
    ]
    
    for _ in range(20):
        words = random.choice(word_lists)
        random.shuffle(words)
        original = words.copy()
        sorted_words = sorted(words)
        
        examples.append(f"""User: Sort these words alphabetically: {original}
Assistant: <think>I need to sort the list {original} in alphabetical order.</think>
<model>
words = {original}
sorted_words = sorted(words)
print(f"Original: {{words}}")
print(f"Sorted: {{sorted_words}}")
</model>
<requires>python:data</requires>

Sorted alphabetically: {sorted_words}.""")
    
    # String reversal
    for _ in range(15):
        word = random.choice(words)
        reversed_word = word[::-1]
        
        examples.append(f"""User: Reverse the string "{word}"
Assistant: <think>I need to reverse the string "{word}" by reading it backwards.</think>
<model>
text = "{word}"
reversed_text = text[::-1]
print(f"Original: {{text}}")
print(f"Reversed: {{reversed_text}}")
</model>
<requires>python:text</requires>

The string "{word}" reversed is "{reversed_word}".""")
    
    return examples

def generate_temperature_conversions() -> List[str]:
    """Generate temperature conversion examples."""
    examples = []
    
    # Celsius to Fahrenheit
    for _ in range(25):
        celsius = random.randint(-40, 100)
        fahrenheit = (celsius * 9/5) + 32
        
        examples.append(f"""User: Convert {celsius}°C to Fahrenheit
Assistant: <think>To convert Celsius to Fahrenheit, I use the formula: F = (C × 9/5) + 32. So F = ({celsius} × 9/5) + 32.</think>
<model>
celsius = {celsius}
fahrenheit = (celsius * 9/5) + 32
print(f"{{celsius}}°C = {{fahrenheit}}°F")
</model>
<requires>python:conversion</requires>

{celsius}°C equals {fahrenheit}°F.""")
    
    # Fahrenheit to Celsius
    for _ in range(25):
        fahrenheit = random.randint(0, 200)
        celsius = (fahrenheit - 32) * 5/9
        
        examples.append(f"""User: Convert {fahrenheit}°F to Celsius
Assistant: <think>To convert Fahrenheit to Celsius, I use the formula: C = (F - 32) × 5/9. So C = ({fahrenheit} - 32) × 5/9.</think>
<model>
fahrenheit = {fahrenheit}
celsius = (fahrenheit - 32) * 5/9
print(f"{{fahrenheit}}°F = {{celsius:.2f}}°C")
</model>
<requires>python:conversion</requires>

{fahrenheit}°F equals {celsius:.2f}°C.""")
    
    # Kelvin conversions
    for _ in range(15):
        if random.choice([True, False]):
            celsius = random.randint(-50, 200)
            kelvin = celsius + 273.15
            examples.append(f"""User: Convert {celsius}°C to Kelvin
Assistant: <think>To convert Celsius to Kelvin, I add 273.15. So K = {celsius} + 273.15.</think>
<model>
celsius = {celsius}
kelvin = celsius + 273.15
print(f"{{celsius}}°C = {{kelvin}}K")
</model>
<requires>python:conversion</requires>

{celsius}°C equals {kelvin}K.""")
        else:
            kelvin = random.randint(273, 573)
            celsius = kelvin - 273.15
            examples.append(f"""User: Convert {kelvin}K to Celsius
Assistant: <think>To convert Kelvin to Celsius, I subtract 273.15. So C = {kelvin} - 273.15.</think>
<model>
kelvin = {kelvin}
celsius = kelvin - 273.15
print(f"{{kelvin}}K = {{celsius}}°C")
</model>
<requires>python:conversion</requires>

{kelvin}K equals {celsius}°C.""")
    
    return examples

def generate_chemistry_examples() -> List[str]:
    """Generate chemistry calculation examples."""
    examples = []
    
    # Molarity calculations
    for _ in range(30):
        moles = random.uniform(0.1, 5.0)
        volume_l = random.uniform(0.1, 2.0)
        molarity = moles / volume_l
        
        examples.append(f"""User: Calculate the molarity of a solution with {moles:.2f} moles of solute in {volume_l:.2f} L of solution
Assistant: <think>Molarity (M) = moles of solute / liters of solution. M = {moles:.2f} mol / {volume_l:.2f} L.</think>
<model>
moles = {moles:.2f}
volume_l = {volume_l:.2f}
molarity = moles / volume_l
print(f"Molarity = {{moles}} mol / {{volume_l}} L = {{molarity:.3f}} M")
</model>
<requires>python:chemistry</requires>

The molarity is {molarity:.3f} M.""")
    
    # Molar mass calculations
    compounds = [
        ("H2O", 18.015, "water"),
        ("CO2", 44.01, "carbon dioxide"),
        ("NaCl", 58.44, "sodium chloride"),
        ("C6H12O6", 180.16, "glucose"),
        ("CaCO3", 100.09, "calcium carbonate"),
        ("H2SO4", 98.08, "sulfuric acid")
    ]
    
    for _ in range(20):
        compound, molar_mass, name = random.choice(compounds)
        mass = random.uniform(1.0, 100.0)
        moles = mass / molar_mass
        
        examples.append(f"""User: How many moles are in {mass:.1f} g of {compound} ({name})?
Assistant: <think>Number of moles = mass / molar mass. Molar mass of {compound} is {molar_mass} g/mol. So moles = {mass:.1f} g / {molar_mass} g/mol.</think>
<model>
mass_g = {mass:.1f}
molar_mass = {molar_mass}  # g/mol for {compound}
moles = mass_g / molar_mass
print(f"Moles of {compound} = {{mass_g}} g / {{molar_mass}} g/mol = {{moles:.4f}} mol")
</model>
<requires>python:chemistry</requires>

{mass:.1f} g of {compound} contains {moles:.4f} moles.""")
    
    # Gas law calculations (PV=nRT)
    for _ in range(15):
        pressure = random.uniform(0.5, 5.0)  # atm
        volume = random.uniform(1.0, 50.0)   # L
        temperature = random.uniform(250, 400)  # K
        R = 0.0821  # L·atm/(mol·K)
        moles = (pressure * volume) / (R * temperature)
        
        examples.append(f"""User: Calculate moles of gas at {pressure:.2f} atm, {volume:.1f} L, and {temperature:.0f} K
Assistant: <think>Using ideal gas law PV = nRT. Solving for n: n = PV/(RT). Where R = 0.0821 L·atm/(mol·K).</think>
<model>
P = {pressure:.2f}  # atm
V = {volume:.1f}     # L
T = {temperature:.0f}     # K
R = 0.0821    # L·atm/(mol·K)
n = (P * V) / (R * T)
print(f"Moles = ({{P}} × {{V}}) / ({{R}} × {{T}}) = {{n:.4f}} mol")
</model>
<requires>python:chemistry</requires>

The gas contains {moles:.4f} moles.""")
    
    return examples

def generate_physics_examples() -> List[str]:
    """Generate physics calculation examples."""
    examples = []
    
    # Projectile motion - time to hit ground
    for _ in range(25):
        height = random.uniform(5.0, 100.0)
        g = 9.81  # m/s²
        time = math.sqrt(2 * height / g)
        
        examples.append(f"""User: How long does it take for an object dropped from {height:.1f} m to hit the ground?
Assistant: <think>For free fall from height h, time t = √(2h/g) where g = 9.81 m/s². So t = √(2 × {height:.1f} / 9.81).</think>
<model>
import math
height = {height:.1f}  # meters
g = 9.81        # m/s²
time = math.sqrt(2 * height / g)
print(f"Time to fall {{height}} m = √(2 × {{height}} / {{g}}) = {{time:.2f}} seconds")
</model>
<requires>python:physics</requires>

It takes {time:.2f} seconds to fall from {height:.1f} m.""")
    
    # Projectile motion with initial velocity
    for _ in range(20):
        v0 = random.uniform(10.0, 50.0)  # initial velocity m/s
        angle = random.uniform(20.0, 70.0)  # degrees
        g = 9.81
        angle_rad = math.radians(angle)
        time_flight = 2 * v0 * math.sin(angle_rad) / g
        max_height = (v0 * math.sin(angle_rad))**2 / (2 * g)
        
        examples.append(f"""User: Calculate flight time for projectile launched at {v0:.1f} m/s at {angle:.0f}° angle
Assistant: <think>Flight time for projectile: t = 2v₀sin(θ)/g. With v₀ = {v0:.1f} m/s, θ = {angle:.0f}°, g = 9.81 m/s².</think>
<model>
import math
v0 = {v0:.1f}        # m/s
angle_deg = {angle:.0f}   # degrees
g = 9.81          # m/s²

angle_rad = math.radians(angle_deg)
flight_time = 2 * v0 * math.sin(angle_rad) / g
print(f"Flight time = 2 × {{v0}} × sin({{angle_deg}}°) / {{g}} = {{flight_time:.2f}} seconds")
</model>
<requires>python:physics</requires>

The projectile flight time is {time_flight:.2f} seconds.""")
    
    # Wave calculations
    for _ in range(20):
        frequency = random.uniform(100, 10000)  # Hz
        speed_of_light = 3e8  # m/s
        wavelength = speed_of_light / frequency
        
        examples.append(f"""User: Calculate wavelength for electromagnetic wave with frequency {frequency:.0f} Hz
Assistant: <think>Wavelength λ = c/f where c = 3×10⁸ m/s and f = {frequency:.0f} Hz. So λ = 3×10⁸ / {frequency:.0f}.</think>
<model>
frequency = {frequency:.0f}  # Hz
c = 3e8           # speed of light in m/s
wavelength = c / frequency
print(f"Wavelength = {{c}} / {{frequency}} = {{wavelength:.6f}} m")
print(f"Wavelength = {{wavelength*1000:.3f}} mm")
</model>
<requires>python:physics</requires>

The wavelength is {wavelength:.6f} m or {wavelength*1000:.3f} mm.""")
    
    return examples

def generate_electrical_examples() -> List[str]:
    """Generate electrical engineering examples."""
    examples = []
    
    # Ohm's law calculations
    for _ in range(25):
        case = random.choice(['voltage', 'current', 'resistance'])
        if case == 'voltage':
            current = random.uniform(0.1, 10.0)
            resistance = random.uniform(10, 1000)
            voltage = current * resistance
            examples.append(f"""User: Calculate voltage when current is {current:.2f} A and resistance is {resistance:.0f} Ω
Assistant: <think>Using Ohm's law: V = I × R. V = {current:.2f} A × {resistance:.0f} Ω.</think>
<model>
current = {current:.2f}    # Amperes
resistance = {resistance:.0f} # Ohms
voltage = current * resistance
print(f"Voltage = {{current}} A × {{resistance}} Ω = {{voltage:.1f}} V")
</model>
<requires>python:electrical</requires>

The voltage is {voltage:.1f} V.""")
        elif case == 'current':
            voltage = random.uniform(5, 50)
            resistance = random.uniform(10, 1000)
            current = voltage / resistance
            examples.append(f"""User: Calculate current when voltage is {voltage:.1f} V and resistance is {resistance:.0f} Ω
Assistant: <think>Using Ohm's law: I = V / R. I = {voltage:.1f} V / {resistance:.0f} Ω.</think>
<model>
voltage = {voltage:.1f}      # Volts
resistance = {resistance:.0f}   # Ohms
current = voltage / resistance
print(f"Current = {{voltage}} V / {{resistance}} Ω = {{current:.4f}} A")
</model>
<requires>python:electrical</requires>

The current is {current:.4f} A.""")
        else:
            voltage = random.uniform(5, 50)
            current = random.uniform(0.1, 5.0)
            resistance = voltage / current
            examples.append(f"""User: Calculate resistance when voltage is {voltage:.1f} V and current is {current:.2f} A
Assistant: <think>Using Ohm's law: R = V / I. R = {voltage:.1f} V / {current:.2f} A.</think>
<model>
voltage = {voltage:.1f}    # Volts
current = {current:.2f}    # Amperes
resistance = voltage / current
print(f"Resistance = {{voltage}} V / {{current}} A = {{resistance:.1f}} Ω")
</model>
<requires>python:electrical</requires>

The resistance is {resistance:.1f} Ω.""")
    
    # LC circuit resonant frequency
    for _ in range(20):
        L = random.uniform(1e-6, 1e-3)  # Inductance in H
        C = random.uniform(1e-12, 1e-6)  # Capacitance in F
        f = 1 / (2 * math.pi * math.sqrt(L * C))
        
        examples.append(f"""User: Calculate resonant frequency of LC circuit with L = {L:.2e} H and C = {C:.2e} F
Assistant: <think>Resonant frequency f = 1/(2π√(LC)). With L = {L:.2e} H and C = {C:.2e} F.</think>
<model>
import math
L = {L:.2e}  # Inductance in Henries
C = {C:.2e}  # Capacitance in Farads
f = 1 / (2 * math.pi * math.sqrt(L * C))
print(f"Resonant frequency = 1 / (2π√(LC))")
print(f"f = 1 / (2π√({{L:.2e}} × {{C:.2e}})) = {{f:.2e}} Hz")
print(f"f = {{f/1000:.1f}} kHz")
</model>
<requires>python:electrical</requires>

The resonant frequency is {f:.2e} Hz ({f/1000:.1f} kHz).""")
    
    # Power calculations
    for _ in range(15):
        voltage = random.uniform(120, 240)
        current = random.uniform(1, 20)
        power = voltage * current
        energy_1h = power / 1000  # kWh for 1 hour
        
        examples.append(f"""User: Calculate power consumption for device at {voltage:.0f} V drawing {current:.1f} A
Assistant: <think>Power P = V × I. P = {voltage:.0f} V × {current:.1f} A. Energy in 1 hour = P × 1h.</think>
<model>
voltage = {voltage:.0f}  # Volts
current = {current:.1f}  # Amperes
power = voltage * current
energy_1h = power / 1000  # kWh for 1 hour
print(f"Power = {{voltage}} V × {{current}} A = {{power:.0f}} W")
print(f"Energy in 1 hour = {{energy_1h:.2f}} kWh")
</model>
<requires>python:electrical</requires>

The power consumption is {power:.0f} W ({energy_1h:.2f} kWh per hour).""")
    
    return examples

def generate_rf_examples() -> List[str]:
    """Generate RF and wavelength examples."""
    examples = []
    
    # RF frequency to wavelength
    frequencies = [
        (50e6, "50 MHz", "VHF"),
        (100e6, "100 MHz", "FM radio"),
        (400e6, "400 MHz", "UHF"),
        (900e6, "900 MHz", "GSM"),
        (2.4e9, "2.4 GHz", "WiFi"),
        (5.8e9, "5.8 GHz", "WiFi"),
        (10e9, "10 GHz", "X-band"),
        (24e9, "24 GHz", "K-band")
    ]
    
    for freq_hz, freq_str, band in frequencies:
        c = 3e8
        wavelength = c / freq_hz
        
        examples.append(f"""User: Calculate wavelength for {freq_str} ({band}) radio frequency
Assistant: <think>Wavelength λ = c/f where c = 3×10⁸ m/s. For f = {freq_hz:.0e} Hz, λ = 3×10⁸ / {freq_hz:.0e}.</think>
<model>
frequency = {freq_hz:.0e}  # Hz
c = 3e8               # speed of light m/s
wavelength = c / frequency
print(f"Wavelength = {{c}} / {{frequency}} = {{wavelength:.3f}} m")
if wavelength >= 1:
    print(f"Wavelength = {{wavelength:.2f}} meters")
elif wavelength >= 0.01:
    print(f"Wavelength = {{wavelength*100:.1f}} cm")
else:
    print(f"Wavelength = {{wavelength*1000:.1f}} mm")
</model>
<requires>python:rf</requires>

The wavelength for {freq_str} is {wavelength:.3f} m.""")
    
    # Antenna length calculations (quarter wave)
    for _ in range(15):
        frequency = random.choice([27e6, 50e6, 144e6, 433e6, 915e6, 2.4e9])
        c = 3e8
        wavelength = c / frequency
        quarter_wave = wavelength / 4
        
        freq_mhz = frequency / 1e6 if frequency < 1e9 else frequency / 1e9
        unit = "MHz" if frequency < 1e9 else "GHz"
        
        examples.append(f"""User: Calculate quarter-wave antenna length for {freq_mhz:.0f} {unit}
Assistant: <think>Quarter-wave antenna length = λ/4. First find λ = c/f, then divide by 4. λ = 3×10⁸ / {frequency:.0e}.</think>
<model>
frequency = {frequency:.0e}  # Hz
c = 3e8               # speed of light
wavelength = c / frequency
quarter_wave = wavelength / 4
print(f"Wavelength = {{wavelength:.3f}} m")
print(f"Quarter-wave length = {{quarter_wave:.3f}} m")
if quarter_wave >= 0.01:
    print(f"Quarter-wave = {{quarter_wave*100:.1f}} cm")
else:
    print(f"Quarter-wave = {{quarter_wave*1000:.1f}} mm")
</model>
<requires>python:rf</requires>

Quarter-wave antenna for {freq_mhz:.0f} {unit} is {quarter_wave:.3f} m ({quarter_wave*100:.1f} cm).""")
    
    return examples

def generate_statistical_examples() -> List[str]:
    """Generate statistical algorithm examples."""
    examples = []
    
    # Standard deviation
    for _ in range(20):
        data = [random.randint(10, 100) for _ in range(random.randint(6, 12))]
        mean = sum(data) / len(data)
        variance = sum((x - mean)**2 for x in data) / len(data)
        std_dev = math.sqrt(variance)
        
        examples.append(f"""User: Calculate standard deviation of: {data}
Assistant: <think>Standard deviation = √(Σ(x-μ)²/n). First find mean μ, then calculate variance, then square root.</think>
<model>
import math
data = {data}
n = len(data)
mean = sum(data) / n
variance = sum((x - mean)**2 for x in data) / n
std_dev = math.sqrt(variance)
print(f"Data: {{data}}")
print(f"Mean = {{mean:.2f}}")
print(f"Variance = {{variance:.2f}}")
print(f"Standard deviation = {{std_dev:.3f}}")
</model>
<requires>python:statistics</requires>

The standard deviation is {std_dev:.3f}.""")
    
    # Median calculation
    for _ in range(15):
        data = [random.randint(1, 100) for _ in range(random.randint(7, 15))]
        sorted_data = sorted(data)
        n = len(sorted_data)
        if n % 2 == 1:
            median = sorted_data[n // 2]
        else:
            median = (sorted_data[n // 2 - 1] + sorted_data[n // 2]) / 2
        
        examples.append(f"""User: Find the median of: {data}
Assistant: <think>To find median, I need to sort the data and find the middle value(s). For {len(data)} values, median is the middle value.</think>
<model>
data = {data}
sorted_data = sorted(data)
n = len(sorted_data)
if n % 2 == 1:
    median = sorted_data[n // 2]
else:
    median = (sorted_data[n // 2 - 1] + sorted_data[n // 2]) / 2
print(f"Original data: {{data}}")
print(f"Sorted data: {{sorted_data}}")
print(f"Median = {{median}}")
</model>
<requires>python:statistics</requires>

The median is {median}.""")
    
    return examples

def generate_algorithm_examples() -> List[str]:
    """Generate computer science algorithm examples."""
    examples = []
    
    # Binary search
    for _ in range(10):
        arr = sorted([random.randint(1, 100) for _ in range(10)])
        target = random.choice(arr + [random.randint(1, 100)])
        
        examples.append(f"""User: Implement binary search to find {target} in sorted array {arr}
Assistant: <think>Binary search repeatedly divides the search space in half. Compare target with middle element and eliminate half the array.</think>
<model>
def binary_search(arr, target):
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1

arr = {arr}
target = {target}
result = binary_search(arr, target)
if result != -1:
    print(f"Found {{target}} at index {{result}}")
else:
    print(f"{{target}} not found in array")
</model>
<requires>python:algorithms</requires>

Binary search for {target} in the array.""")
    
    # Bubble sort
    for _ in range(8):
        arr = [random.randint(1, 50) for _ in range(8)]
        
        examples.append(f"""User: Sort this array using bubble sort: {arr}
Assistant: <think>Bubble sort repeatedly steps through the list, compares adjacent elements and swaps them if they're in wrong order.</think>
<model>
def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr

arr = {arr}
print(f"Original: {{arr}}")
sorted_arr = bubble_sort(arr.copy())
print(f"Sorted: {{sorted_arr}}")
</model>
<requires>python:algorithms</requires>

Bubble sort implementation for the given array.""")
    
    return examples

def main():
    """Generate all science and algorithm examples."""
    print("🔬 Generating Science & Algorithm Examples...")
    
    all_examples = []
    
    print("  String algorithms (65 examples)...")
    all_examples.extend(generate_string_algorithms())
    
    print("  Temperature conversions (65 examples)...")
    all_examples.extend(generate_temperature_conversions())
    
    print("  Chemistry calculations (65 examples)...")
    all_examples.extend(generate_chemistry_examples())
    
    print("  Physics calculations (65 examples)...")
    all_examples.extend(generate_physics_examples())
    
    print("  Electrical engineering (60 examples)...")
    all_examples.extend(generate_electrical_examples())
    
    print("  RF and wavelength (23 examples)...")
    all_examples.extend(generate_rf_examples())
    
    print("  Statistical algorithms (35 examples)...")
    all_examples.extend(generate_statistical_examples())
    
    print("  Computer algorithms (18 examples)...")
    all_examples.extend(generate_algorithm_examples())
    
    # Shuffle for variety
    random.shuffle(all_examples)
    
    print(f"\n✅ Generated {len(all_examples)} science & algorithm examples!")
    
    # Save to file
    output_file = "/home/bigattichouse/workspace/worldmodel/data/science_algorithm_examples.txt"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n\n'.join(all_examples))
    
    print(f"📁 Saved to: {output_file}")
    print(f"📏 File size: {len('\\n\\n'.join(all_examples)) / 1024:.1f} KB")
    
    # Show sample
    print(f"\n📝 Sample examples:")
    for i, example in enumerate(all_examples[:3]):
        lines = example.split('\\n')
        print(f"\n--- Example {i+1} ---")
        print(lines[0])  # User prompt
        print(lines[1][:80] + "...")  # Think snippet
    
    # Category breakdown
    categories = {
        'text': sum(1 for ex in all_examples if 'python:text' in ex),
        'conversion': sum(1 for ex in all_examples if 'python:conversion' in ex),
        'chemistry': sum(1 for ex in all_examples if 'python:chemistry' in ex),
        'physics': sum(1 for ex in all_examples if 'python:physics' in ex),
        'electrical': sum(1 for ex in all_examples if 'python:electrical' in ex),
        'rf': sum(1 for ex in all_examples if 'python:rf' in ex),
        'statistics': sum(1 for ex in all_examples if 'python:statistics' in ex),
        'algorithms': sum(1 for ex in all_examples if 'python:algorithms' in ex)
    }
    
    print(f"\n📊 Category breakdown:")
    for category, count in categories.items():
        print(f"   {category}: {count} examples")
    
    return output_file

if __name__ == "__main__":
    main()