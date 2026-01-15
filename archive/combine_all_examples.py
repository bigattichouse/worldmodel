#!/usr/bin/env python3
"""
Combine All Training Examples
============================

Combines the original 950 examples with 396 new science/algorithm examples
for a total of 1,300+ comprehensive training examples.
"""

def main():
    print("🔗 Combining All Training Examples...")
    
    # Load original examples
    with open('/home/bigattichouse/workspace/worldmodel/data/worldmodel_training_1000.txt', 'r') as f:
        original_examples = f.read().strip()
    
    # Load science examples  
    with open('/home/bigattichouse/workspace/worldmodel/data/science_algorithm_examples.txt', 'r') as f:
        science_examples = f.read().strip()
    
    # Combine
    combined = original_examples + '\n\n' + science_examples
    
    # Save combined dataset
    output_file = '/home/bigattichouse/workspace/worldmodel/data/worldmodel_training_complete.txt'
    with open(output_file, 'w') as f:
        f.write(combined)
    
    # Count examples
    example_count = combined.count('User:')
    
    print(f"✅ Combined dataset created:")
    print(f"   Original examples: 950")
    print(f"   Science examples: 396") 
    print(f"   Total examples: {example_count}")
    print(f"   File: {output_file}")
    print(f"   Size: {len(combined) / 1024:.1f} KB")
    
    # Show category breakdown
    categories = {
        'math': combined.count('python:math'),
        'text': combined.count('python:text'), 
        'chemistry': combined.count('python:chemistry'),
        'physics': combined.count('python:physics'),
        'electrical': combined.count('python:electrical'),
        'rf': combined.count('python:rf'),
        'statistics': combined.count('python:statistics'),
        'algorithms': combined.count('python:algorithms'),
        'conversion': combined.count('python:conversion'),
        'data': combined.count('python:data'),
        'logic': combined.count('python:logic'),
        'computation': combined.count('python:computation')
    }
    
    print(f"\n📊 Complete dataset categories:")
    for category, count in sorted(categories.items(), key=lambda x: x[1], reverse=True):
        if count > 0:
            print(f"   {category}: {count} examples")
    
    print(f"\n🎯 This comprehensive dataset covers:")
    print(f"   • Basic math & percentages")
    print(f"   • String algorithms (counting R's in strawberry, etc.)")
    print(f"   • Temperature conversions (C↔F, Kelvin)")
    print(f"   • Chemistry (molarity, gas laws, stoichiometry)")
    print(f"   • Physics (projectile motion, waves, RF)")
    print(f"   • Electrical (Ohm's law, LC circuits, power)")
    print(f"   • RF engineering (frequency↔wavelength)")
    print(f"   • Statistics (std dev, median)")
    print(f"   • Computer algorithms (binary search, sorting)")
    
    return output_file

if __name__ == "__main__":
    output = main()
    print(f"\n🚀 Ready for training with: {output}")
    print(f"💡 Update your training script to use this complete dataset!")