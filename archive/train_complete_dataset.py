#!/usr/bin/env python3
"""
Production Training with Complete Dataset
=========================================

Uses the complete 1,346-example dataset covering:
- Math & algorithms
- String processing (counting R's in strawberry)
- Temperature conversions
- Chemistry (molarity, gas laws)
- Physics (projectile motion, waves)
- Electrical engineering (LC circuits, Ohm's law)
- RF engineering (frequency to wavelength)
- Statistics and computer science algorithms
"""

# Import the working production trainer and update the dataset path
import sys
import os

# Update the dataset file to use complete dataset
def update_data_file():
    """Update the training script to use complete dataset."""
    script_content = open('train_worldmodel_production.py', 'r').read()
    
    # Replace dataset path
    updated_content = script_content.replace(
        'DATA_FILE = "/home/bigattichouse/workspace/worldmodel/data/worldmodel_training_1000.txt"',
        'DATA_FILE = "/home/bigattichouse/workspace/worldmodel/data/worldmodel_training_complete.txt"'
    ).replace(
        'OUTPUT_DIR = "./worldmodel_production_training"',
        'OUTPUT_DIR = "./worldmodel_complete_training"'
    ).replace(
        'print("=== WorldModel Production Training ===")',
        'print("=== WorldModel Complete Dataset Training ===")'
    ).replace(
        'print("🔥 WorldModel Production Training")',
        'print("🔥 WorldModel Complete Dataset Training (1,346 examples)")'
    )
    
    return updated_content

if __name__ == "__main__":
    print("🧪 WorldModel Complete Dataset Training")
    print("=" * 60)
    print("📊 Dataset: 1,346 examples covering:")
    print("   • Math & basic algorithms (680 examples)")
    print("   • Text processing (195 examples)")  
    print("   • Temperature conversions (135 examples)")
    print("   • Chemistry calculations (65 examples)")
    print("   • Physics problems (65 examples)")
    print("   • Electrical engineering (60 examples)")
    print("   • Boolean logic (50 examples)")
    print("   • Statistics (35 examples)")
    print("   • RF/wavelength (23 examples)")
    print("   • Data structures (20 examples)")
    print("   • Computer algorithms (18 examples)")
    print("=" * 60)
    
    # Execute the updated training script
    exec(update_data_file())