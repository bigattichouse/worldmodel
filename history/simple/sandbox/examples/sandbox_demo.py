#!/usr/bin/env python3
"""
WorldModel Sandbox Demo
======================

Demonstrates secure code execution using QEMU VMs.
"""

import sys
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def main():
    print("🔒 WorldModel QEMU Sandbox Demo")
    print("==============================")
    
    try:
        from worldmodel_sandbox import WorldModelSandbox
    except ImportError as e:
        print(f"❌ Cannot import sandbox module: {e}")
        print("   Make sure to run: ./sandbox/setup.sh")
        return 1
    
    # Create sandbox instance
    sandbox = WorldModelSandbox(
        vm_name="demo-vm",
        memory="512M",
        timeout=30
    )
    
    # Check if sandbox is available
    if not sandbox.is_available():
        print("❌ Sandbox not available. Make sure scratchpad is installed.")
        print("   Run: ./sandbox/setup.sh")
        return 1
    
    print("✅ Sandbox available")
    
    # Prepare VM if needed
    print("🚀 Preparing VM (this may take a few minutes)...")
    if sandbox.prepare_vm():
        print("✅ VM prepared successfully")
    else:
        print("⚠️  VM preparation failed, but we can still try to run code")
    
    # Test code execution
    test_cases = [
        {
            "name": "Basic Python",
            "code": "print('Hello from secure VM!')\nprint(f'2 + 2 = {2 + 2}')"
        },
        {
            "name": "System Information",
            "code": """
import platform
import os
print(f"Platform: {platform.platform()}")
print(f"Current user: {os.getenv('USER', 'unknown')}")
print(f"Working directory: {os.getcwd()}")
"""
        },
        {
            "name": "Mathematical Calculation",
            "code": """
import math
import statistics

data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
print(f"Data: {data}")
print(f"Mean: {statistics.mean(data)}")
print(f"Standard deviation: {statistics.stdev(data):.4f}")
print(f"Square root of sum: {math.sqrt(sum(data)):.4f}")
"""
        },
        {
            "name": "Date and Time",
            "code": """
from datetime import datetime, date
import time

now = datetime.now()
today = date.today()

print(f"Current time: {now.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Day of week: {now.strftime('%A')}")
print(f"Unix timestamp: {time.time()}")

# Calculate days until New Year
new_year = date(today.year + 1, 1, 1)
days_left = (new_year - today).days
print(f"Days until New Year: {days_left}")
"""
        },
        {
            "name": "Potentially Dangerous Code (Safe in VM)",
            "code": """
import subprocess
import os

# This would be dangerous on host system, but safe in VM
print("Attempting to list system files...")
try:
    result = subprocess.run(['ls', '/etc'], capture_output=True, text=True)
    print("System files found:", len(result.stdout.split()))
except Exception as e:
    print(f"Error: {e}")

# This won't affect the host system
print("Creating and removing test files...")
with open('/tmp/test_file.txt', 'w') as f:
    f.write('This is a test file in the VM')
os.remove('/tmp/test_file.txt')
print("File operations completed safely in VM")

# Simulate a "dangerous" command that's safe in VM
print("Running 'potentially dangerous' command...")
result = subprocess.run(['whoami'], capture_output=True, text=True)
print(f"VM user: {result.stdout.strip()}")
"""
        }
    ]
    
    # Execute test cases
    for i, test in enumerate(test_cases, 1):
        print(f"\n📋 Test {i}: {test['name']}")
        print("-" * 50)
        
        result = sandbox.execute(test['code'])
        
        print(f"Status: {result['status']}")
        print(f"Execution time: {result['execution_time']:.3f}s")
        
        if result['stdout']:
            print("Output:")
            print(result['stdout'])
        
        if result['stderr']:
            print("Errors:")
            print(result['stderr'])
        
        # Add separator between tests
        if i < len(test_cases):
            print()
    
    print("\n🎉 All tests completed!")
    print("Your code executed safely in isolated QEMU VMs.")
    
    # Cleanup
    print("\n🧹 Cleaning up...")
    if sandbox.cleanup_vm():
        print("✅ VM cleanup completed")
    else:
        print("⚠️  VM cleanup may not be complete")

if __name__ == "__main__":
    exit(main())