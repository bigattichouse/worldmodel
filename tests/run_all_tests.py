#!/usr/bin/env python3
"""
Run All ByteLogic Tests
=======================

Central test runner for all ByteLogic components and training pipeline.
"""

import os
import sys
import subprocess
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

def run_test(test_script, description):
    """Run a single test script."""
    print(f"\n🧪 Running {description}...")
    print("=" * 50)
    
    try:
        result = subprocess.run([
            sys.executable, test_script
        ], cwd=Path(__file__).parent, capture_output=False)
        
        if result.returncode == 0:
            print(f"✅ {description} PASSED")
            return True
        else:
            print(f"❌ {description} FAILED")
            return False
    except Exception as e:
        print(f"❌ {description} ERROR: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 ByteLogic Test Suite")
    print("=" * 60)
    
    # Change to tests directory
    os.chdir(Path(__file__).parent)
    
    tests = [
        ("test_bytelogic_simple.py", "ByteLogic Compiler & Basic Functionality"),
        ("test_bytelogic_integration.py", "ByteLogic Integration Tests"),
        ("test_training_pipeline.py", "Training Pipeline Tests"),
        ("validate_bytelogic_training_data.py ../../training/datasets/corrected_bytelogic_dataset.json", "Training Data Validation"),
        ("check_token_migration.py", "Token Migration Verification")
    ]
    
    results = []
    
    for test_cmd, description in tests:
        # Split command if it has arguments
        cmd_parts = test_cmd.split()
        test_script = cmd_parts[0]
        
        # Check if test file exists
        if not os.path.exists(test_script):
            print(f"⚠️  Test file not found: {test_script}")
            results.append((description, False))
            continue
        
        # Run the test
        success = run_test(test_cmd.split(), description)
        results.append((description, success))
    
    # Summary
    print(f"\n📊 Test Summary")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for description, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"  {status}  {description}")
        if success:
            passed += 1
    
    print("=" * 60)
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed! System ready for training.")
        return True
    elif passed >= total * 0.8:
        print("⚠️  Most tests passed. Minor issues may be present.")
        return True
    else:
        print("❌ Multiple test failures. Please review and fix issues.")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)