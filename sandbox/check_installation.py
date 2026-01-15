#!/usr/bin/env python3
"""Check WorldModel sandbox installation."""

import subprocess
import sys
from pathlib import Path

def check_command(cmd, name):
    """Check if a command is available."""
    try:
        result = subprocess.run([cmd, "--version"], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print(f"✅ {name}: Available")
            return True
        else:
            print(f"❌ {name}: Command failed")
            return False
    except FileNotFoundError:
        print(f"❌ {name}: Not found")
        return False
    except Exception as e:
        print(f"⚠️  {name}: Error - {e}")
        return False

def main():
    print("🔍 WorldModel Sandbox Installation Check")
    print("=======================================")
    
    all_good = True
    
    # Check system dependencies
    print("\n📦 System Dependencies:")
    all_good &= check_command("qemu-system-x86_64", "QEMU")
    all_good &= check_command("ssh", "SSH Client")
    all_good &= check_command("node", "Node.js")
    all_good &= check_command("npm", "NPM")
    
    # Check scratchpad submodule
    print("\n🏗️  Scratchpad:")
    scratchpad_path = Path("scratchpad")
    if scratchpad_path.exists():
        print("✅ Scratchpad: Downloaded")
        
        # Check if Node.js scratchpad exists
        node_path = scratchpad_path / "node"
        if node_path.exists():
            print("✅ Node.js implementation: Found")
            
            # Check if CLI works
            try:
                result = subprocess.run(
                    ["node", "scratchpad-cli.js", "--help"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                    cwd=node_path
                )
                if result.returncode == 0:
                    print("✅ Scratchpad CLI: Working")
                else:
                    print("❌ Scratchpad CLI: Not working")
                    all_good = False
            except Exception as e:
                print(f"❌ Scratchpad CLI: Error - {e}")
                all_good = False
        else:
            print("❌ Node.js implementation: Not found")
            all_good = False
    else:
        print("❌ Scratchpad: Not found (run: git submodule update --init)")
        all_good = False
    
    # Check WorldModel integration
    print("\n🧠 WorldModel Integration:")
    if Path("src/worldmodel_sandbox.py").exists():
        print("✅ Sandbox module: Created")
        
        # Try importing the module
        try:
            sys.path.insert(0, str(Path("src")))
            from worldmodel_sandbox import WorldModelSandbox
            print("✅ Module import: Working")
        except ImportError as e:
            print(f"❌ Module import: Failed - {e}")
            all_good = False
    else:
        print("❌ Sandbox module: Missing")
        all_good = False
    
    # Check KVM (Linux only)
    if sys.platform.startswith("linux"):
        print("\n🚀 Virtualization:")
        if Path("/dev/kvm").exists():
            try:
                with open("/dev/kvm", "r"):
                    print("✅ KVM: Accessible")
            except PermissionError:
                print("⚠️  KVM: Permission denied (run: sudo usermod -aG kvm $USER)")
                print("   You may need to log out and back in")
            except Exception:
                print("⚠️  KVM: Access test failed")
        else:
            print("⚠️  KVM: Not available (VM will use TCG - slower)")
    
    # Check if we're in the right directory
    print("\n📁 Directory Structure:")
    if Path("../train_worldmodel_rocm.py").exists():
        print("✅ WorldModel directory: Correct")
    else:
        print("❌ WorldModel directory: Run this from worldmodel/sandbox/")
        all_good = False
    
    # Summary
    print(f"\n{'='*50}")
    if all_good:
        print("🎉 Installation complete! You can now use:")
        print("   python3 examples/sandbox_demo.py")
        print("   cd .. && python3 run_worldmodel_inference.py --sandbox 'What is today\\'s date?'")
        print("\nThe sandbox provides complete isolation for AI-generated code execution.")
    else:
        print("❌ Installation incomplete. Please fix the issues above.")
        print("   Run: ./setup.sh")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())