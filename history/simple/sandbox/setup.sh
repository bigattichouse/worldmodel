#!/bin/bash
set -e

echo "🔧 Setting up WorldModel QEMU Sandbox"
echo "===================================="

# Check if we're in the right directory
if [ ! -f "../train_worldmodel_rocm.py" ]; then
    echo "❌ Error: Run this script from the worldmodel/sandbox/ directory"
    exit 1
fi

# Initialize git submodules
echo "📦 Initializing scratchpad submodule..."
git submodule init
git submodule update

# Check for system dependencies
echo "🔍 Checking system dependencies..."

# Check for QEMU
if ! command -v qemu-system-x86_64 &> /dev/null; then
    echo "❌ QEMU not found. Installing..."
    
    if command -v apt-get &> /dev/null; then
        echo "   Using apt-get (Ubuntu/Debian)"
        sudo apt-get update
        sudo apt-get install -y qemu-system-x86 genisoimage ssh-client
    elif command -v dnf &> /dev/null; then
        echo "   Using dnf (Fedora/RHEL)"
        sudo dnf install -y qemu-system-x86 genisoimage openssh-clients
    elif command -v brew &> /dev/null; then
        echo "   Using brew (macOS)"
        brew install qemu
    else
        echo "❌ Unable to install QEMU automatically. Please install manually:"
        echo "   Ubuntu/Debian: sudo apt-get install qemu-system-x86 genisoimage ssh-client"
        echo "   Fedora/RHEL: sudo dnf install qemu-system-x86 genisoimage openssh-clients"
        echo "   macOS: brew install qemu"
        exit 1
    fi
else
    echo "✅ QEMU found: $(qemu-system-x86_64 --version | head -1)"
fi

# Check for KVM support (Linux only)
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    if [ ! -r /dev/kvm ]; then
        echo "⚠️  KVM not accessible. Adding user to kvm group..."
        sudo usermod -aG kvm $USER
        echo "   Please log out and back in, or run: newgrp kvm"
    else
        echo "✅ KVM support available"
    fi
fi

# Set up Node.js scratchpad (recommended for getting started)
echo "🚀 Setting up Node.js scratchpad..."
cd scratchpad/node

if ! command -v npm &> /dev/null; then
    echo "❌ npm not found. Please install Node.js first:"
    echo "   Ubuntu/Debian: sudo apt-get install nodejs npm"
    echo "   Fedora/RHEL: sudo dnf install nodejs npm"
    echo "   macOS: brew install node"
    exit 1
fi

# Install Node.js dependencies
echo "   Installing npm dependencies..."
npm install

# Install globally for easy access
echo "   Installing scratchpad CLI globally..."
sudo npm install -g .

# Test the installation
echo "🧪 Testing scratchpad installation..."
if scratchpad-cli.js --help &> /dev/null; then
    echo "✅ Scratchpad CLI installed successfully"
else
    echo "⚠️  Scratchpad CLI installation may have issues"
fi

# Create WorldModel sandbox integration
cd ../../
echo "🔗 Creating WorldModel sandbox integration..."

# Create Python integration module
cat > src/worldmodel_sandbox.py << 'EOF'
"""
WorldModel QEMU Sandbox Integration
==================================

Provides secure code execution using QEMU VMs via the scratchpad system.
"""

import subprocess
import tempfile
import json
import time
from pathlib import Path
from typing import Dict, Optional, List

class WorldModelSandbox:
    """Secure code execution using QEMU VMs."""
    
    def __init__(self, 
                 vm_name: str = "worldmodel-sandbox",
                 memory: str = "512M",
                 timeout: int = 30,
                 persistent: bool = False):
        """
        Initialize WorldModel sandbox.
        
        Args:
            vm_name: Name for the VM instance
            memory: Memory allocation for VM
            timeout: Execution timeout in seconds
            persistent: Whether to keep VM changes between runs
        """
        self.vm_name = vm_name
        self.memory = memory
        self.timeout = timeout
        self.persistent = persistent
        self.scratchpad_path = Path(__file__).parent.parent / "sandbox" / "scratchpad"
        
    def execute(self, code: str, language: str = "python3") -> Dict:
        """
        Execute code in sandboxed QEMU VM.
        
        Args:
            code: The code to execute
            language: Programming language (default: python3)
            
        Returns:
            Dict with execution results: {
                'status': 'success|error|timeout',
                'stdout': str,
                'stderr': str,
                'execution_time': float
            }
        """
        start_time = time.time()
        
        try:
            # Create temporary file with code
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(code)
                temp_file = f.name
            
            # Build scratchpad command
            cmd = [
                "scratchpad-cli.js", "run",
                "--vm", self.vm_name,
                "--timeout", str(self.timeout)
            ]
            
            if self.persistent:
                cmd.append("-p")
            
            # Add the command to execute
            cmd.append(f"{language} {temp_file}")
            
            # Execute in VM
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout + 10,  # Extra buffer for VM overhead
                cwd=self.scratchpad_path / "node"
            )
            
            execution_time = time.time() - start_time
            
            return {
                'status': 'success' if result.returncode == 0 else 'error',
                'stdout': result.stdout.strip(),
                'stderr': result.stderr.strip(),
                'execution_time': execution_time,
                'return_code': result.returncode
            }
            
        except subprocess.TimeoutExpired:
            return {
                'status': 'timeout',
                'stdout': '',
                'stderr': f'Execution timed out after {self.timeout} seconds',
                'execution_time': self.timeout
            }
        except Exception as e:
            return {
                'status': 'error',
                'stdout': '',
                'stderr': f'Sandbox error: {str(e)}',
                'execution_time': time.time() - start_time
            }
        finally:
            # Clean up temp file
            try:
                Path(temp_file).unlink()
            except:
                pass
    
    def prepare_vm(self) -> bool:
        """
        Prepare a VM with required packages for WorldModel.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            cmd = [
                "scratchpad-prepare.js",
                "--name", self.vm_name,
                "--memory", self.memory,
                "python3", "python3-pip", "python3-dev",
                "curl", "wget", "git", "vim"
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,  # 5 minutes for VM preparation
                cwd=self.scratchpad_path / "node"
            )
            
            return result.returncode == 0
            
        except Exception as e:
            print(f"Failed to prepare VM: {e}")
            return False
    
    def is_available(self) -> bool:
        """Check if sandbox is available and working."""
        try:
            # Test basic scratchpad functionality
            result = subprocess.run(
                ["scratchpad-cli.js", "--help"],
                capture_output=True,
                text=True,
                timeout=10,
                cwd=self.scratchpad_path / "node"
            )
            return result.returncode == 0
        except:
            return False
    
    def list_vms(self) -> List[str]:
        """List available VMs."""
        try:
            # This would need to be implemented based on scratchpad's VM listing
            # For now, return empty list
            return []
        except:
            return []

EOF

echo "✅ WorldModel sandbox integration created"

# Create example usage
cat > examples/sandbox_demo.py << 'EOF'
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

from worldmodel_sandbox import WorldModelSandbox

def main():
    print("🔒 WorldModel QEMU Sandbox Demo")
    print("==============================")
    
    # Create sandbox instance
    sandbox = WorldModelSandbox(
        vm_name="demo-vm",
        memory="512M",
        timeout=30
    )
    
    # Check if sandbox is available
    if not sandbox.is_available():
        print("❌ Sandbox not available. Make sure scratchpad is installed.")
        print("   Run: ./setup.sh")
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
    
    print("\n🎉 All tests completed!")
    print("Your code executed safely in isolated QEMU VMs.")

if __name__ == "__main__":
    exit(main())
EOF

chmod +x examples/sandbox_demo.py

echo "✅ Example demo created"

# Create installation check script
cat > check_installation.py << 'EOF'
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
    
    # Check scratchpad
    print("\n🏗️  Scratchpad:")
    scratchpad_path = Path("sandbox/scratchpad/node")
    if scratchpad_path.exists():
        print("✅ Scratchpad: Downloaded")
        
        # Check if CLI is installed
        try:
            result = subprocess.run(
                ["scratchpad-cli.js", "--help"],
                capture_output=True,
                text=True,
                timeout=5,
                cwd=scratchpad_path
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
        print("❌ Scratchpad: Not found")
        all_good = False
    
    # Check WorldModel integration
    print("\n🧠 WorldModel Integration:")
    if Path("src/worldmodel_sandbox.py").exists():
        print("✅ Sandbox module: Created")
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
                all_good = False
        else:
            print("⚠️  KVM: Not available (VM will use TCG - slower)")
    
    # Summary
    print(f"\n{'='*50}")
    if all_good:
        print("🎉 Installation complete! You can now use:")
        print("   python3 examples/sandbox_demo.py")
        print("   python3 run_worldmodel_inference.py --sandbox 'Your query'")
    else:
        print("❌ Installation incomplete. Please fix the issues above.")
        print("   Run: ./sandbox/setup.sh")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
EOF

chmod +x check_installation.py

echo ""
echo "🎉 WorldModel QEMU Sandbox setup complete!"
echo ""
echo "Next steps:"
echo "1. Test the installation: python3 check_installation.py"
echo "2. Run the demo: python3 examples/sandbox_demo.py"
echo "3. Use with WorldModel: python3 run_worldmodel_inference.py --sandbox 'Calculate 25% of 400'"
echo ""
echo "The sandbox provides complete isolation for AI-generated code execution."
echo "See sandbox/README.md for detailed documentation."