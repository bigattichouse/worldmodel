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
        self.scratchpad_path = Path(__file__).parent.parent / "scratchpad"
        
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
                'execution_time': float,
                'return_code': int
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
                "node", "scratchpad-cli.js", "run",
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
                'execution_time': self.timeout,
                'return_code': 124  # Standard timeout exit code
            }
        except Exception as e:
            return {
                'status': 'error',
                'stdout': '',
                'stderr': f'Sandbox error: {str(e)}',
                'execution_time': time.time() - start_time,
                'return_code': 1
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
                "node", "scratchpad-prepare.js",
                "--name", self.vm_name,
                "--memory", self.memory,
                "python3", "python3-pip", "python3-dev",
                "curl", "wget", "git", "vim", "build-essential"
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
                ["node", "scratchpad-cli.js", "--help"],
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
            result = subprocess.run(
                ["node", "scratchpad-live.js", "list"],
                capture_output=True,
                text=True,
                timeout=10,
                cwd=self.scratchpad_path / "node"
            )
            if result.returncode == 0:
                # Parse VM list from output
                return [line.strip() for line in result.stdout.split('\n') if line.strip()]
            return []
        except:
            return []
    
    def cleanup_vm(self) -> bool:
        """Clean up VM resources."""
        try:
            result = subprocess.run(
                ["node", "scratchpad-live.js", "stop", self.vm_name],
                capture_output=True,
                text=True,
                timeout=30,
                cwd=self.scratchpad_path / "node"
            )
            return result.returncode == 0
        except:
            return False

class SandboxedWorldModelInference:
    """WorldModel inference with QEMU sandbox integration."""
    
    def __init__(self, model_path: str, sandbox_config: Optional[Dict] = None):
        """
        Initialize sandboxed inference.
        
        Args:
            model_path: Path to trained WorldModel
            sandbox_config: Optional sandbox configuration
        """
        # Import here to avoid circular imports
        import sys
        sys.path.append(str(Path(__file__).parent.parent.parent))
        
        from run_worldmodel_inference import WorldModelInference
        
        self.inference = WorldModelInference(model_path)
        
        # Configure sandbox
        sandbox_config = sandbox_config or {}
        self.sandbox = WorldModelSandbox(
            vm_name=sandbox_config.get('vm_name', 'worldmodel-inference'),
            memory=sandbox_config.get('memory', '512M'),
            timeout=sandbox_config.get('timeout', 30),
            persistent=sandbox_config.get('persistent', False)
        )
        
        # Check if sandbox is available
        self.sandbox_available = self.sandbox.is_available()
        if not self.sandbox_available:
            print("⚠️  QEMU sandbox not available - falling back to direct execution")
    
    def execute_code_secure(self, code: str, language: str = "python") -> Dict:
        """Execute code with sandbox if available, fallback to direct execution."""
        if self.sandbox_available and code.strip():
            # Use secure VM execution
            print("🔒 Executing in secure QEMU VM...")
            return self.sandbox.execute(code, "python3")
        else:
            # Fallback to original execution method
            if hasattr(self.inference, 'execute_code'):
                return self.inference.execute_code(code, language)
            else:
                return {
                    'status': 'error',
                    'stdout': '',
                    'stderr': 'Code execution not available',
                    'execution_time': 0,
                    'return_code': 1
                }
    
    def process_query(self, user_query: str) -> Dict:
        """Process query with sandboxed code execution."""
        print(f"\n🔍 Processing: {user_query}")
        print("=" * 60)
        
        # Generate response using original inference
        raw_response = self.inference.generate_response(user_query)
        print(f"📝 Raw response: {raw_response[:100]}...")
        
        # Parse structured response
        parsed = self.inference.parse_worldmodel_response(raw_response)
        
        # Show parsed components
        print(f"\n🧠 Thinking: {parsed['thinking'][:80]}..." if parsed['thinking'] else "🧠 Thinking: (none)")
        print(f"💻 Code: {len(parsed['code'])} characters" if parsed['code'] else "💻 Code: (none)")
        print(f"📋 Requires: {parsed['requires']}" if parsed['requires'] else "📋 Requires: (none)")
        
        # Execute code with sandbox
        execution_result = None
        if parsed['code']:
            execution_result = self.execute_code_secure(parsed['code'])
            
            print(f"\n⚡ Execution Status: {execution_result['status']}")
            if execution_result['stdout']:
                print(f"📤 Output: {execution_result['stdout']}")
            if execution_result['stderr']:
                print(f"❌ Error: {execution_result['stderr']}")
            print(f"⏱️  Time: {execution_result['execution_time']:.3f}s")
        
        # Final explanation
        if parsed['explanation']:
            print(f"\n💬 Explanation: {parsed['explanation']}")
        
        return {
            'query': user_query,
            'raw_response': raw_response,
            'parsed': parsed,
            'execution': execution_result,
            'sandbox_used': self.sandbox_available and parsed['code'] is not None
        }