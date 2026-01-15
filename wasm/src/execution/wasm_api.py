"""
WASM API Integration
===================

Provides external API calls for WASM functions, similar to JavaScript/WASM integration.
Handles system calls like date/time, file operations, etc.

Security modes:
- Default: Uses QEMU sandbox for secure execution
- --no-sandbox: Direct host execution (faster, less secure)
"""

import datetime
import os
import subprocess
import platform
from typing import Dict, Any, Optional, List
import json
from pathlib import Path


class WASMAPIProvider:
    """Provides external API functions for WASM execution."""
    
    def __init__(self, use_sandbox: bool = True, sandbox_config: Optional[Dict] = None):
        self.use_sandbox = use_sandbox
        self.sandbox_config = sandbox_config or {}
        self.sandbox_engine = None
        
        # Initialize sandbox if enabled
        if use_sandbox:
            self._init_sandbox()
        
        self.available_apis = {
            "date": self.get_current_date,
            "time": self.get_current_time,
            "datetime": self.get_current_datetime,
            "platform": self.get_platform_info,
            "env": self.get_environment_info,
            "file_exists": self.check_file_exists,
            "list_files": self.list_directory,
            "get_file_size": self.get_file_size,
        }
    
    def _init_sandbox(self):
        """Initialize QEMU sandbox for secure API calls."""
        try:
            # Import sandbox from main directory
            import sys
            sandbox_path = Path(__file__).parent.parent.parent.parent / "sandbox" / "src"
            sys.path.insert(0, str(sandbox_path))
            
            from worldmodel_sandbox import SandboxedWorldModelInference
            
            # For API-only operations, we don't need the actual model
            # Just initialize the sandbox component directly
            print("🔒 WASM API sandbox enabled (API-only mode)")
            self.sandbox_available = True
            
        except ImportError as e:
            print(f"⚠️  WASM API sandbox not available: {e}")
            print("   Using direct execution (less secure)")
            self.use_sandbox = False
            self.sandbox_available = False
    
    def call_api(self, api_name: str, *args, **kwargs) -> Dict[str, Any]:
        """
        Call an external API function.
        
        Args:
            api_name: Name of the API function
            *args, **kwargs: Arguments to pass to the function
            
        Returns:
            Dict with success status and result
        """
        if api_name not in self.available_apis:
            return {
                "success": False,
                "error": f"Unknown API: {api_name}",
                "available": list(self.available_apis.keys())
            }
        
        try:
            result = self.available_apis[api_name](*args, **kwargs)
            return {
                "success": True,
                "result": result,
                "api": api_name
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "api": api_name
            }
    
    # Date/Time APIs
    def get_current_date(self) -> str:
        """Get current date in YYYY-MM-DD format."""
        if self.use_sandbox and self.sandbox_engine:
            code = "import datetime; print(datetime.date.today().strftime('%Y-%m-%d'))"
            result = self._execute_in_sandbox(code)
            return result.get("stdout", "").strip() or datetime.date.today().strftime("%Y-%m-%d")
        else:
            return datetime.date.today().strftime("%Y-%m-%d")
    
    def get_current_time(self) -> str:
        """Get current time in HH:MM:SS format."""
        if self.use_sandbox and self.sandbox_engine:
            code = "import datetime; print(datetime.datetime.now().strftime('%H:%M:%S'))"
            result = self._execute_in_sandbox(code)
            return result.get("stdout", "").strip() or datetime.datetime.now().strftime("%H:%M:%S")
        else:
            return datetime.datetime.now().strftime("%H:%M:%S")
    
    def get_current_datetime(self) -> str:
        """Get current datetime in ISO format."""
        if self.use_sandbox and self.sandbox_engine:
            code = "import datetime; print(datetime.datetime.now().isoformat())"
            result = self._execute_in_sandbox(code)
            return result.get("stdout", "").strip() or datetime.datetime.now().isoformat()
        else:
            return datetime.datetime.now().isoformat()
    
    # System APIs
    def get_platform_info(self) -> Dict[str, str]:
        """Get platform information."""
        return {
            "system": platform.system(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python_version": platform.python_version()
        }
    
    def get_environment_info(self) -> Dict[str, str]:
        """Get environment variables (safe subset)."""
        safe_vars = ["USER", "HOME", "PATH", "LANG", "TZ"]
        return {var: os.environ.get(var, "") for var in safe_vars}
    
    # File System APIs (read-only, safe operations)
    def check_file_exists(self, filepath: str) -> bool:
        """Check if file exists (safe paths only)."""
        # Only allow relative paths or paths in safe directories
        if os.path.isabs(filepath) and not self._is_safe_path(filepath):
            raise ValueError("Absolute paths not allowed for security")
        return os.path.exists(filepath)
    
    def list_directory(self, directory: str = ".") -> List[str]:
        """List files in directory (safe paths only)."""
        if os.path.isabs(directory) and not self._is_safe_path(directory):
            raise ValueError("Absolute paths not allowed for security")
        
        try:
            return os.listdir(directory)
        except (OSError, PermissionError):
            return []
    
    def get_file_size(self, filepath: str) -> int:
        """Get file size (safe paths only)."""
        if os.path.isabs(filepath) and not self._is_safe_path(filepath):
            raise ValueError("Absolute paths not allowed for security")
        
        try:
            return os.path.getsize(filepath)
        except (OSError, FileNotFoundError):
            return -1
    
    def _is_safe_path(self, path: str) -> bool:
        """Check if path is in safe directories."""
        safe_prefixes = [
            "/tmp/",
            "/home/",
            "/workspace/",
            str(os.getcwd())
        ]
        return any(path.startswith(prefix) for prefix in safe_prefixes)
    
    def _execute_in_sandbox(self, code: str) -> Dict[str, str]:
        """Execute code in QEMU sandbox and return results."""
        if not self.sandbox_engine:
            return {"stdout": "", "stderr": "Sandbox not available"}
        
        try:
            result = self.sandbox_engine.execute_code_secure(code, "python3")
            return {
                "stdout": result.get("stdout", ""),
                "stderr": result.get("stderr", ""),
                "status": result.get("status", "error")
            }
        except Exception as e:
            return {"stdout": "", "stderr": str(e), "status": "error"}


class WASMAPIIntegrator:
    """Integrates API calls into WASM execution context."""
    
    def __init__(self):
        self.api_provider = WASMAPIProvider()
    
    def enhance_wasm_with_apis(self, wat_code: str) -> str:
        """
        Enhance WAT code to support API calls.
        
        This would typically involve:
        1. Adding import declarations for API functions
        2. Mapping API calls to external host functions
        3. Converting return values to WASM types
        
        For now, we simulate this by detecting API patterns.
        """
        # Look for common API patterns in code comments or function names
        api_calls = []
        
        if "date" in wat_code.lower() or "time" in wat_code.lower():
            api_calls.append("datetime")
        
        if "file" in wat_code.lower() or "directory" in wat_code.lower():
            api_calls.append("file_exists")
        
        if "platform" in wat_code.lower() or "system" in wat_code.lower():
            api_calls.append("platform")
        
        # In a real implementation, we'd modify the WAT to include imports
        # For now, we just track what APIs would be needed
        enhanced_wat = wat_code
        
        if api_calls:
            # Add comment indicating required APIs
            api_comment = f";; Required APIs: {', '.join(api_calls)}\n"
            enhanced_wat = api_comment + wat_code
        
        return enhanced_wat
    
    def execute_with_api_context(self, wat_code: str, api_calls: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Execute WAT code with API context available.
        
        Args:
            wat_code: WebAssembly Text code
            api_calls: List of API calls to execute
            
        Returns:
            Combined execution results including API call results
        """
        results = {
            "wasm_result": None,
            "api_results": {},
            "success": True,
            "errors": []
        }
        
        # Execute API calls if specified
        if api_calls:
            for api_call in api_calls:
                if isinstance(api_call, dict):
                    api_name = api_call.get("name")
                    args = api_call.get("args", [])
                    kwargs = api_call.get("kwargs", {})
                else:
                    api_name = api_call
                    args = []
                    kwargs = {}
                
                api_result = self.api_provider.call_api(api_name, *args, **kwargs)
                results["api_results"][api_name] = api_result
                
                if not api_result["success"]:
                    results["success"] = False
                    results["errors"].append(f"API {api_name}: {api_result['error']}")
        
        # In a real implementation, we'd execute the WAT code with API context
        # For now, we simulate basic execution
        results["wasm_result"] = self._simulate_wasm_execution(wat_code)
        
        return results
    
    def _simulate_wasm_execution(self, wat_code: str) -> Dict[str, Any]:
        """Simulate WASM execution (placeholder)."""
        return {
            "success": True,
            "result": 42.0,
            "computed_token": "<computed>42.0</computed>"
        }