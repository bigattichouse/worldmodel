"""
VM Interface for WorldModel LLM experiment.

Provides unified interface for code execution in isolated QEMU environment,
supporting multiple programming languages with safety and resource limits.
"""

import asyncio
import subprocess
import tempfile
import json
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass
from enum import Enum
import shlex

from ..utils.config import get_config
from ..utils.logging import get_logger

logger = get_logger('vmInterface')


class ExecutionStatus(Enum):
    """Status of code execution."""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    ERROR = "error"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


@dataclass
class ExecutionResult:
    """Result of code execution."""
    status: ExecutionStatus
    stdout: str = ""
    stderr: str = ""
    return_code: Optional[int] = None
    execution_time: float = 0.0
    memory_used: int = 0  # KB
    error_message: str = ""
    files_created: List[str] = None
    files_modified: List[str] = None
    
    def __post_init__(self):
        if self.files_created is None:
            self.files_created = []
        if self.files_modified is None:
            self.files_modified = []
    
    @property
    def success(self) -> bool:
        """Check if execution was successful."""
        return self.status == ExecutionStatus.SUCCESS and self.return_code == 0
    
    @property
    def output(self) -> str:
        """Get combined output."""
        return f"{self.stdout}\n{self.stderr}".strip()


class LanguageExecutor:
    """Base class for language-specific executors."""
    
    def __init__(self, language: str, vm_path: str, timeout: int = 30):
        self.language = language
        self.vm_path = vm_path
        self.timeout = timeout
        self.logger = get_logger(f'executor.{language}')
    
    async def execute(self, code: str, requirements: List[str] = None) -> ExecutionResult:
        """Execute code in the VM. Must be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement execute method")
    
    def _validate_code(self, code: str) -> List[str]:
        """Validate code before execution. Return list of validation errors."""
        errors = []
        if not code.strip():
            errors.append("Empty code provided")
        return errors
    
    def _prepare_execution_environment(self, code: str) -> Tuple[str, str]:
        """Prepare code and return (script_content, filename)."""
        return code, f"script.{self._get_file_extension()}"
    
    def _get_file_extension(self) -> str:
        """Get file extension for this language."""
        extensions = {
            'python': 'py',
            'javascript': 'js', 
            'bash': 'sh',
            'c': 'c'
        }
        return extensions.get(self.language, 'txt')
    
    async def _execute_in_vm(self, script_path: str, args: List[str] = None) -> ExecutionResult:
        """Execute script in VM environment."""
        if args is None:
            args = []
            
        # Construct VM execution command
        cmd = self._build_vm_command(script_path, args)
        
        self.logger.debug(f"Executing command: {' '.join(cmd)}")
        
        start_time = time.time()
        
        try:
            # Execute with timeout
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                limit=1024 * 1024  # 1MB output limit
            )
            
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=self.timeout
                )
                
                execution_time = time.time() - start_time
                
                result = ExecutionResult(
                    status=ExecutionStatus.SUCCESS if process.returncode == 0 else ExecutionStatus.ERROR,
                    stdout=stdout.decode('utf-8', errors='replace'),
                    stderr=stderr.decode('utf-8', errors='replace'),
                    return_code=process.returncode,
                    execution_time=execution_time
                )
                
                self.logger.info(f"Execution completed in {execution_time:.2f}s, return code: {process.returncode}")
                
                return result
                
            except asyncio.TimeoutError:
                self.logger.warning(f"Execution timeout after {self.timeout}s")
                try:
                    process.terminate()
                    await asyncio.wait_for(process.wait(), timeout=5)
                except:
                    process.kill()
                
                return ExecutionResult(
                    status=ExecutionStatus.TIMEOUT,
                    error_message=f"Execution timed out after {self.timeout} seconds",
                    execution_time=time.time() - start_time
                )
                
        except Exception as e:
            self.logger.error(f"VM execution failed: {e}", exc_info=True)
            return ExecutionResult(
                status=ExecutionStatus.ERROR,
                error_message=f"VM execution failed: {str(e)}",
                execution_time=time.time() - start_time
            )
    
    def _build_vm_command(self, script_path: str, args: List[str]) -> List[str]:
        """Build command to execute script in VM using QEMU."""
        # Use QEMU for proper isolation
        vm_command = [
            'qemu-system-x86_64',
            '-m', '512M',  # 512MB RAM
            '-smp', '1',   # 1 CPU core
            '-enable-kvm',  # Use KVM if available
            '-nographic',   # No graphics
            '-netdev', 'user,id=net0,restrict=on',  # Restricted networking
            '-device', 'rtl8139,netdev=net0',
            '-drive', f'file={self.vm_path},format=qcow2,if=virtio',  # VM image
            '-virtfs', f'local,path={Path(script_path).parent},mount_tag=hostshare,security_model=mapped,readonly=on',  # Mount script directory
            '-append', f'init=/bin/sh -c "mount -t 9p -o trans=virtio hostshare /mnt && {self._get_vm_execution_command(script_path, args)}"'
        ]
        return vm_command
    
    def _get_vm_execution_command(self, script_path: str, args: List[str]) -> str:
        """Get the command to run inside the VM. Must be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement VM execution command")
    
    def _build_direct_command(self, script_path: str, args: List[str]) -> List[str]:
        """Build direct execution command (for testing when QEMU unavailable)."""
        raise NotImplementedError("Subclasses must implement command building")


class PythonExecutor(LanguageExecutor):
    """Executor for Python code."""
    
    def __init__(self, vm_path: str, timeout: int = 30):
        super().__init__("python", vm_path, timeout)
    
    def _validate_code(self, code: str) -> List[str]:
        """Validate Python code syntax."""
        errors = super()._validate_code(code)
        
        try:
            compile(code, '<script>', 'exec')
        except SyntaxError as e:
            errors.append(f"Python syntax error: {e}")
        
        return errors
    
    async def execute(self, code: str, requirements: List[str] = None) -> ExecutionResult:
        """Execute Python code."""
        # Validate code
        validation_errors = self._validate_code(code)
        if validation_errors:
            return ExecutionResult(
                status=ExecutionStatus.ERROR,
                error_message=f"Validation failed: {'; '.join(validation_errors)}"
            )
        
        # Prepare execution
        script_content, filename = self._prepare_execution_environment(code)
        
        # Write to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(script_content)
            temp_script = f.name
        
        try:
            result = await self._execute_in_vm(temp_script)
            return result
        finally:
            # Cleanup
            Path(temp_script).unlink(missing_ok=True)
    
    def _get_vm_execution_command(self, script_path: str, args: List[str]) -> str:
        """Get Python execution command for inside VM."""
        script_name = Path(script_path).name
        cmd_parts = ['python3', f'/mnt/{script_name}'] + args
        return ' '.join(cmd_parts)
    
    def _build_direct_command(self, script_path: str, args: List[str]) -> List[str]:
        """Build Python execution command for direct execution (testing only)."""
        return ['python3', script_path] + args


class JavaScriptExecutor(LanguageExecutor):
    """Executor for JavaScript code."""
    
    def __init__(self, vm_path: str, timeout: int = 30):
        super().__init__("javascript", vm_path, timeout)
    
    async def execute(self, code: str, requirements: List[str] = None) -> ExecutionResult:
        """Execute JavaScript code."""
        # Validate code
        validation_errors = self._validate_code(code)
        if validation_errors:
            return ExecutionResult(
                status=ExecutionStatus.ERROR,
                error_message=f"Validation failed: {'; '.join(validation_errors)}"
            )
        
        # Prepare execution
        script_content, filename = self._prepare_execution_environment(code)
        
        # Write to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.js', delete=False) as f:
            f.write(script_content)
            temp_script = f.name
        
        try:
            result = await self._execute_in_vm(temp_script)
            return result
        finally:
            # Cleanup
            Path(temp_script).unlink(missing_ok=True)
    
    def _get_vm_execution_command(self, script_path: str, args: List[str]) -> str:
        """Get Node.js execution command for inside VM."""
        script_name = Path(script_path).name
        cmd_parts = ['node', f'/mnt/{script_name}'] + args
        return ' '.join(cmd_parts)
    
    def _build_direct_command(self, script_path: str, args: List[str]) -> List[str]:
        """Build JavaScript execution command for direct execution (testing only)."""
        return ['node', script_path] + args


class BashExecutor(LanguageExecutor):
    """Executor for Bash scripts."""
    
    def __init__(self, vm_path: str, timeout: int = 30):
        super().__init__("bash", vm_path, timeout)
    
    def _validate_code(self, code: str) -> List[str]:
        """Validate bash code for dangerous commands."""
        errors = super()._validate_code(code)
        
        # Check for potentially dangerous commands
        dangerous_commands = ['rm -rf', 'format', 'fdisk', 'mkfs', 'dd if=', '> /dev/']
        for cmd in dangerous_commands:
            if cmd in code:
                errors.append(f"Potentially dangerous command detected: {cmd}")
        
        return errors
    
    async def execute(self, code: str, requirements: List[str] = None) -> ExecutionResult:
        """Execute Bash code."""
        # Validate code
        validation_errors = self._validate_code(code)
        if validation_errors:
            return ExecutionResult(
                status=ExecutionStatus.ERROR,
                error_message=f"Validation failed: {'; '.join(validation_errors)}"
            )
        
        # Prepare execution
        script_content = f"#!/bin/bash\nset -e\n{code}"
        
        # Write to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.sh', delete=False) as f:
            f.write(script_content)
            temp_script = f.name
        
        # Make executable
        Path(temp_script).chmod(0o755)
        
        try:
            result = await self._execute_in_vm(temp_script)
            return result
        finally:
            # Cleanup
            Path(temp_script).unlink(missing_ok=True)
    
    def _get_vm_execution_command(self, script_path: str, args: List[str]) -> str:
        """Get Bash execution command for inside VM."""
        script_name = Path(script_path).name
        cmd_parts = ['bash', f'/mnt/{script_name}'] + args
        return ' '.join(cmd_parts)
    
    def _build_direct_command(self, script_path: str, args: List[str]) -> List[str]:
        """Build Bash execution command for direct execution (testing only)."""
        return ['bash', script_path] + args


class CExecutor(LanguageExecutor):
    """Executor for C code."""
    
    def __init__(self, vm_path: str, timeout: int = 30):
        super().__init__("c", vm_path, timeout)
        self.compile_timeout = 10  # Compilation timeout
    
    async def execute(self, code: str, requirements: List[str] = None) -> ExecutionResult:
        """Execute C code (compile + run)."""
        # Validate code
        validation_errors = self._validate_code(code)
        if validation_errors:
            return ExecutionResult(
                status=ExecutionStatus.ERROR,
                error_message=f"Validation failed: {'; '.join(validation_errors)}"
            )
        
        # Prepare execution
        with tempfile.TemporaryDirectory() as temp_dir:
            source_file = Path(temp_dir) / "program.c"
            executable_file = Path(temp_dir) / "program"
            
            # Write source code
            with open(source_file, 'w') as f:
                f.write(code)
            
            # Compile
            compile_result = await self._compile_c_code(source_file, executable_file)
            if not compile_result.success:
                return compile_result
            
            # Execute
            result = await self._execute_in_vm(str(executable_file))
            return result
    
    async def _compile_c_code(self, source_file: Path, executable_file: Path) -> ExecutionResult:
        """Compile C code."""
        cmd = ['gcc', '-o', str(executable_file), str(source_file), '-Wall', '-Wextra']
        
        start_time = time.time()
        
        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=self.compile_timeout
            )
            
            compilation_time = time.time() - start_time
            
            if process.returncode != 0:
                return ExecutionResult(
                    status=ExecutionStatus.ERROR,
                    stderr=stderr.decode('utf-8', errors='replace'),
                    return_code=process.returncode,
                    execution_time=compilation_time,
                    error_message="Compilation failed"
                )
            
            return ExecutionResult(
                status=ExecutionStatus.SUCCESS,
                stdout=stdout.decode('utf-8', errors='replace'),
                stderr=stderr.decode('utf-8', errors='replace'),
                return_code=process.returncode,
                execution_time=compilation_time
            )
            
        except asyncio.TimeoutError:
            return ExecutionResult(
                status=ExecutionStatus.TIMEOUT,
                error_message=f"Compilation timed out after {self.compile_timeout} seconds",
                execution_time=time.time() - start_time
            )
        except Exception as e:
            return ExecutionResult(
                status=ExecutionStatus.ERROR,
                error_message=f"Compilation failed: {str(e)}",
                execution_time=time.time() - start_time
            )
    
    def _get_vm_execution_command(self, script_path: str, args: List[str]) -> str:
        """Get C executable execution command for inside VM."""
        script_name = Path(script_path).name
        cmd_parts = [f'/mnt/{script_name}'] + args
        return ' '.join(cmd_parts)
    
    def _build_direct_command(self, script_path: str, args: List[str]) -> List[str]:
        """Build C executable command."""
        return [script_path] + args


class VMInterface:
    """Unified interface for code execution across multiple languages."""
    
    def __init__(self, vm_path: Optional[str] = None):
        self.config = get_config()
        self.vm_path = vm_path or self.config.execution.vm_path
        self.timeout = self.config.execution.timeout_seconds
        self.allowed_languages = set(self.config.execution.allowed_languages)
        self.logger = get_logger('vmInterface')
        
        # Initialize executors
        self.executors: Dict[str, LanguageExecutor] = {
            'python': PythonExecutor(self.vm_path, self.timeout),
            'javascript': JavaScriptExecutor(self.vm_path, self.timeout),
            'js': JavaScriptExecutor(self.vm_path, self.timeout),  # Alias
            'bash': BashExecutor(self.vm_path, self.timeout),
            'c': CExecutor(self.vm_path, self.timeout)
        }
        
        self.logger.info(f"VMInterface initialized with {len(self.executors)} executors")
        self.logger.debug(f"Allowed languages: {self.allowed_languages}")
    
    async def execute_code(self, language: str, code: str, 
                          requirements: List[str] = None) -> ExecutionResult:
        """
        Execute code in the specified language.
        
        Args:
            language: Programming language (python, javascript, bash, c)
            code: Code to execute
            requirements: Optional list of requirements for execution
            
        Returns:
            ExecutionResult with execution details
        """
        # Normalize language name
        language = language.lower().strip()
        
        self.logger.debug(f"Executing {language} code of length {len(code)}")
        
        # Check if language is allowed
        if language not in self.allowed_languages:
            return ExecutionResult(
                status=ExecutionStatus.ERROR,
                error_message=f"Language '{language}' not allowed. Allowed: {self.allowed_languages}"
            )
        
        # Get executor for language
        executor = self.executors.get(language)
        if not executor:
            return ExecutionResult(
                status=ExecutionStatus.ERROR,
                error_message=f"No executor available for language: {language}"
            )
        
        try:
            # Execute code
            result = await executor.execute(code, requirements)
            
            self.logger.info(f"{language} execution completed: {result.status.value}")
            if not result.success:
                self.logger.warning(f"Execution failed: {result.error_message}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Execution failed with exception: {e}", exc_info=True)
            return ExecutionResult(
                status=ExecutionStatus.ERROR,
                error_message=f"Execution failed: {str(e)}"
            )
    
    async def execute_model_tag(self, model_tag) -> ExecutionResult:
        """Execute code from a parsed model tag."""
        # Import here to avoid circular imports
        from ..core.tagParser import ModelTag
        
        if not isinstance(model_tag, ModelTag):
            return ExecutionResult(
                status=ExecutionStatus.ERROR,
                error_message="Invalid model tag provided"
            )
        
        return await self.execute_code(
            language=model_tag.language,
            code=model_tag.code
        )
    
    async def batch_execute(self, executions: List[Dict[str, Any]]) -> List[ExecutionResult]:
        """
        Execute multiple code blocks in batch.
        
        Args:
            executions: List of dicts with 'language', 'code', and optional 'requirements'
            
        Returns:
            List of ExecutionResult in same order as input
        """
        tasks = []
        
        for exec_spec in executions:
            task = self.execute_code(
                language=exec_spec['language'],
                code=exec_spec['code'],
                requirements=exec_spec.get('requirements')
            )
            tasks.append(task)
        
        # Execute all in parallel
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Convert exceptions to error results
        final_results = []
        for result in results:
            if isinstance(result, Exception):
                final_results.append(ExecutionResult(
                    status=ExecutionStatus.ERROR,
                    error_message=f"Batch execution failed: {str(result)}"
                ))
            else:
                final_results.append(result)
        
        return final_results
    
    def is_language_supported(self, language: str) -> bool:
        """Check if a language is supported and allowed."""
        language = language.lower().strip()
        return language in self.allowed_languages and language in self.executors
    
    def get_supported_languages(self) -> List[str]:
        """Get list of supported languages."""
        return list(self.allowed_languages)
    
    def get_executor_info(self) -> Dict[str, Dict[str, Any]]:
        """Get information about available executors."""
        info = {}
        for lang, executor in self.executors.items():
            if lang in self.allowed_languages:
                info[lang] = {
                    'class': executor.__class__.__name__,
                    'timeout': executor.timeout,
                    'file_extension': executor._get_file_extension()
                }
        return info


# Convenience functions
async def execute_code(language: str, code: str, requirements: List[str] = None) -> ExecutionResult:
    """Convenience function for executing code."""
    vm = VMInterface()
    return await vm.execute_code(language, code, requirements)

async def execute_python(code: str) -> ExecutionResult:
    """Convenience function for executing Python code."""
    return await execute_code('python', code)

async def execute_javascript(code: str) -> ExecutionResult:
    """Convenience function for executing JavaScript code."""
    return await execute_code('javascript', code)

async def execute_bash(code: str) -> ExecutionResult:
    """Convenience function for executing Bash code."""
    return await execute_code('bash', code)

def is_vm_available() -> bool:
    """Check if VM environment is available."""
    try:
        vm = VMInterface()
        return Path(vm.vm_path).exists() if vm.vm_path else True
    except:
        return False