"""
Testing utilities for WorldModel LLM experiment.

Provides test utilities, mock objects, and integration test helpers
to support comprehensive unit and integration testing.
"""

import pytest
import asyncio
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Any, Optional, Generator, Callable, Union
from dataclasses import dataclass
from unittest.mock import Mock, AsyncMock, MagicMock, patch
import json
import numpy as np
import torch
import time
from contextlib import contextmanager

from ..core.tagParser import TagType, ParsedTag, ModelTag, RequiresTag, ParseResult
from ..execution.vmInterface import ExecutionResult, ExecutionStatus
from ..execution.approvalSystem import ApprovalDecision, ApprovalResult
from ..memory.ragSystem import Document, SearchResult as RAGSearchResult
from ..training.dataGenerator import TrainingExample, ProblemTemplate


@dataclass
class MockConfig:
    """Mock configuration for testing."""
    model_path: str = "/mock/model/path"
    device: str = "cpu"
    torch_dtype: str = "float32"
    max_length: int = 2048
    vm_path: str = "/mock/vm.qcow2"
    timeout_seconds: int = 30
    allowed_languages: List[str] = None
    
    def __post_init__(self):
        if self.allowed_languages is None:
            self.allowed_languages = ['python', 'javascript', 'bash', 'c']


@dataclass 
class MockModelOutput:
    """Mock model output for testing generation."""
    logits: torch.Tensor
    hidden_states: Optional[torch.Tensor] = None
    attentions: Optional[torch.Tensor] = None


class MockTokenizer:
    """Mock tokenizer for testing."""
    
    def __init__(self, vocab_size: int = 32000):
        self.vocab_size = vocab_size
        self.pad_token = "<pad>"
        self.eos_token = "</s>"
        self.pad_token_id = 0
        self.eos_token_id = 1
        
    def __call__(self, text: Union[str, List[str]], **kwargs):
        """Mock tokenization."""
        if isinstance(text, str):
            # Simple mock: split by spaces and convert to IDs
            tokens = text.split()
            input_ids = [hash(token) % self.vocab_size for token in tokens]
            
            return {
                'input_ids': torch.tensor([input_ids]),
                'attention_mask': torch.ones((1, len(input_ids)))
            }
        else:
            # Batch processing
            batch_input_ids = []
            batch_attention_masks = []
            max_len = 0
            
            for t in text:
                tokens = t.split()
                input_ids = [hash(token) % self.vocab_size for token in tokens]
                batch_input_ids.append(input_ids)
                max_len = max(max_len, len(input_ids))
            
            # Pad sequences
            for i in range(len(batch_input_ids)):
                current_len = len(batch_input_ids[i])
                pad_length = max_len - current_len
                batch_input_ids[i].extend([self.pad_token_id] * pad_length)
                attention_mask = [1] * current_len + [0] * pad_length
                batch_attention_masks.append(attention_mask)
            
            return {
                'input_ids': torch.tensor(batch_input_ids),
                'attention_mask': torch.tensor(batch_attention_masks)
            }
    
    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """Mock encoding."""
        tokens = text.split()
        return [hash(token) % self.vocab_size for token in tokens]
    
    def decode(self, token_ids: Union[List[int], torch.Tensor], skip_special_tokens: bool = False) -> str:
        """Mock decoding."""
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        
        # Simple mock: convert IDs back to pseudo-words
        words = [f"token_{tid}" for tid in token_ids if tid not in [self.pad_token_id, self.eos_token_id]]
        return " ".join(words)
    
    def __len__(self):
        """Return length for compatibility."""
        return self.vocab_size


class MockModel:
    """Mock language model for testing."""
    
    def __init__(self, vocab_size: int = 32000, hidden_size: int = 768):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.device = torch.device('cpu')
        self.training = False
        
    def __call__(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, **kwargs):
        """Mock forward pass."""
        batch_size, seq_len = input_ids.shape
        
        # Generate random logits
        logits = torch.randn(batch_size, seq_len, self.vocab_size)
        
        return MockModelOutput(logits=logits)
    
    def generate(self, input_ids: torch.Tensor, **kwargs):
        """Mock generation."""
        batch_size = input_ids.shape[0]
        max_new_tokens = kwargs.get('max_new_tokens', 50)
        
        # Generate random new tokens
        new_tokens = torch.randint(0, self.vocab_size, (batch_size, max_new_tokens))
        
        # Concatenate with input
        return torch.cat([input_ids, new_tokens], dim=1)
    
    def eval(self):
        """Set to eval mode."""
        self.training = False
    
    def train(self):
        """Set to train mode."""
        self.training = True
    
    def to(self, device):
        """Move to device."""
        self.device = device
        return self
    
    def parameters(self):
        """Mock parameters."""
        return [torch.randn(100, 100) for _ in range(5)]


class TestFixtures:
    """Collection of test fixtures for WorldModel components."""
    
    @staticmethod
    def sample_think_tag() -> ParsedTag:
        """Create a sample think tag."""
        return ParsedTag(
            tag_type=TagType.THINK,
            content="Let me think about this step by step. First, I need to understand the problem.",
            start_pos=0,
            end_pos=100
        )
    
    @staticmethod  
    def sample_model_tag() -> ModelTag:
        """Create a sample model tag."""
        tag = ModelTag(
            tag_type=TagType.MODEL,
            content="python: print('Hello, World!')",
            start_pos=100,
            end_pos=150
        )
        tag._parse_model_content()
        return tag
    
    @staticmethod
    def sample_requires_tag() -> RequiresTag:
        """Create a sample requires tag."""
        tag = RequiresTag(
            tag_type=TagType.REQUIRES,
            content="python:system, python:file_write",
            start_pos=150,
            end_pos=200
        )
        tag._parse_requirements()
        return tag
    
    @staticmethod
    def sample_parse_result() -> ParseResult:
        """Create a sample parse result."""
        return ParseResult(
            original_text="<think>Let me think</think><model>python: print('Hello')</model><requires>python:system</requires>",
            think_tags=[TestFixtures.sample_think_tag()],
            model_tags=[TestFixtures.sample_model_tag()],
            requires_tags=[TestFixtures.sample_requires_tag()],
            parsing_errors=[]
        )
    
    @staticmethod
    def sample_execution_result(success: bool = True) -> ExecutionResult:
        """Create a sample execution result."""
        if success:
            return ExecutionResult(
                status=ExecutionStatus.SUCCESS,
                stdout="Hello, World!\n",
                stderr="",
                return_code=0,
                execution_time=0.5,
                memory_used=1024
            )
        else:
            return ExecutionResult(
                status=ExecutionStatus.ERROR,
                stdout="",
                stderr="Traceback (most recent call last):\n  File \"<stdin>\", line 1, in <module>\nNameError: name 'undefined' is not defined",
                return_code=1,
                execution_time=0.2,
                error_message="Code execution failed"
            )
    
    @staticmethod
    def sample_approval_result(approved: bool = True) -> ApprovalResult:
        """Create a sample approval result."""
        decision = ApprovalDecision.APPROVE if approved else ApprovalDecision.DENY
        return ApprovalResult(
            approved=approved,
            decision=decision,
            reason="Automated test approval" if approved else "Test denial",
            risk_level="low" if approved else "high",
            metadata={'test': True}
        )
    
    @staticmethod
    def sample_training_example() -> TrainingExample:
        """Create a sample training example."""
        return TrainingExample(
            problem="Calculate the sum of 2 + 3",
            solution="<think>I need to add 2 and 3 together.</think>\n<model>python: result = 2 + 3\nprint(result)</model>\n<requires>python:math</requires>\n\nThe sum of 2 + 3 is 5.",
            category="math",
            difficulty="easy",
            metadata={'template': 'basic_arithmetic'}
        )
    
    @staticmethod
    def sample_document() -> Document:
        """Create a sample RAG document."""
        return Document(
            content="Python is a high-level programming language with dynamic semantics.",
            metadata={'source': 'test', 'topic': 'programming'},
            embedding=np.random.rand(768).astype(np.float32)
        )
    
    @staticmethod
    def sample_rag_search_result() -> RAGSearchResult:
        """Create a sample RAG search result."""
        return RAGSearchResult(
            document=TestFixtures.sample_document(),
            similarity=0.85,
            rank=1
        )


class MockServices:
    """Collection of mock services for integration testing."""
    
    @staticmethod
    @contextmanager
    def mock_vm_interface(success_responses: bool = True):
        """Context manager for mocking VM interface."""
        with patch('src.execution.vmInterface.VMInterface') as mock_vm:
            mock_instance = Mock()
            
            if success_responses:
                mock_instance.execute_code.return_value = asyncio.coroutine(
                    lambda: TestFixtures.sample_execution_result(success=True)
                )()
            else:
                mock_instance.execute_code.return_value = asyncio.coroutine(
                    lambda: TestFixtures.sample_execution_result(success=False)
                )()
            
            mock_vm.return_value = mock_instance
            yield mock_instance
    
    @staticmethod
    @contextmanager
    def mock_approval_system(auto_approve: bool = True):
        """Context manager for mocking approval system."""
        with patch('src.execution.approvalSystem.ApprovalSystem') as mock_approval:
            mock_instance = Mock()
            mock_instance.request_approval.return_value = TestFixtures.sample_approval_result(auto_approve)
            mock_approval.return_value = mock_instance
            yield mock_instance
    
    @staticmethod
    @contextmanager  
    def mock_rag_system(return_results: bool = True):
        """Context manager for mocking RAG system."""
        with patch('src.memory.ragSystem.RAGSystem') as mock_rag:
            mock_instance = Mock()
            
            if return_results:
                mock_instance.search.return_value = [TestFixtures.sample_rag_search_result()]
            else:
                mock_instance.search.return_value = []
            
            mock_rag.return_value = mock_instance
            yield mock_instance
    
    @staticmethod
    @contextmanager
    def mock_model_components():
        """Context manager for mocking model and tokenizer."""
        mock_tokenizer = MockTokenizer()
        mock_model = MockModel()
        
        with patch('transformers.AutoTokenizer.from_pretrained') as mock_tok, \
             patch('transformers.AutoModelForCausalLM.from_pretrained') as mock_mod:
            
            mock_tok.return_value = mock_tokenizer
            mock_mod.return_value = mock_model
            
            yield mock_model, mock_tokenizer


class AsyncTestHelper:
    """Helper class for async testing."""
    
    @staticmethod
    def run_async(coro):
        """Run an async coroutine in tests."""
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # Create new event loop if one is already running
            import nest_asyncio
            nest_asyncio.apply()
            return loop.run_until_complete(coro)
        else:
            return asyncio.run(coro)
    
    @staticmethod
    async def create_async_mock(*args, **kwargs):
        """Create an async mock that can be awaited."""
        mock = AsyncMock(*args, **kwargs)
        return mock


class TempFileManager:
    """Helper for managing temporary files in tests."""
    
    def __init__(self):
        self.temp_dirs = []
        self.temp_files = []
    
    def create_temp_dir(self, prefix: str = "worldmodel_test_") -> Path:
        """Create a temporary directory."""
        temp_dir = Path(tempfile.mkdtemp(prefix=prefix))
        self.temp_dirs.append(temp_dir)
        return temp_dir
    
    def create_temp_file(self, suffix: str = ".txt", content: str = "") -> Path:
        """Create a temporary file."""
        fd, path = tempfile.mkstemp(suffix=suffix)
        temp_path = Path(path)
        
        with open(fd, 'w') as f:
            f.write(content)
        
        self.temp_files.append(temp_path)
        return temp_path
    
    def cleanup(self):
        """Clean up all temporary files and directories."""
        for temp_file in self.temp_files:
            try:
                temp_file.unlink(missing_ok=True)
            except Exception:
                pass
        
        for temp_dir in self.temp_dirs:
            try:
                shutil.rmtree(temp_dir, ignore_errors=True)
            except Exception:
                pass
        
        self.temp_dirs.clear()
        self.temp_files.clear()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()


class TestDataGenerator:
    """Generate test data for various WorldModel components."""
    
    @staticmethod
    def generate_conversation_history(num_messages: int = 5) -> List[Dict[str, Any]]:
        """Generate a conversation history for testing."""
        conversation = []
        for i in range(num_messages):
            if i % 2 == 0:
                conversation.append({
                    'role': 'user',
                    'content': f"Test user message {i//2 + 1}: What is {i+1} + {i+2}?",
                    'timestamp': time.time() - (num_messages - i) * 60
                })
            else:
                conversation.append({
                    'role': 'assistant', 
                    'content': f"<think>I need to calculate {i} + {i+1}</think>\n<model>python: result = {i} + {i+1}\nprint(result)</model>\n<requires>python:math</requires>\n\nThe answer is {i + i + 1}.",
                    'timestamp': time.time() - (num_messages - i) * 60
                })
        return conversation
    
    @staticmethod
    def generate_training_examples(count: int = 10) -> List[TrainingExample]:
        """Generate training examples for testing."""
        examples = []
        categories = ['math', 'programming', 'logic', 'text_analysis']
        difficulties = ['easy', 'medium', 'hard']
        
        for i in range(count):
            category = categories[i % len(categories)]
            difficulty = difficulties[i % len(difficulties)]
            
            example = TrainingExample(
                problem=f"Test problem {i+1} in {category}",
                solution=f"<think>Solving test problem {i+1}</think>\n<model>python: print('Solution {i+1}')</model>\n<requires>python:system</requires>\n\nSolution for problem {i+1}.",
                category=category,
                difficulty=difficulty,
                metadata={'test_id': i+1, 'generated': True}
            )
            examples.append(example)
        
        return examples
    
    @staticmethod
    def generate_embeddings(count: int = 20, dimension: int = 768) -> List[np.ndarray]:
        """Generate random embeddings for testing."""
        return [np.random.rand(dimension).astype(np.float32) for _ in range(count)]


class PerformanceTimer:
    """Helper for timing operations in tests."""
    
    def __init__(self):
        self.start_time = None
        self.end_time = None
    
    def start(self):
        """Start timing."""
        self.start_time = time.time()
    
    def stop(self) -> float:
        """Stop timing and return elapsed time."""
        self.end_time = time.time()
        if self.start_time is None:
            return 0.0
        return self.end_time - self.start_time
    
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
    
    @property
    def elapsed(self) -> float:
        """Get elapsed time."""
        if self.start_time is None:
            return 0.0
        end = self.end_time if self.end_time else time.time()
        return end - self.start_time


# Pytest fixtures for common test setup
@pytest.fixture
def temp_file_manager():
    """Fixture for temporary file management."""
    manager = TempFileManager()
    yield manager
    manager.cleanup()


@pytest.fixture
def mock_config():
    """Fixture for mock configuration."""
    return MockConfig()


@pytest.fixture
def sample_training_examples():
    """Fixture for sample training examples."""
    return TestDataGenerator.generate_training_examples(5)


@pytest.fixture
def mock_execution_result():
    """Fixture for mock execution result."""
    return TestFixtures.sample_execution_result()


@pytest.fixture
def performance_timer():
    """Fixture for performance timing."""
    return PerformanceTimer()


# Test utilities for integration testing
class IntegrationTestBase:
    """Base class for integration tests."""
    
    def setup_method(self):
        """Setup for each test method."""
        self.temp_manager = TempFileManager()
        self.performance_timer = PerformanceTimer()
    
    def teardown_method(self):
        """Cleanup after each test method."""
        self.temp_manager.cleanup()
    
    def assert_execution_success(self, result: ExecutionResult):
        """Assert that execution was successful."""
        assert result.status == ExecutionStatus.SUCCESS
        assert result.return_code == 0
        assert result.error_message == ""
    
    def assert_execution_failure(self, result: ExecutionResult):
        """Assert that execution failed."""
        assert result.status in [ExecutionStatus.ERROR, ExecutionStatus.TIMEOUT]
        assert result.return_code != 0 or result.error_message != ""
    
    def assert_parse_result_valid(self, parse_result: ParseResult):
        """Assert that parse result is valid."""
        assert parse_result is not None
        assert len(parse_result.parsing_errors) == 0
        assert parse_result.has_tags
    
    def create_test_environment(self) -> Path:
        """Create a test environment directory."""
        test_dir = self.temp_manager.create_temp_dir("integration_test_")
        
        # Create subdirectories
        (test_dir / "models").mkdir()
        (test_dir / "data").mkdir()
        (test_dir / "logs").mkdir()
        (test_dir / "embeddings").mkdir()
        
        return test_dir


# Export commonly used testing utilities
__all__ = [
    'MockConfig',
    'MockTokenizer', 
    'MockModel',
    'TestFixtures',
    'MockServices',
    'AsyncTestHelper',
    'TempFileManager',
    'TestDataGenerator',
    'PerformanceTimer',
    'IntegrationTestBase'
]