"""
Configuration management for WorldModel LLM experiment.

Handles environment setup, path resolution, and centralized configuration
for all modules in the project.
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict


@dataclass
class ModelConfig:
    """Configuration for the base language model."""
    model_name: str = "google/gemma-3-270m-it"
    model_path: str = "../model"
    device: str = "cuda"
    torch_dtype: str = "bfloat16"
    max_length: int = 2048
    temperature: float = 0.7
    top_p: float = 0.9


@dataclass
class ExecutionConfig:
    """Configuration for code execution environment."""
    vm_type: str = "qemu"
    vm_path: str = "../scratchpad"
    timeout_seconds: int = 30
    max_memory_mb: int = 512
    allowed_languages: list = None
    
    def __post_init__(self):
        if self.allowed_languages is None:
            self.allowed_languages = ["python", "javascript", "bash", "c"]


@dataclass
class MemoryConfig:
    """Configuration for RAG and model memory systems."""
    vector_db_type: str = "faiss"
    vector_db_path: str = "./data/vectors"
    embedding_model: str = "huggingface"  # Use local model for embeddings
    embedding_dim: int = 2048  # Gemma hidden size
    max_retrievals: int = 5
    similarity_threshold: float = 0.7
    model_registry_path: str = "./data/models"


@dataclass
class TrainingConfig:
    """Configuration for training pipeline."""
    batch_size: int = 4
    learning_rate: float = 5e-5
    num_epochs: int = 3
    warmup_steps: int = 100
    gradient_accumulation_steps: int = 4
    max_grad_norm: float = 1.0
    save_steps: int = 500
    eval_steps: int = 100
    output_dir: str = "./checkpoints"
    synthetic_data_size: int = 2500


@dataclass
class UncertaintyConfig:
    """Configuration for uncertainty detection."""
    perplexity_threshold: float = 50.0
    confidence_threshold: float = 0.8
    min_tokens_for_detection: int = 10
    detection_method: str = "perplexity"


@dataclass
class ApprovalConfig:
    """Configuration for execution approval system."""
    auto_approve_categories: list = None
    require_approval_categories: list = None
    approval_timeout_seconds: int = 300
    require_user_approval: bool = True
    cache_approvals: bool = False
    approval_cache_ttl: int = 3600
    auto_approve_low_risk: bool = False
    auto_approve_patterns: list = None
    auto_deny_patterns: list = None
    
    def __post_init__(self):
        if self.auto_approve_categories is None:
            self.auto_approve_categories = [
                "python:math",
                "python:data_processing",
                "javascript:computation",
                "bash:file_read"
            ]
        if self.require_approval_categories is None:
            self.require_approval_categories = [
                "bash:system",
                "python:web",
                "c:all"
            ]
            
        if self.auto_approve_patterns is None:
            self.auto_approve_patterns = []
            
        if self.auto_deny_patterns is None:
            self.auto_deny_patterns = []


@dataclass
class WorldModelConfig:
    """Main configuration container for the entire system."""
    model: ModelConfig
    execution: ExecutionConfig
    memory: MemoryConfig
    training: TrainingConfig
    uncertainty: UncertaintyConfig
    approval: ApprovalConfig
    
    @classmethod
    def default(cls) -> 'WorldModelConfig':
        """Create default configuration."""
        return cls(
            model=ModelConfig(),
            execution=ExecutionConfig(),
            memory=MemoryConfig(),
            training=TrainingConfig(),
            uncertainty=UncertaintyConfig(),
            approval=ApprovalConfig()
        )
    
    @classmethod
    def from_file(cls, config_path: str) -> 'WorldModelConfig':
        """Load configuration from JSON file."""
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(path, 'r') as f:
            config_dict = json.load(f)
        
        return cls.from_dict(config_dict)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'WorldModelConfig':
        """Create configuration from dictionary."""
        return cls(
            model=ModelConfig(**config_dict.get('model', {})),
            execution=ExecutionConfig(**config_dict.get('execution', {})),
            memory=MemoryConfig(**config_dict.get('memory', {})),
            training=TrainingConfig(**config_dict.get('training', {})),
            uncertainty=UncertaintyConfig(**config_dict.get('uncertainty', {})),
            approval=ApprovalConfig(**config_dict.get('approval', {}))
        )
    
    def to_file(self, config_path: str) -> None:
        """Save configuration to JSON file."""
        path = Path(config_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'model': asdict(self.model),
            'execution': asdict(self.execution),
            'memory': asdict(self.memory),
            'training': asdict(self.training),
            'uncertainty': asdict(self.uncertainty),
            'approval': asdict(self.approval)
        }


class ConfigManager:
    """Manages configuration loading and environment setup."""
    
    def __init__(self, config_path: Optional[str] = None):
        self._config: Optional[WorldModelConfig] = None
        self._config_path = config_path or os.getenv('WORLDMODEL_CONFIG', './config.json')
    
    @property
    def config(self) -> WorldModelConfig:
        """Get current configuration, loading default if not set."""
        if self._config is None:
            self._config = self._load_config()
        return self._config
    
    def _load_config(self) -> WorldModelConfig:
        """Load configuration from file or create default."""
        try:
            return WorldModelConfig.from_file(self._config_path)
        except FileNotFoundError:
            # Create default config and save it
            default_config = WorldModelConfig.default()
            default_config.to_file(self._config_path)
            return default_config
    
    def reload_config(self) -> None:
        """Reload configuration from file."""
        self._config = None
    
    def update_config(self, **kwargs) -> None:
        """Update configuration sections."""
        config_dict = self.config.to_dict()
        for section, updates in kwargs.items():
            if section in config_dict:
                config_dict[section].update(updates)
        
        self._config = WorldModelConfig.from_dict(config_dict)
    
    def setup_paths(self) -> None:
        """Create necessary directories based on configuration."""
        paths_to_create = [
            self.config.memory.vector_db_path,
            self.config.memory.model_registry_path,
            self.config.training.output_dir,
            Path(self.config.model.model_path).parent
        ]
        
        for path in paths_to_create:
            Path(path).mkdir(parents=True, exist_ok=True)
    
    def validate_config(self) -> bool:
        """Validate configuration values."""
        config = self.config
        
        # Validate model path
        model_path = Path(config.model.model_path)
        if not model_path.exists():
            raise ValueError(f"Model path does not exist: {model_path}")
        
        # Validate execution environment
        if config.execution.vm_type == "qemu":
            vm_path = Path(config.execution.vm_path)
            if not vm_path.exists():
                raise ValueError(f"VM path does not exist: {vm_path}")
        
        # Validate thresholds
        if not 0 <= config.uncertainty.confidence_threshold <= 1:
            raise ValueError("Confidence threshold must be between 0 and 1")
        
        if config.training.batch_size <= 0:
            raise ValueError("Batch size must be positive")
        
        return True


# Global config manager instance
config_manager = ConfigManager()

# Convenience function to get config
def get_config() -> WorldModelConfig:
    """Get the global configuration."""
    return config_manager.config