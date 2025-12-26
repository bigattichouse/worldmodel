"""
Structured logging system for WorldModel LLM experiment.

Provides unified logging across all components with experiment tracking,
debug output, and configurable log levels.
"""

import logging
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Union
from contextlib import contextmanager

from .config import get_config


class StructuredFormatter(logging.Formatter):
    """Custom formatter that outputs structured JSON logs."""
    
    def __init__(self, include_extra: bool = True):
        super().__init__()
        self.include_extra = include_extra
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as structured JSON."""
        log_entry = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # Add exception info if present
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)
        
        # Add extra fields if enabled
        if self.include_extra:
            extra_fields = {
                key: value for key, value in record.__dict__.items()
                if key not in {
                    'name', 'msg', 'args', 'levelname', 'levelno', 'pathname',
                    'filename', 'module', 'exc_info', 'exc_text', 'stack_info',
                    'lineno', 'funcName', 'created', 'msecs', 'relativeCreated',
                    'thread', 'threadName', 'processName', 'process', 'getMessage'
                }
            }
            if extra_fields:
                log_entry['extra'] = extra_fields
        
        return json.dumps(log_entry, default=str)


class ExperimentLogger:
    """Logger for tracking experiment progress and metrics."""
    
    def __init__(self, experiment_name: str, log_dir: Optional[str] = None):
        self.experiment_name = experiment_name
        self.log_dir = Path(log_dir or "./logs/experiments")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create experiment-specific log file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.log_dir / f"{experiment_name}_{timestamp}.jsonl"
        
        self._setup_logger()
    
    def _setup_logger(self):
        """Set up the experiment logger."""
        self.logger = logging.getLogger(f"experiment.{self.experiment_name}")
        self.logger.setLevel(logging.INFO)
        
        # Remove existing handlers to avoid duplicates
        self.logger.handlers.clear()
        
        # File handler for experiment logs
        file_handler = logging.FileHandler(self.log_file)
        file_handler.setFormatter(StructuredFormatter())
        self.logger.addHandler(file_handler)
        
        # Prevent propagation to avoid duplicate logs
        self.logger.propagate = False
    
    def log_metric(self, metric_name: str, value: Union[int, float], step: Optional[int] = None):
        """Log a metric value."""
        self.logger.info(
            f"Metric: {metric_name}",
            extra={
                'type': 'metric',
                'metric_name': metric_name,
                'value': value,
                'step': step
            }
        )
    
    def log_parameter(self, param_name: str, value: Any):
        """Log a parameter value."""
        self.logger.info(
            f"Parameter: {param_name}",
            extra={
                'type': 'parameter',
                'param_name': param_name,
                'value': value
            }
        )
    
    def log_model_generation(self, input_text: str, output_text: str, 
                           tags_found: Optional[Dict[str, Any]] = None,
                           execution_result: Optional[Dict[str, Any]] = None):
        """Log model generation with tags and execution results."""
        self.logger.info(
            "Model generation completed",
            extra={
                'type': 'generation',
                'input_text': input_text,
                'output_text': output_text,
                'tags_found': tags_found or {},
                'execution_result': execution_result or {}
            }
        )
    
    def log_training_step(self, epoch: int, step: int, loss: float, 
                         learning_rate: float, batch_size: int):
        """Log training step information."""
        self.logger.info(
            f"Training step {step}",
            extra={
                'type': 'training_step',
                'epoch': epoch,
                'step': step,
                'loss': loss,
                'learning_rate': learning_rate,
                'batch_size': batch_size
            }
        )
    
    def log_evaluation(self, eval_metrics: Dict[str, Any], step: Optional[int] = None):
        """Log evaluation results."""
        self.logger.info(
            "Evaluation completed",
            extra={
                'type': 'evaluation',
                'metrics': eval_metrics,
                'step': step
            }
        )
    
    def log_execution_attempt(self, language: str, code: str, requirements: list,
                            success: bool, output: str, error: Optional[str] = None):
        """Log code execution attempt."""
        self.logger.info(
            f"Code execution ({'success' if success else 'failure'})",
            extra={
                'type': 'execution',
                'language': language,
                'code': code,
                'requirements': requirements,
                'success': success,
                'output': output,
                'error': error
            }
        )
    
    def log_uncertainty_detection(self, input_text: str, perplexity: float,
                                confidence: float, triggered: bool):
        """Log uncertainty detection results."""
        self.logger.info(
            f"Uncertainty detection ({'triggered' if triggered else 'normal'})",
            extra={
                'type': 'uncertainty',
                'input_text': input_text,
                'perplexity': perplexity,
                'confidence': confidence,
                'triggered': triggered
            }
        )
    
    def log_rag_retrieval(self, query: str, num_results: int, 
                         retrieved_models: list, similarity_scores: list):
        """Log RAG retrieval results."""
        self.logger.info(
            f"RAG retrieval: {num_results} results",
            extra={
                'type': 'rag_retrieval',
                'query': query,
                'num_results': num_results,
                'retrieved_models': retrieved_models,
                'similarity_scores': similarity_scores
            }
        )


class WorldModelLogger:
    """Main logging system for the WorldModel experiment."""
    
    _instance = None
    _experiment_logger = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self.config = get_config()
        self._setup_root_logger()
        self._initialized = True
    
    def _setup_root_logger(self):
        """Set up the root logger for the application."""
        # Get or create root logger
        self.root_logger = logging.getLogger('worldmodel')
        self.root_logger.setLevel(logging.DEBUG)
        
        # Remove existing handlers to avoid duplicates
        self.root_logger.handlers.clear()
        
        # Console handler with simple format for development
        console_handler = logging.StreamHandler(sys.stdout)
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        console_handler.setLevel(logging.INFO)
        self.root_logger.addHandler(console_handler)
        
        # File handler with structured format for analysis
        log_dir = Path("./logs")
        log_dir.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_dir / "worldmodel.jsonl")
        file_handler.setFormatter(StructuredFormatter())
        file_handler.setLevel(logging.DEBUG)
        self.root_logger.addHandler(file_handler)
    
    def get_logger(self, name: str) -> logging.Logger:
        """Get a logger for a specific module."""
        return logging.getLogger(f'worldmodel.{name}')
    
    def start_experiment(self, experiment_name: str) -> ExperimentLogger:
        """Start a new experiment logging session."""
        self._experiment_logger = ExperimentLogger(experiment_name)
        
        # Log experiment start
        self.root_logger.info(
            f"Starting experiment: {experiment_name}",
            extra={'type': 'experiment_start', 'experiment_name': experiment_name}
        )
        
        return self._experiment_logger
    
    def get_experiment_logger(self) -> Optional[ExperimentLogger]:
        """Get the current experiment logger."""
        return self._experiment_logger
    
    def log_config(self, config_dict: Dict[str, Any]):
        """Log the current configuration."""
        self.root_logger.info(
            "Configuration loaded",
            extra={'type': 'config', 'config': config_dict}
        )
    
    @contextmanager
    def log_execution_time(self, operation_name: str, logger_name: Optional[str] = None):
        """Context manager to log execution time of operations."""
        logger = self.get_logger(logger_name or 'timer')
        start_time = datetime.now()
        
        logger.debug(f"Starting operation: {operation_name}")
        
        try:
            yield
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            logger.info(
                f"Operation completed: {operation_name}",
                extra={
                    'type': 'timing',
                    'operation': operation_name,
                    'duration_seconds': duration,
                    'start_time': start_time.isoformat(),
                    'end_time': end_time.isoformat()
                }
            )
        except Exception as e:
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            logger.error(
                f"Operation failed: {operation_name}",
                extra={
                    'type': 'timing',
                    'operation': operation_name,
                    'duration_seconds': duration,
                    'start_time': start_time.isoformat(),
                    'end_time': end_time.isoformat(),
                    'error': str(e)
                },
                exc_info=True
            )
            raise
    
    def set_log_level(self, level: Union[str, int], logger_name: Optional[str] = None):
        """Set log level for a specific logger or root logger."""
        if logger_name:
            logger = self.get_logger(logger_name)
        else:
            logger = self.root_logger
        
        if isinstance(level, str):
            level = getattr(logging, level.upper())
        
        logger.setLevel(level)


# Global logger instance
logger_instance = WorldModelLogger()

# Convenience functions
def get_logger(name: str) -> logging.Logger:
    """Get a logger for a specific module."""
    return logger_instance.get_logger(name)

def start_experiment(experiment_name: str) -> ExperimentLogger:
    """Start a new experiment logging session."""
    return logger_instance.start_experiment(experiment_name)

def get_experiment_logger() -> Optional[ExperimentLogger]:
    """Get the current experiment logger."""
    return logger_instance.get_experiment_logger()

def log_config(config_dict: Dict[str, Any]):
    """Log the current configuration."""
    return logger_instance.log_config(config_dict)

def log_execution_time(operation_name: str, logger_name: Optional[str] = None):
    """Context manager to log execution time of operations."""
    return logger_instance.log_execution_time(operation_name, logger_name)

def set_log_level(level: Union[str, int], logger_name: Optional[str] = None):
    """Set log level for a specific logger or root logger."""
    return logger_instance.set_log_level(level, logger_name)