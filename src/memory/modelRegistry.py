import json
import shutil
import hashlib
from datetime import datetime, timezone
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple
import asyncio
import logging

from ..utils.config import MemoryConfig
from ..utils.logging import get_logger


@dataclass
class ModelVersion:
    version_id: str
    model_name: str
    base_model: str
    training_method: str
    performance_metrics: Dict[str, float]
    metadata: Dict[str, Any]
    file_path: str
    file_size: int
    file_hash: str
    created_at: datetime
    tags: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['created_at'] = self.created_at.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelVersion':
        data = data.copy()
        data['created_at'] = datetime.fromisoformat(data['created_at'])
        return cls(**data)

@dataclass
class ModelInfo:
    name: str
    description: str
    base_model: str
    current_version: str
    versions: List[str]
    created_at: datetime
    updated_at: datetime
    total_versions: int
    best_version: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['created_at'] = self.created_at.isoformat()
        data['updated_at'] = self.updated_at.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelInfo':
        data = data.copy()
        data['created_at'] = datetime.fromisoformat(data['created_at'])
        data['updated_at'] = datetime.fromisoformat(data['updated_at'])
        return cls(**data)

@dataclass
class RegistryStats:
    total_models: int
    total_versions: int
    total_size_bytes: int
    models_by_base: Dict[str, int]
    versions_by_method: Dict[str, int]
    recent_activity: List[Dict[str, Any]]

class ModelRegistry:
    def __init__(self, config: MemoryConfig):
        self.config = config
        self.logger = get_logger(__name__)
        
        self.registry_path = Path(config.model_registry_path)
        self.models_path = self.registry_path / "models"
        self.metadata_path = self.registry_path / "metadata"
        
        self._initialize_registry()
        
        # In-memory caches
        self._model_cache: Dict[str, ModelInfo] = {}
        self._version_cache: Dict[str, ModelVersion] = {}
        self._cache_dirty = True
    
    def _initialize_registry(self):
        self.registry_path.mkdir(parents=True, exist_ok=True)
        self.models_path.mkdir(parents=True, exist_ok=True)
        self.metadata_path.mkdir(parents=True, exist_ok=True)
        
        # Create registry index if it doesn't exist
        index_file = self.metadata_path / "registry_index.json"
        if not index_file.exists():
            with open(index_file, 'w') as f:
                json.dump({'models': {}, 'versions': {}, 'created_at': datetime.now(timezone.utc).isoformat()}, f)
        
        self.logger.info(f"Model registry initialized at {self.registry_path}")
    
    def _load_cache(self):
        if not self._cache_dirty:
            return
        
        try:
            index_file = self.metadata_path / "registry_index.json"
            with open(index_file, 'r') as f:
                index_data = json.load(f)
            
            # Load models
            for model_name, model_file in index_data.get('models', {}).items():
                try:
                    model_path = self.metadata_path / model_file
                    with open(model_path, 'r') as f:
                        model_data = json.load(f)
                    self._model_cache[model_name] = ModelInfo.from_dict(model_data)
                except Exception as e:
                    self.logger.error(f"Failed to load model {model_name}: {e}")
            
            # Load versions
            for version_id, version_file in index_data.get('versions', {}).items():
                try:
                    version_path = self.metadata_path / version_file
                    with open(version_path, 'r') as f:
                        version_data = json.load(f)
                    self._version_cache[version_id] = ModelVersion.from_dict(version_data)
                except Exception as e:
                    self.logger.error(f"Failed to load version {version_id}: {e}")
            
            self._cache_dirty = False
            self.logger.debug(f"Loaded {len(self._model_cache)} models and {len(self._version_cache)} versions")
            
        except Exception as e:
            self.logger.error(f"Failed to load registry cache: {e}")
    
    def _save_index(self):
        try:
            index_data = {
                'models': {name: f"model_{name}.json" for name in self._model_cache.keys()},
                'versions': {version_id: f"version_{version_id}.json" for version_id in self._version_cache.keys()},
                'created_at': datetime.now(timezone.utc).isoformat(),
                'updated_at': datetime.now(timezone.utc).isoformat()
            }
            
            index_file = self.metadata_path / "registry_index.json"
            with open(index_file, 'w') as f:
                json.dump(index_data, f, indent=2)
            
        except Exception as e:
            self.logger.error(f"Failed to save registry index: {e}")
    
    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA-256 hash of a file."""
        hash_sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()
    
    def _generate_version_id(self, model_name: str, timestamp: datetime = None) -> str:
        """Generate a unique version ID."""
        if timestamp is None:
            timestamp = datetime.now(timezone.utc)
        
        # Use model name and timestamp to create unique ID
        id_string = f"{model_name}_{timestamp.isoformat()}"
        return hashlib.md5(id_string.encode()).hexdigest()[:16]
    
    async def register_model(self, model_name: str, description: str, 
                           base_model: str) -> ModelInfo:
        """Register a new model in the registry."""
        self._load_cache()
        
        if model_name in self._model_cache:
            raise ValueError(f"Model '{model_name}' already exists")
        
        now = datetime.now(timezone.utc)
        model_info = ModelInfo(
            name=model_name,
            description=description,
            base_model=base_model,
            current_version="",
            versions=[],
            created_at=now,
            updated_at=now,
            total_versions=0
        )
        
        # Save to cache and disk
        self._model_cache[model_name] = model_info
        
        model_file = self.metadata_path / f"model_{model_name}.json"
        with open(model_file, 'w') as f:
            json.dump(model_info.to_dict(), f, indent=2)
        
        self._save_index()
        
        self.logger.info(f"Registered new model: {model_name}")
        return model_info
    
    async def add_version(self, model_name: str, model_file_path: str,
                         training_method: str, performance_metrics: Dict[str, float] = None,
                         metadata: Dict[str, Any] = None, tags: List[str] = None) -> ModelVersion:
        """Add a new version to an existing model."""
        self._load_cache()
        
        if model_name not in self._model_cache:
            raise ValueError(f"Model '{model_name}' not found")
        
        source_path = Path(model_file_path)
        if not source_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_file_path}")
        
        # Generate version ID and copy file
        now = datetime.now(timezone.utc)
        version_id = self._generate_version_id(model_name, now)
        
        # Create version directory and copy file
        version_dir = self.models_path / model_name / version_id
        version_dir.mkdir(parents=True, exist_ok=True)
        
        target_path = version_dir / source_path.name
        shutil.copy2(source_path, target_path)
        
        # Calculate file metadata
        file_size = target_path.stat().st_size
        file_hash = self._calculate_file_hash(target_path)
        
        # Create version object
        model_info = self._model_cache[model_name]
        model_version = ModelVersion(
            version_id=version_id,
            model_name=model_name,
            base_model=model_info.base_model,
            training_method=training_method,
            performance_metrics=performance_metrics or {},
            metadata=metadata or {},
            file_path=str(target_path),
            file_size=file_size,
            file_hash=file_hash,
            created_at=now,
            tags=tags or []
        )
        
        # Update model info
        model_info.versions.append(version_id)
        model_info.current_version = version_id
        model_info.total_versions += 1
        model_info.updated_at = now
        
        # Update best version based on performance metrics
        if performance_metrics and 'accuracy' in performance_metrics:
            current_best = self.get_best_version(model_name)
            if (current_best is None or 
                performance_metrics['accuracy'] > current_best.performance_metrics.get('accuracy', 0)):
                model_info.best_version = version_id
        
        # Save to cache and disk
        self._version_cache[version_id] = model_version
        
        version_file = self.metadata_path / f"version_{version_id}.json"
        with open(version_file, 'w') as f:
            json.dump(model_version.to_dict(), f, indent=2)
        
        model_file = self.metadata_path / f"model_{model_name}.json"
        with open(model_file, 'w') as f:
            json.dump(model_info.to_dict(), f, indent=2)
        
        self._save_index()
        
        self.logger.info(f"Added version {version_id} to model {model_name}")
        return model_version
    
    def get_model(self, model_name: str) -> Optional[ModelInfo]:
        """Get model information."""
        self._load_cache()
        return self._model_cache.get(model_name)
    
    def get_version(self, version_id: str) -> Optional[ModelVersion]:
        """Get specific model version."""
        self._load_cache()
        return self._version_cache.get(version_id)
    
    def get_model_versions(self, model_name: str) -> List[ModelVersion]:
        """Get all versions for a model."""
        self._load_cache()
        
        model_info = self._model_cache.get(model_name)
        if not model_info:
            return []
        
        versions = []
        for version_id in model_info.versions:
            version = self._version_cache.get(version_id)
            if version:
                versions.append(version)
        
        # Sort by creation date (newest first)
        versions.sort(key=lambda v: v.created_at, reverse=True)
        return versions
    
    def get_latest_version(self, model_name: str) -> Optional[ModelVersion]:
        """Get the latest version of a model."""
        model_info = self.get_model(model_name)
        if not model_info or not model_info.current_version:
            return None
        
        return self.get_version(model_info.current_version)
    
    def get_best_version(self, model_name: str) -> Optional[ModelVersion]:
        """Get the best performing version of a model."""
        model_info = self.get_model(model_name)
        if not model_info or not model_info.best_version:
            return None
        
        return self.get_version(model_info.best_version)
    
    def list_models(self) -> List[ModelInfo]:
        """List all registered models."""
        self._load_cache()
        return list(self._model_cache.values())
    
    def search_models(self, query: str = None, base_model: str = None,
                     tags: List[str] = None) -> List[ModelInfo]:
        """Search models by various criteria."""
        self._load_cache()
        
        results = []
        for model_info in self._model_cache.values():
            # Check query match
            if query:
                if (query.lower() not in model_info.name.lower() and
                    query.lower() not in model_info.description.lower()):
                    continue
            
            # Check base model match
            if base_model and model_info.base_model != base_model:
                continue
            
            # Check tags match
            if tags:
                # Get tags from all versions
                model_tags = set()
                for version_id in model_info.versions:
                    version = self._version_cache.get(version_id)
                    if version:
                        model_tags.update(version.tags)
                
                if not any(tag in model_tags for tag in tags):
                    continue
            
            results.append(model_info)
        
        return results
    
    async def delete_model(self, model_name: str) -> bool:
        """Delete a model and all its versions."""
        self._load_cache()
        
        model_info = self._model_cache.get(model_name)
        if not model_info:
            return False
        
        try:
            # Delete all version files and metadata
            for version_id in model_info.versions:
                version = self._version_cache.get(version_id)
                if version:
                    # Delete model file
                    file_path = Path(version.file_path)
                    if file_path.exists():
                        file_path.unlink()
                    
                    # Delete version directory if empty
                    version_dir = file_path.parent
                    if version_dir.exists() and not any(version_dir.iterdir()):
                        version_dir.rmdir()
                    
                    # Delete metadata file
                    version_metadata = self.metadata_path / f"version_{version_id}.json"
                    if version_metadata.exists():
                        version_metadata.unlink()
                    
                    # Remove from cache
                    del self._version_cache[version_id]
            
            # Delete model directory if empty
            model_dir = self.models_path / model_name
            if model_dir.exists() and not any(model_dir.iterdir()):
                model_dir.rmdir()
            
            # Delete model metadata
            model_metadata = self.metadata_path / f"model_{model_name}.json"
            if model_metadata.exists():
                model_metadata.unlink()
            
            # Remove from cache
            del self._model_cache[model_name]
            
            self._save_index()
            
            self.logger.info(f"Deleted model: {model_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error deleting model {model_name}: {e}")
            return False
    
    async def delete_version(self, version_id: str) -> bool:
        """Delete a specific model version."""
        self._load_cache()
        
        version = self._version_cache.get(version_id)
        if not version:
            return False
        
        try:
            model_info = self._model_cache.get(version.model_name)
            if not model_info:
                return False
            
            # Delete model file
            file_path = Path(version.file_path)
            if file_path.exists():
                file_path.unlink()
            
            # Delete version directory if empty
            version_dir = file_path.parent
            if version_dir.exists() and not any(version_dir.iterdir()):
                version_dir.rmdir()
            
            # Delete metadata file
            version_metadata = self.metadata_path / f"version_{version_id}.json"
            if version_metadata.exists():
                version_metadata.unlink()
            
            # Update model info
            model_info.versions.remove(version_id)
            model_info.total_versions -= 1
            
            # Update current version if this was it
            if model_info.current_version == version_id:
                if model_info.versions:
                    # Set to latest remaining version
                    remaining_versions = self.get_model_versions(version.model_name)
                    if remaining_versions:
                        model_info.current_version = remaining_versions[0].version_id
                else:
                    model_info.current_version = ""
            
            # Update best version if this was it
            if model_info.best_version == version_id:
                model_info.best_version = None
                # Find new best version
                remaining_versions = self.get_model_versions(version.model_name)
                best_accuracy = -1
                for v in remaining_versions:
                    accuracy = v.performance_metrics.get('accuracy', -1)
                    if accuracy > best_accuracy:
                        best_accuracy = accuracy
                        model_info.best_version = v.version_id
            
            # Save updated model info
            model_file = self.metadata_path / f"model_{version.model_name}.json"
            with open(model_file, 'w') as f:
                json.dump(model_info.to_dict(), f, indent=2)
            
            # Remove from cache
            del self._version_cache[version_id]
            
            self._save_index()
            
            self.logger.info(f"Deleted version: {version_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error deleting version {version_id}: {e}")
            return False
    
    def get_stats(self) -> RegistryStats:
        """Get registry statistics."""
        self._load_cache()
        
        total_size = 0
        models_by_base = {}
        versions_by_method = {}
        recent_activity = []
        
        # Calculate statistics
        for version in self._version_cache.values():
            total_size += version.file_size
            
            # Count by base model
            base = version.base_model
            models_by_base[base] = models_by_base.get(base, 0) + 1
            
            # Count by training method
            method = version.training_method
            versions_by_method[method] = versions_by_method.get(method, 0) + 1
            
            # Recent activity (last 10 versions)
            recent_activity.append({
                'version_id': version.version_id,
                'model_name': version.model_name,
                'training_method': version.training_method,
                'created_at': version.created_at.isoformat(),
                'file_size': version.file_size
            })
        
        # Sort recent activity by creation time
        recent_activity.sort(key=lambda x: x['created_at'], reverse=True)
        recent_activity = recent_activity[:10]
        
        return RegistryStats(
            total_models=len(self._model_cache),
            total_versions=len(self._version_cache),
            total_size_bytes=total_size,
            models_by_base=models_by_base,
            versions_by_method=versions_by_method,
            recent_activity=recent_activity
        )

# Convenience functions
async def create_model_registry(config: MemoryConfig) -> ModelRegistry:
    """Create a new model registry instance."""
    return ModelRegistry(config)

async def quick_register(registry: ModelRegistry, model_name: str, 
                        model_file: str, description: str = "",
                        base_model: str = "unknown") -> str:
    """Quickly register a model and add its first version."""
    # Register model if it doesn't exist
    try:
        model_info = registry.get_model(model_name)
        if not model_info:
            await registry.register_model(model_name, description, base_model)
    except ValueError:
        pass  # Model already exists
    
    # Add version
    version = await registry.add_version(
        model_name=model_name,
        model_file_path=model_file,
        training_method="unknown",
        performance_metrics={},
        metadata={'description': description}
    )
    
    return version.version_id