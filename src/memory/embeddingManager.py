"""
Embedding Manager for WorldModel LLM experiment.

Handles multi-path embedding accumulation, semantic indexing, and clustering
for enhanced model memory and retrieval capabilities.
"""

import numpy as np
import torch
from typing import Dict, List, Any, Optional, Tuple, Set, Union
from dataclasses import dataclass, field
from enum import Enum
import json
import pickle
from pathlib import Path
import time
from datetime import datetime
from collections import defaultdict
import uuid

try:
    from sklearn.cluster import KMeans, DBSCAN
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.decomposition import PCA
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

from ..utils.config import get_config
from ..utils.logging import get_logger

logger = get_logger('embeddingManager')


class EmbeddingType(Enum):
    """Types of embeddings managed by the system."""
    CONVERSATION = "conversation"    # Conversation context embeddings
    CODE_EXECUTION = "code_execution"  # Code execution embeddings  
    PROBLEM_SOLVING = "problem_solving"  # Problem-solving path embeddings
    KNOWLEDGE = "knowledge"          # General knowledge embeddings
    ERROR_PATTERN = "error_pattern"   # Error pattern embeddings
    SUCCESS_PATTERN = "success_pattern"  # Success pattern embeddings


@dataclass
class EmbeddingVector:
    """Container for embedding vectors with metadata."""
    id: str
    vector: np.ndarray
    embedding_type: EmbeddingType
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    tags: Set[str] = field(default_factory=set)
    
    def __post_init__(self):
        if isinstance(self.tags, list):
            self.tags = set(self.tags)
        if self.id is None:
            self.id = str(uuid.uuid4())
    
    @property
    def dimension(self) -> int:
        """Get the dimensionality of the embedding vector."""
        return self.vector.shape[0] if self.vector is not None else 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'id': self.id,
            'vector': self.vector.tolist() if self.vector is not None else None,
            'embedding_type': self.embedding_type.value,
            'content': self.content,
            'metadata': self.metadata,
            'timestamp': self.timestamp,
            'tags': list(self.tags)
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EmbeddingVector':
        """Create from dictionary."""
        return cls(
            id=data['id'],
            vector=np.array(data['vector']) if data['vector'] is not None else None,
            embedding_type=EmbeddingType(data['embedding_type']),
            content=data['content'],
            metadata=data.get('metadata', {}),
            timestamp=data.get('timestamp', time.time()),
            tags=set(data.get('tags', []))
        )


@dataclass 
class EmbeddingCluster:
    """Represents a cluster of similar embeddings."""
    id: str
    centroid: np.ndarray
    embedding_ids: List[str]
    cluster_type: EmbeddingType
    description: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    
    @property
    def size(self) -> int:
        """Number of embeddings in this cluster."""
        return len(self.embedding_ids)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'id': self.id,
            'centroid': self.centroid.tolist() if self.centroid is not None else None,
            'embedding_ids': self.embedding_ids,
            'cluster_type': self.cluster_type.value,
            'description': self.description,
            'metadata': self.metadata,
            'created_at': self.created_at
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EmbeddingCluster':
        """Create from dictionary."""
        return cls(
            id=data['id'],
            centroid=np.array(data['centroid']) if data['centroid'] is not None else None,
            embedding_ids=data['embedding_ids'],
            cluster_type=EmbeddingType(data['cluster_type']),
            description=data.get('description', ''),
            metadata=data.get('metadata', {}),
            created_at=data.get('created_at', time.time())
        )


@dataclass
class SearchResult:
    """Result from embedding search."""
    embedding: EmbeddingVector
    similarity: float
    rank: int


@dataclass
class ClusterAnalysis:
    """Analysis results from clustering embeddings."""
    clusters: List[EmbeddingCluster]
    silhouette_score: float = 0.0
    inertia: float = 0.0
    num_noise_points: int = 0
    algorithm_used: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


class EmbeddingManager:
    """Manages multi-path embedding accumulation and semantic indexing."""
    
    def __init__(self, storage_path: Optional[str] = None, dimension: int = 768):
        self.config = get_config()
        self.logger = get_logger('embeddingManager')
        
        # Storage configuration
        self.storage_path = Path(storage_path) if storage_path else Path(self.config.memory.embedding_storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Embedding storage
        self.embeddings: Dict[str, EmbeddingVector] = {}
        self.clusters: Dict[str, EmbeddingCluster] = {}
        self.type_index: Dict[EmbeddingType, Set[str]] = defaultdict(set)
        self.tag_index: Dict[str, Set[str]] = defaultdict(set)
        
        # Configuration
        self.dimension = dimension
        self.max_embeddings = getattr(self.config.memory, 'max_embeddings', 10000)
        self.auto_cluster_threshold = getattr(self.config.memory, 'auto_cluster_threshold', 100)
        
        # Load existing data
        self._load_embeddings()
        self._build_indices()
        
        self.logger.info(f"EmbeddingManager initialized with {len(self.embeddings)} embeddings")
    
    def add_embedding(self, content: str, vector: np.ndarray, 
                     embedding_type: EmbeddingType, 
                     metadata: Optional[Dict[str, Any]] = None,
                     tags: Optional[Set[str]] = None,
                     embedding_id: Optional[str] = None) -> str:
        """
        Add an embedding to the manager.
        
        Args:
            content: The text content associated with the embedding
            vector: The embedding vector
            embedding_type: Type of embedding
            metadata: Optional metadata
            tags: Optional tags for categorization
            embedding_id: Optional custom ID
            
        Returns:
            The ID of the added embedding
        """
        if metadata is None:
            metadata = {}
        if tags is None:
            tags = set()
        
        # Validate vector
        if vector.shape[0] != self.dimension:
            raise ValueError(f"Vector dimension {vector.shape[0]} doesn't match expected {self.dimension}")
        
        # Create embedding
        embedding = EmbeddingVector(
            id=embedding_id or str(uuid.uuid4()),
            vector=vector,
            embedding_type=embedding_type,
            content=content,
            metadata=metadata,
            tags=tags
        )
        
        # Check for capacity
        if len(self.embeddings) >= self.max_embeddings:
            self._evict_old_embeddings()
        
        # Add to storage
        self.embeddings[embedding.id] = embedding
        self.type_index[embedding_type].add(embedding.id)
        
        for tag in tags:
            self.tag_index[tag].add(embedding.id)
        
        # Auto-cluster if threshold reached
        if len(self.embeddings) % self.auto_cluster_threshold == 0:
            self._auto_cluster()
        
        self.logger.debug(f"Added embedding {embedding.id} of type {embedding_type.value}")
        return embedding.id
    
    def get_embedding(self, embedding_id: str) -> Optional[EmbeddingVector]:
        """Get an embedding by ID."""
        return self.embeddings.get(embedding_id)
    
    def search_similar(self, query_vector: np.ndarray, 
                      top_k: int = 10,
                      embedding_type: Optional[EmbeddingType] = None,
                      tags: Optional[Set[str]] = None,
                      min_similarity: float = 0.0) -> List[SearchResult]:
        """
        Search for similar embeddings.
        
        Args:
            query_vector: Query embedding vector
            top_k: Number of top results to return
            embedding_type: Filter by embedding type
            tags: Filter by tags (intersection)
            min_similarity: Minimum similarity threshold
            
        Returns:
            List of SearchResult ordered by similarity
        """
        # Filter candidates
        candidates = set(self.embeddings.keys())
        
        if embedding_type:
            candidates &= self.type_index[embedding_type]
        
        if tags:
            tag_intersection = None
            for tag in tags:
                tag_embeddings = self.tag_index[tag]
                if tag_intersection is None:
                    tag_intersection = tag_embeddings.copy()
                else:
                    tag_intersection &= tag_embeddings
            
            if tag_intersection is not None:
                candidates &= tag_intersection
        
        if not candidates:
            return []
        
        # Calculate similarities
        similarities = []
        query_norm = np.linalg.norm(query_vector)
        
        for embedding_id in candidates:
            embedding = self.embeddings[embedding_id]
            
            # Cosine similarity
            dot_product = np.dot(query_vector, embedding.vector)
            embedding_norm = np.linalg.norm(embedding.vector)
            
            if embedding_norm > 0 and query_norm > 0:
                similarity = dot_product / (query_norm * embedding_norm)
            else:
                similarity = 0.0
            
            if similarity >= min_similarity:
                similarities.append((embedding_id, similarity))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Create results
        results = []
        for i, (embedding_id, similarity) in enumerate(similarities[:top_k]):
            results.append(SearchResult(
                embedding=self.embeddings[embedding_id],
                similarity=similarity,
                rank=i + 1
            ))
        
        self.logger.debug(f"Search found {len(results)} results from {len(candidates)} candidates")
        return results
    
    def search_by_content(self, query: str, **kwargs) -> List[SearchResult]:
        """Search embeddings by text content using simple text matching."""
        query_lower = query.lower()
        matching_ids = []
        
        for embedding_id, embedding in self.embeddings.items():
            if query_lower in embedding.content.lower():
                matching_ids.append(embedding_id)
        
        # Create pseudo-similarity results based on content match quality
        results = []
        for i, embedding_id in enumerate(matching_ids[:kwargs.get('top_k', 10)]):
            embedding = self.embeddings[embedding_id]
            # Simple similarity based on content length and position of match
            content_lower = embedding.content.lower()
            match_position = content_lower.find(query_lower)
            content_length = len(content_lower)
            
            # Higher similarity for earlier matches and shorter content
            similarity = max(0.1, 1.0 - (match_position / content_length) - (content_length / 1000))
            
            results.append(SearchResult(
                embedding=embedding,
                similarity=similarity,
                rank=i + 1
            ))
        
        return sorted(results, key=lambda x: x.similarity, reverse=True)
    
    def cluster_embeddings(self, embedding_type: Optional[EmbeddingType] = None,
                          algorithm: str = 'kmeans', **kwargs) -> ClusterAnalysis:
        """
        Cluster embeddings using the specified algorithm.
        
        Args:
            embedding_type: Filter by embedding type
            algorithm: Clustering algorithm ('kmeans' or 'dbscan')
            **kwargs: Additional parameters for clustering
            
        Returns:
            ClusterAnalysis with clustering results
        """
        if not SKLEARN_AVAILABLE:
            raise RuntimeError("scikit-learn is required for clustering")
        
        # Get embeddings to cluster
        if embedding_type:
            embedding_ids = list(self.type_index[embedding_type])
        else:
            embedding_ids = list(self.embeddings.keys())
        
        if len(embedding_ids) < 2:
            return ClusterAnalysis(clusters=[])
        
        # Prepare data
        vectors = np.array([self.embeddings[eid].vector for eid in embedding_ids])
        
        # Perform clustering
        if algorithm == 'kmeans':
            n_clusters = kwargs.get('n_clusters', min(8, len(embedding_ids) // 2))
            clusterer = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = clusterer.fit_predict(vectors)
            centroids = clusterer.cluster_centers_
            inertia = clusterer.inertia_
            silhouette_score = 0.0  # Would need to compute separately
            num_noise = 0
            
        elif algorithm == 'dbscan':
            eps = kwargs.get('eps', 0.3)
            min_samples = kwargs.get('min_samples', 2)
            clusterer = DBSCAN(eps=eps, min_samples=min_samples)
            cluster_labels = clusterer.fit_predict(vectors)
            
            # Calculate centroids for each cluster
            unique_labels = set(cluster_labels)
            centroids = []
            for label in unique_labels:
                if label != -1:  # -1 is noise in DBSCAN
                    cluster_vectors = vectors[cluster_labels == label]
                    centroid = np.mean(cluster_vectors, axis=0)
                    centroids.append(centroid)
            
            inertia = 0.0
            silhouette_score = 0.0
            num_noise = np.sum(cluster_labels == -1)
        else:
            raise ValueError(f"Unknown clustering algorithm: {algorithm}")
        
        # Create cluster objects
        clusters = []
        cluster_assignments = defaultdict(list)
        
        for i, label in enumerate(cluster_labels):
            if label != -1:  # Skip noise points in DBSCAN
                cluster_assignments[label].append(embedding_ids[i])
        
        for label, member_ids in cluster_assignments.items():
            if algorithm == 'kmeans':
                centroid = centroids[label]
            else:
                # Find centroid index for DBSCAN
                centroid_idx = label if label < len(centroids) else 0
                centroid = centroids[centroid_idx] if centroids else np.zeros(self.dimension)
            
            cluster = EmbeddingCluster(
                id=f"cluster_{algorithm}_{label}_{int(time.time())}",
                centroid=centroid,
                embedding_ids=member_ids,
                cluster_type=embedding_type or EmbeddingType.KNOWLEDGE,
                description=f"{algorithm} cluster {label} with {len(member_ids)} embeddings"
            )
            clusters.append(cluster)
        
        # Store clusters
        for cluster in clusters:
            self.clusters[cluster.id] = cluster
        
        analysis = ClusterAnalysis(
            clusters=clusters,
            silhouette_score=silhouette_score,
            inertia=inertia,
            num_noise_points=num_noise,
            algorithm_used=algorithm,
            metadata={'total_embeddings': len(embedding_ids)}
        )
        
        self.logger.info(f"Clustered {len(embedding_ids)} embeddings into {len(clusters)} clusters using {algorithm}")
        return analysis
    
    def get_cluster_summary(self, cluster_id: str) -> Optional[Dict[str, Any]]:
        """Get a summary of a cluster including representative examples."""
        cluster = self.clusters.get(cluster_id)
        if not cluster:
            return None
        
        # Get embeddings in cluster
        cluster_embeddings = [self.embeddings[eid] for eid in cluster.embedding_ids 
                            if eid in self.embeddings]
        
        if not cluster_embeddings:
            return None
        
        # Find most representative embeddings (closest to centroid)
        similarities = []
        for embedding in cluster_embeddings:
            sim = np.dot(embedding.vector, cluster.centroid) / (
                np.linalg.norm(embedding.vector) * np.linalg.norm(cluster.centroid)
            )
            similarities.append((embedding, sim))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Get top examples
        top_examples = [emb.content[:200] for emb, _ in similarities[:3]]
        
        # Count by type
        type_counts = defaultdict(int)
        for embedding in cluster_embeddings:
            type_counts[embedding.embedding_type.value] += 1
        
        return {
            'cluster_id': cluster_id,
            'size': cluster.size,
            'cluster_type': cluster.cluster_type.value,
            'description': cluster.description,
            'type_distribution': dict(type_counts),
            'representative_examples': top_examples,
            'created_at': cluster.created_at
        }
    
    def accumulate_path_embeddings(self, path_embeddings: List[np.ndarray],
                                 path_metadata: Dict[str, Any]) -> str:
        """
        Accumulate embeddings from a multi-step reasoning path.
        
        Args:
            path_embeddings: List of embedding vectors from reasoning steps
            path_metadata: Metadata about the reasoning path
            
        Returns:
            ID of the accumulated path embedding
        """
        if not path_embeddings:
            raise ValueError("No embeddings provided for path accumulation")
        
        # Simple accumulation: average the embeddings
        accumulated_vector = np.mean(path_embeddings, axis=0)
        
        # Create summary content
        step_count = len(path_embeddings)
        content = f"Reasoning path with {step_count} steps"
        if 'problem' in path_metadata:
            content += f": {path_metadata['problem'][:100]}"
        
        # Add path embedding
        return self.add_embedding(
            content=content,
            vector=accumulated_vector,
            embedding_type=EmbeddingType.PROBLEM_SOLVING,
            metadata={
                'path_length': step_count,
                'accumulation_method': 'average',
                **path_metadata
            },
            tags={'reasoning_path', 'accumulated'}
        )
    
    def _evict_old_embeddings(self):
        """Evict oldest embeddings to make room for new ones."""
        # Remove 10% of oldest embeddings
        num_to_remove = max(1, len(self.embeddings) // 10)
        
        # Sort by timestamp
        sorted_embeddings = sorted(
            self.embeddings.items(),
            key=lambda x: x[1].timestamp
        )
        
        for embedding_id, _ in sorted_embeddings[:num_to_remove]:
            self._remove_embedding(embedding_id)
        
        self.logger.info(f"Evicted {num_to_remove} old embeddings")
    
    def _remove_embedding(self, embedding_id: str):
        """Remove an embedding and update indices."""
        embedding = self.embeddings.get(embedding_id)
        if not embedding:
            return
        
        # Remove from main storage
        del self.embeddings[embedding_id]
        
        # Update indices
        self.type_index[embedding.embedding_type].discard(embedding_id)
        
        for tag in embedding.tags:
            self.tag_index[tag].discard(embedding_id)
            if not self.tag_index[tag]:
                del self.tag_index[tag]
        
        # Remove from clusters
        for cluster in self.clusters.values():
            if embedding_id in cluster.embedding_ids:
                cluster.embedding_ids.remove(embedding_id)
    
    def _auto_cluster(self):
        """Automatically cluster embeddings when threshold is reached."""
        try:
            # Cluster each type separately if there are enough embeddings
            for embedding_type in EmbeddingType:
                type_embeddings = self.type_index[embedding_type]
                if len(type_embeddings) >= 10:  # Minimum for clustering
                    self.cluster_embeddings(
                        embedding_type=embedding_type,
                        algorithm='kmeans',
                        n_clusters=min(5, len(type_embeddings) // 3)
                    )
        except Exception as e:
            self.logger.warning(f"Auto-clustering failed: {e}")
    
    def _build_indices(self):
        """Build type and tag indices from loaded embeddings."""
        self.type_index.clear()
        self.tag_index.clear()
        
        for embedding_id, embedding in self.embeddings.items():
            self.type_index[embedding.embedding_type].add(embedding_id)
            
            for tag in embedding.tags:
                self.tag_index[tag].add(embedding_id)
    
    def save_to_disk(self):
        """Save all embeddings and clusters to disk."""
        embeddings_file = self.storage_path / 'embeddings.pkl'
        clusters_file = self.storage_path / 'clusters.pkl'
        
        try:
            # Save embeddings
            with open(embeddings_file, 'wb') as f:
                pickle.dump({
                    'embeddings': {eid: emb.to_dict() for eid, emb in self.embeddings.items()},
                    'dimension': self.dimension
                }, f)
            
            # Save clusters  
            with open(clusters_file, 'wb') as f:
                pickle.dump({
                    'clusters': {cid: cluster.to_dict() for cid, cluster in self.clusters.items()}
                }, f)
            
            self.logger.info(f"Saved {len(self.embeddings)} embeddings and {len(self.clusters)} clusters")
            
        except Exception as e:
            self.logger.error(f"Failed to save embeddings: {e}")
    
    def _load_embeddings(self):
        """Load embeddings and clusters from disk."""
        embeddings_file = self.storage_path / 'embeddings.pkl'
        clusters_file = self.storage_path / 'clusters.pkl'
        
        try:
            # Load embeddings
            if embeddings_file.exists():
                with open(embeddings_file, 'rb') as f:
                    data = pickle.load(f)
                    self.embeddings = {
                        eid: EmbeddingVector.from_dict(emb_data) 
                        for eid, emb_data in data['embeddings'].items()
                    }
                    self.dimension = data.get('dimension', self.dimension)
            
            # Load clusters
            if clusters_file.exists():
                with open(clusters_file, 'rb') as f:
                    data = pickle.load(f)
                    self.clusters = {
                        cid: EmbeddingCluster.from_dict(cluster_data)
                        for cid, cluster_data in data['clusters'].items()
                    }
            
            self.logger.info(f"Loaded {len(self.embeddings)} embeddings and {len(self.clusters)} clusters")
            
        except Exception as e:
            self.logger.warning(f"Failed to load embeddings: {e}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the embedding manager."""
        type_counts = {et.value: len(self.type_index[et]) for et in EmbeddingType}
        
        return {
            'total_embeddings': len(self.embeddings),
            'total_clusters': len(self.clusters),
            'dimension': self.dimension,
            'max_capacity': self.max_embeddings,
            'type_distribution': type_counts,
            'total_tags': len(self.tag_index),
            'storage_path': str(self.storage_path)
        }


# Convenience functions
def create_embedding_manager(storage_path: Optional[str] = None, 
                           dimension: int = 768) -> EmbeddingManager:
    """Create an embedding manager with the specified configuration."""
    return EmbeddingManager(storage_path=storage_path, dimension=dimension)

def add_conversation_embedding(content: str, vector: np.ndarray, 
                             metadata: Optional[Dict] = None) -> str:
    """Quick function to add a conversation embedding."""
    manager = create_embedding_manager()
    return manager.add_embedding(
        content=content,
        vector=vector,
        embedding_type=EmbeddingType.CONVERSATION,
        metadata=metadata or {}
    )