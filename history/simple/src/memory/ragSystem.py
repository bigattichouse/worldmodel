import asyncio
import json
import pickle
import numpy as np
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any, Tuple, Union
from pathlib import Path
import sqlite3
from datetime import datetime, timezone
import hashlib
import logging

from ..utils.config import MemoryConfig
from ..utils.logging import get_logger


@dataclass
class Document:
    content: str
    metadata: Dict[str, Any]
    doc_id: Optional[str] = None
    embedding: Optional[np.ndarray] = None
    created_at: Optional[datetime] = None
    
    def __post_init__(self):
        if self.doc_id is None:
            self.doc_id = self._generate_id()
        if self.created_at is None:
            self.created_at = datetime.now(timezone.utc)
    
    def _generate_id(self) -> str:
        content_hash = hashlib.md5(self.content.encode('utf-8')).hexdigest()
        metadata_hash = hashlib.md5(json.dumps(self.metadata, sort_keys=True).encode('utf-8')).hexdigest()
        return f"{content_hash[:8]}_{metadata_hash[:8]}"

@dataclass
class SearchResult:
    document: Document
    similarity_score: float
    rank: int

@dataclass
class SearchQuery:
    text: str
    filters: Optional[Dict[str, Any]] = None
    k: int = 5
    min_similarity: float = 0.3  # Match config default

class EmbeddingGenerator:
    def __init__(self, config: MemoryConfig):
        self.config = config
        self.logger = get_logger(__name__)
        self.model = None
        self.tokenizer = None
        self._initialize_model()
    
    def _initialize_model(self):
        try:
            if self.config.embedding_model == "sentence-transformers":
                from sentence_transformers import SentenceTransformer
                self.model = SentenceTransformer(self.config.embedding_model)
            elif self.config.embedding_model == "huggingface":
                from transformers import AutoModel, AutoTokenizer
                import torch
                model_path = "../model"  # Use local Gemma model
                self.tokenizer = AutoTokenizer.from_pretrained(model_path)
                self.model = AutoModel.from_pretrained(model_path)
                self.model.eval()
            else:
                raise ValueError(f"Unsupported embedding model: {self.config.embedding_model}")
                
            self.logger.info(f"Initialized embedding model: {self.config.embedding_model}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize embedding model: {e}")
            self.model = None
    
    def generate_embedding(self, text: str) -> np.ndarray:
        if self.model is None:
            return self._fallback_embedding(text)
        
        try:
            if self.config.embedding_model == "sentence-transformers":
                embedding = self.model.encode([text])[0]
                return embedding.astype(np.float32)
            
            elif self.config.embedding_model == "huggingface":
                import torch
                inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
                
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    embeddings = outputs.last_hidden_state.mean(dim=1)
                    return embeddings.squeeze().numpy().astype(np.float32)
            
        except Exception as e:
            self.logger.error(f"Error generating embedding: {e}")
            return self._fallback_embedding(text)
    
    def _fallback_embedding(self, text: str) -> np.ndarray:
        # Simple hash-based fallback embedding
        hash_val = hashlib.md5(text.encode('utf-8')).hexdigest()
        # Convert hex to numbers and normalize
        embedding = np.array([int(hash_val[i:i+2], 16) for i in range(0, len(hash_val), 2)])
        embedding = embedding.astype(np.float32) / 255.0
        
        # Pad or truncate to desired dimension
        target_dim = self.config.embedding_dim
        if len(embedding) < target_dim:
            embedding = np.pad(embedding, (0, target_dim - len(embedding)), 'constant')
        else:
            embedding = embedding[:target_dim]
        
        return embedding

class VectorDatabase:
    def __init__(self, config: MemoryConfig):
        self.config = config
        self.logger = get_logger(__name__)
        self.db_path = Path(config.vector_db_path) / "vector_db.sqlite"
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize_database()
    
    def _initialize_database(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id TEXT PRIMARY KEY,
            content TEXT NOT NULL,
            metadata TEXT NOT NULL,
            created_at TEXT NOT NULL,
            embedding_hash TEXT
        )
        """)
        
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS embeddings (
            doc_id TEXT PRIMARY KEY,
            embedding BLOB NOT NULL,
            dimension INTEGER NOT NULL,
            FOREIGN KEY (doc_id) REFERENCES documents (id)
        )
        """)
        
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_created_at ON documents(created_at)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_metadata ON documents(metadata)")
        
        conn.commit()
        conn.close()
        
        self.logger.info(f"Vector database initialized at {self.db_path}")
    
    def store_document(self, document: Document) -> bool:
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Store document metadata
            cursor.execute("""
            INSERT OR REPLACE INTO documents (id, content, metadata, created_at, embedding_hash)
            VALUES (?, ?, ?, ?, ?)
            """, (
                document.doc_id,
                document.content,
                json.dumps(document.metadata),
                document.created_at.isoformat(),
                self._hash_embedding(document.embedding) if document.embedding is not None else None
            ))
            
            # Store embedding if available
            if document.embedding is not None:
                embedding_blob = pickle.dumps(document.embedding)
                cursor.execute("""
                INSERT OR REPLACE INTO embeddings (doc_id, embedding, dimension)
                VALUES (?, ?, ?)
                """, (
                    document.doc_id,
                    embedding_blob,
                    len(document.embedding)
                ))
            
            conn.commit()
            conn.close()
            
            self.logger.debug(f"Stored document: {document.doc_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error storing document: {e}")
            return False
    
    def get_document(self, doc_id: str) -> Optional[Document]:
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
            SELECT d.content, d.metadata, d.created_at, e.embedding
            FROM documents d
            LEFT JOIN embeddings e ON d.id = e.doc_id
            WHERE d.id = ?
            """, (doc_id,))
            
            result = cursor.fetchone()
            conn.close()
            
            if result:
                content, metadata_json, created_at_str, embedding_blob = result
                metadata = json.loads(metadata_json)
                created_at = datetime.fromisoformat(created_at_str)
                
                embedding = None
                if embedding_blob:
                    embedding = pickle.loads(embedding_blob)
                
                return Document(
                    content=content,
                    metadata=metadata,
                    doc_id=doc_id,
                    embedding=embedding,
                    created_at=created_at
                )
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error retrieving document {doc_id}: {e}")
            return None
    
    def search_documents(self, query_embedding: np.ndarray, 
                        filters: Optional[Dict[str, Any]] = None,
                        k: int = 5, min_similarity: float = 0.3) -> List[Tuple[str, float]]:
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Build filter query
            base_query = """
            SELECT d.id, e.embedding
            FROM documents d
            JOIN embeddings e ON d.id = e.doc_id
            """
            
            where_clauses = []
            params = []
            
            if filters:
                for key, value in filters.items():
                    where_clauses.append("d.metadata LIKE ?")
                    params.append(f'%"{key}":"{value}"%')
            
            if where_clauses:
                base_query += " WHERE " + " AND ".join(where_clauses)
            
            cursor.execute(base_query, params)
            results = cursor.fetchall()
            conn.close()
            
            # Calculate similarities
            similarities = []
            for doc_id, embedding_blob in results:
                doc_embedding = pickle.loads(embedding_blob)
                similarity = self._cosine_similarity(query_embedding, doc_embedding)
                
                if similarity >= min_similarity:
                    similarities.append((doc_id, similarity))
            
            # Sort by similarity and return top k
            similarities.sort(key=lambda x: x[1], reverse=True)
            return similarities[:k]
            
        except Exception as e:
            self.logger.error(f"Error searching documents: {e}")
            return []
    
    def delete_document(self, doc_id: str) -> bool:
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("DELETE FROM embeddings WHERE doc_id = ?", (doc_id,))
            cursor.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
            
            conn.commit()
            deleted = cursor.rowcount > 0
            conn.close()
            
            if deleted:
                self.logger.debug(f"Deleted document: {doc_id}")
            
            return deleted
            
        except Exception as e:
            self.logger.error(f"Error deleting document {doc_id}: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("SELECT COUNT(*) FROM documents")
            total_docs = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM embeddings")
            total_embeddings = cursor.fetchone()[0]
            
            cursor.execute("""
            SELECT AVG(LENGTH(content)), MAX(LENGTH(content)), MIN(LENGTH(content))
            FROM documents
            """)
            content_stats = cursor.fetchone()
            
            conn.close()
            
            return {
                'total_documents': total_docs,
                'total_embeddings': total_embeddings,
                'avg_content_length': content_stats[0] if content_stats[0] else 0,
                'max_content_length': content_stats[1] if content_stats[1] else 0,
                'min_content_length': content_stats[2] if content_stats[2] else 0,
                'database_size': self.db_path.stat().st_size
            }
            
        except Exception as e:
            self.logger.error(f"Error getting database stats: {e}")
            return {}
    
    def _hash_embedding(self, embedding: np.ndarray) -> str:
        return hashlib.md5(embedding.tobytes()).hexdigest()
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        return dot_product / (norm_a * norm_b)

class RAGSystem:
    def __init__(self, config: MemoryConfig):
        self.config = config
        self.logger = get_logger(__name__)
        
        self.embedding_generator = EmbeddingGenerator(config)
        self.vector_db = VectorDatabase(config)
        
        self.logger.info("RAG System initialized")
    
    async def add_document(self, content: str, metadata: Dict[str, Any] = None) -> str:
        if metadata is None:
            metadata = {}
        
        # Create document
        document = Document(content=content, metadata=metadata)
        
        # Generate embedding
        try:
            document.embedding = await asyncio.get_event_loop().run_in_executor(
                None, self.embedding_generator.generate_embedding, content
            )
        except Exception as e:
            self.logger.error(f"Failed to generate embedding for document: {e}")
            # Store document without embedding
            document.embedding = None
        
        # Store in database
        success = self.vector_db.store_document(document)
        
        if success:
            self.logger.info(f"Added document: {document.doc_id}")
            return document.doc_id
        else:
            raise RuntimeError("Failed to store document in vector database")
    
    async def search(self, query: SearchQuery) -> List[SearchResult]:
        # Generate query embedding
        try:
            query_embedding = await asyncio.get_event_loop().run_in_executor(
                None, self.embedding_generator.generate_embedding, query.text
            )
        except Exception as e:
            self.logger.error(f"Failed to generate query embedding: {e}")
            return []
        
        # Search vector database
        similar_docs = self.vector_db.search_documents(
            query_embedding,
            filters=query.filters,
            k=query.k,
            min_similarity=query.min_similarity
        )
        
        # Convert to SearchResult objects
        results = []
        for rank, (doc_id, similarity) in enumerate(similar_docs):
            document = self.vector_db.get_document(doc_id)
            if document:
                result = SearchResult(
                    document=document,
                    similarity_score=similarity,
                    rank=rank
                )
                results.append(result)
        
        self.logger.info(f"Search for '{query.text}' returned {len(results)} results")
        return results
    
    async def get_document(self, doc_id: str) -> Optional[Document]:
        return self.vector_db.get_document(doc_id)
    
    async def delete_document(self, doc_id: str) -> bool:
        return self.vector_db.delete_document(doc_id)
    
    async def update_document(self, doc_id: str, content: str = None, 
                            metadata: Dict[str, Any] = None) -> bool:
        # Get existing document
        existing_doc = self.vector_db.get_document(doc_id)
        if not existing_doc:
            return False
        
        # Update content and metadata
        new_content = content if content is not None else existing_doc.content
        new_metadata = metadata if metadata is not None else existing_doc.metadata
        
        # Create updated document
        updated_doc = Document(
            content=new_content,
            metadata=new_metadata,
            doc_id=doc_id,
            created_at=existing_doc.created_at
        )
        
        # Regenerate embedding if content changed
        if content is not None:
            try:
                updated_doc.embedding = await asyncio.get_event_loop().run_in_executor(
                    None, self.embedding_generator.generate_embedding, new_content
                )
            except Exception as e:
                self.logger.error(f"Failed to regenerate embedding: {e}")
                updated_doc.embedding = existing_doc.embedding
        else:
            updated_doc.embedding = existing_doc.embedding
        
        # Store updated document
        success = self.vector_db.store_document(updated_doc)
        
        if success:
            self.logger.info(f"Updated document: {doc_id}")
        
        return success
    
    async def add_model_experience(self, input_text: str, output_text: str, 
                                 execution_result: Optional[Dict[str, Any]] = None,
                                 tags: Optional[Dict[str, Any]] = None) -> str:
        # Create comprehensive metadata for model experience
        metadata = {
            'type': 'model_experience',
            'input_length': len(input_text),
            'output_length': len(output_text),
            'has_execution': execution_result is not None,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
        
        if tags:
            metadata['tags'] = tags
        
        if execution_result:
            metadata['execution_success'] = execution_result.get('success', False)
            metadata['execution_language'] = execution_result.get('language')
        
        # Combine input and output for content
        content = f"INPUT: {input_text}\n\nOUTPUT: {output_text}"
        
        if execution_result and execution_result.get('output'):
            content += f"\n\nEXECUTION: {execution_result['output']}"
        
        return await self.add_document(content, metadata)
    
    async def search_similar_experiences(self, query_text: str, 
                                       experience_type: str = None,
                                       k: int = 3) -> List[SearchResult]:
        filters = {'type': 'model_experience'}
        if experience_type:
            filters['execution_language'] = experience_type
        
        search_query = SearchQuery(
            text=query_text,
            filters=filters,
            k=k,
            min_similarity=0.6
        )
        
        return await self.search(search_query)
    
    def get_stats(self) -> Dict[str, Any]:
        base_stats = self.vector_db.get_stats()
        base_stats['embedding_model'] = self.config.embedding_model
        base_stats['embedding_dimension'] = self.config.embedding_dim
        return base_stats

# Convenience functions
async def create_rag_system(config: MemoryConfig) -> RAGSystem:
    return RAGSystem(config)

async def quick_search(rag: RAGSystem, query: str, k: int = 5) -> List[str]:
    search_query = SearchQuery(text=query, k=k)
    results = await rag.search(search_query)
    return [result.document.content for result in results]