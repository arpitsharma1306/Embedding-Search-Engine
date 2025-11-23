"""
Cache Manager
Handles caching of document embeddings using SQLite for persistence.
"""
import sqlite3
import json
import time
from pathlib import Path
from typing import Optional, List, Dict
import numpy as np


class CacheManager:
    """Manages embedding cache using SQLite database."""
    
    def __init__(self, cache_db: str = "embeddings_cache.db"):
        self.cache_db = Path(cache_db)
        self._init_database()
    
    def _init_database(self):
        """Initialize SQLite database with embeddings table."""
        conn = sqlite3.connect(self.cache_db)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS embeddings (
                doc_id TEXT PRIMARY KEY,
                embedding TEXT NOT NULL,
                hash TEXT NOT NULL,
                updated_at REAL NOT NULL,
                filename TEXT,
                doc_length INTEGER
            )
        """)
        
        # Create index on hash for faster lookups
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_hash ON embeddings(hash)
        """)
        
        conn.commit()
        conn.close()
    
    def get_embedding(self, doc_id: str, doc_hash: str) -> Optional[np.ndarray]:
        """
        Retrieve cached embedding if document hash matches.
        
        Args:
            doc_id: Document identifier
            doc_hash: SHA256 hash of document text
            
        Returns:
            Cached embedding as numpy array, or None if not found/mismatched
        """
        conn = sqlite3.connect(self.cache_db)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT embedding, hash FROM embeddings
            WHERE doc_id = ?
        """, (doc_id,))
        
        result = cursor.fetchone()
        conn.close()
        
        if result:
            cached_embedding_json, cached_hash = result
            # Check if hash matches (document hasn't changed)
            if cached_hash == doc_hash:
                embedding = np.array(json.loads(cached_embedding_json))
                return embedding
        
        return None
    
    def store_embedding(self, doc_id: str, embedding: np.ndarray, doc_hash: str,
                       filename: str = None, doc_length: int = None):
        """
        Store embedding in cache.
        
        Args:
            doc_id: Document identifier
            embedding: Embedding vector as numpy array
            doc_hash: SHA256 hash of document text
            filename: Optional filename
            doc_length: Optional document length
        """
        conn = sqlite3.connect(self.cache_db)
        cursor = conn.cursor()
        
        embedding_json = json.dumps(embedding.tolist())
        timestamp = time.time()
        
        cursor.execute("""
            INSERT OR REPLACE INTO embeddings
            (doc_id, embedding, hash, updated_at, filename, doc_length)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (doc_id, embedding_json, doc_hash, timestamp, filename, doc_length))
        
        conn.commit()
        conn.close()
    
    def get_all_cached_embeddings(self) -> Dict[str, np.ndarray]:
        """
        Retrieve all cached embeddings.
        
        Returns:
            Dictionary mapping doc_id to embedding array
        """
        conn = sqlite3.connect(self.cache_db)
        cursor = conn.cursor()
        
        cursor.execute("SELECT doc_id, embedding FROM embeddings")
        results = cursor.fetchall()
        conn.close()
        
        embeddings = {}
        for doc_id, embedding_json in results:
            embeddings[doc_id] = np.array(json.loads(embedding_json))
        
        return embeddings
    
    def get_cache_stats(self) -> Dict:
        """
        Get statistics about the cache.
        
        Returns:
            Dictionary with cache statistics
        """
        conn = sqlite3.connect(self.cache_db)
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM embeddings")
        count = cursor.fetchone()[0]
        
        cursor.execute("SELECT MIN(updated_at), MAX(updated_at) FROM embeddings")
        min_time, max_time = cursor.fetchone()
        
        conn.close()
        
        return {
            "total_cached": count,
            "oldest_entry": min_time,
            "newest_entry": max_time
        }
    
    def clear_cache(self):
        """Clear all cached embeddings."""
        conn = sqlite3.connect(self.cache_db)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM embeddings")
        conn.commit()
        conn.close()
        print("Cache cleared")

