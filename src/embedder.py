"""
Embedding Generator
Generates embeddings using sentence-transformers with caching support.
"""
import numpy as np
from typing import List, Optional
from sentence_transformers import SentenceTransformer
from .cache_manager import CacheManager
from .preprocessor import Document


class EmbeddingGenerator:
    """Generates and caches document embeddings."""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 cache_db: str = "embeddings_cache.db"):
        """
        Initialize embedding generator.
        
        Args:
            model_name: Name of the sentence-transformers model
            cache_db: Path to SQLite cache database
        """
        print(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.cache_manager = CacheManager(cache_db)
        print("Model loaded successfully")
    
    def embed_text(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text.
        
        Args:
            text: Input text
            
        Returns:
            Embedding vector as numpy array
        """
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding
    
    def embed_document(self, document: Document, use_cache: bool = True) -> np.ndarray:
        """
        Generate or retrieve cached embedding for a document.
        
        Args:
            document: Document object
            use_cache: Whether to use cache
            
        Returns:
            Embedding vector as numpy array
        """
        if use_cache:
            cached_embedding = self.cache_manager.get_embedding(
                document.doc_id, document.hash
            )
            if cached_embedding is not None:
                print(f"Using cached embedding for {document.doc_id}")
                return cached_embedding
        
        # Generate new embedding
        print(f"Generating embedding for {document.doc_id}")
        embedding = self.embed_text(document.text)
        
        # Store in cache
        if use_cache:
            self.cache_manager.store_embedding(
                doc_id=document.doc_id,
                embedding=embedding,
                doc_hash=document.hash,
                filename=document.filename,
                doc_length=document.length
            )
        
        return embedding
    
    def embed_documents(self, documents: List[Document], 
                       use_cache: bool = True,
                       batch_size: int = 32) -> np.ndarray:
        """
        Generate embeddings for multiple documents with caching.
        
        Args:
            documents: List of Document objects
            use_cache: Whether to use cache
            batch_size: Batch size for embedding generation
            
        Returns:
            Numpy array of embeddings (n_docs, embedding_dim)
        """
        embeddings = []
        docs_to_embed = []
        doc_indices = []
        
        # Check cache for each document
        for idx, doc in enumerate(documents):
            if use_cache:
                cached = self.cache_manager.get_embedding(doc.doc_id, doc.hash)
                if cached is not None:
                    embeddings.append((idx, cached))
                    continue
            
            docs_to_embed.append(doc)
            doc_indices.append(idx)
        
        # Generate embeddings for uncached documents
        if docs_to_embed:
            texts = [doc.text for doc in docs_to_embed]
            print(f"Generating embeddings for {len(texts)} documents...")
            new_embeddings = self.model.encode(
                texts, 
                batch_size=batch_size,
                convert_to_numpy=True,
                show_progress_bar=True
            )
            
            # Store in cache and add to results
            for doc, embedding in zip(docs_to_embed, new_embeddings):
                if use_cache:
                    self.cache_manager.store_embedding(
                        doc_id=doc.doc_id,
                        embedding=embedding,
                        doc_hash=doc.hash,
                        filename=doc.filename,
                        doc_length=doc.length
                    )
                embeddings.append((doc_indices[docs_to_embed.index(doc)], embedding))
        
        # Sort by original index and return as array
        embeddings.sort(key=lambda x: x[0])
        return np.array([emb for _, emb in embeddings])
    
    def embed_query(self, query: str) -> np.ndarray:
        """
        Generate embedding for a search query.
        
        Args:
            query: Search query text
            
        Returns:
            Embedding vector as numpy array
        """
        return self.embed_text(query)

