"""
Vector Search Engine
Implements FAISS-based vector search with ranking and explanations.
"""
import numpy as np
import faiss
from typing import List, Dict, Tuple, Optional
from collections import Counter
from pathlib import Path
import re

from .embedder import EmbeddingGenerator
from .preprocessor import Document
from .cache_manager import CacheManager


class SearchEngine:
    """Vector search engine with FAISS index and ranking explanations."""
    
    def __init__(self, embedder: EmbeddingGenerator, 
                 index_path: Optional[str] = None):
        """
        Initialize search engine.
        
        Args:
            embedder: EmbeddingGenerator instance
            index_path: Optional path to save/load FAISS index
        """
        self.embedder = embedder
        self.index_path = index_path
        self.index = None
        self.documents = []
        self.embeddings = None
        self.dimension = None
    
    def build_index(self, documents: List[Document], 
                   embeddings: np.ndarray,
                   normalize: bool = True):
        """
        Build FAISS index from documents and embeddings.
        
        Args:
            documents: List of Document objects
            embeddings: Embedding vectors (n_docs, embedding_dim)
            normalize: Whether to normalize embeddings (for cosine similarity)
        """
        self.documents = documents
        self.embeddings = embeddings
        self.dimension = embeddings.shape[1]
        
        # Normalize embeddings for cosine similarity (inner product)
        if normalize:
            faiss.normalize_L2(embeddings)
        
        # Create FAISS index (IndexFlatIP for inner product = cosine similarity)
        self.index = faiss.IndexFlatIP(self.dimension)
        self.index.add(embeddings.astype('float32'))
        
        print(f"Built FAISS index with {len(documents)} documents")
        
        # Save index if path provided
        if self.index_path:
            self.save_index()
    
    def load_index(self, documents: List[Document]):
        """
        Load FAISS index from file.
        
        Args:
            documents: List of Document objects (must match index)
        """
        if self.index_path and Path(self.index_path).exists():
            self.index = faiss.read_index(self.index_path)
            self.documents = documents
            self.dimension = self.index.d
            print(f"Loaded FAISS index with {self.index.ntotal} vectors")
        else:
            print("Index file not found, need to rebuild")
    
    def save_index(self):
        """Save FAISS index to file."""
        if self.index and self.index_path:
            faiss.write_index(self.index, self.index_path)
            print(f"Saved index to {self.index_path}")
    
    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Search for similar documents.
        
        Args:
            query: Search query text
            top_k: Number of top results to return
            
        Returns:
            List of result dictionaries with scores and explanations
        """
        if self.index is None:
            raise ValueError("Index not built. Call build_index() first.")
        
        # Generate query embedding
        query_embedding = self.embedder.embed_query(query)
        query_embedding = query_embedding.reshape(1, -1).astype('float32')
        
        # Normalize query embedding
        faiss.normalize_L2(query_embedding)
        
        # Search
        scores, indices = self.index.search(query_embedding, top_k)
        
        # Build results with explanations
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:  # FAISS returns -1 for invalid indices
                continue
            
            doc = self.documents[idx]
            explanation = self._generate_explanation(query, doc, score)
            
            results.append({
                "doc_id": doc.doc_id,
                "filename": doc.filename,
                "score": float(score),
                "preview": doc.text[:200] + "..." if len(doc.text) > 200 else doc.text,
                "explanation": explanation
            })
        
        return results
    
    def _generate_explanation(self, query: str, document: Document, 
                             score: float) -> Dict:
        """
        Generate ranking explanation for a matched document.
        
        Args:
            query: Search query
            document: Matched document
            score: Similarity score
            
        Returns:
            Explanation dictionary
        """
        # Extract keywords from query
        query_words = set(re.findall(r'\b\w+\b', query.lower()))
        doc_words = set(re.findall(r'\b\w+\b', document.text.lower()))
        
        # Find overlapping keywords
        overlapping = query_words.intersection(doc_words)
        
        # Calculate overlap ratio
        overlap_ratio = len(overlapping) / len(query_words) if query_words else 0
        
        # Document length normalization (longer docs might have more matches)
        length_norm = min(1.0, 1000.0 / document.length) if document.length > 0 else 1.0
        
        # Normalized score (adjusted for document length)
        normalized_score = score * (1 + length_norm * 0.1)
        
        return {
            "similarity_score": float(score),
            "overlapping_keywords": list(overlapping)[:10],  # Top 10 keywords
            "keyword_overlap_count": len(overlapping),
            "overlap_ratio": round(overlap_ratio, 3),
            "document_length": document.length,
            "length_normalization": round(length_norm, 3),
            "normalized_score": round(normalized_score, 3),
            "reason": self._generate_reason(overlapping, overlap_ratio, score)
        }
    
    def _generate_reason(self, overlapping: set, overlap_ratio: float, 
                        score: float) -> str:
        """
        Generate human-readable reason for match.
        
        Args:
            overlapping: Set of overlapping keywords
            score: Similarity score
            
        Returns:
            Reason string
        """
        if overlap_ratio > 0.5:
            return f"Strong keyword match ({len(overlapping)} keywords) with high semantic similarity ({score:.2f})"
        elif overlap_ratio > 0.2:
            return f"Moderate keyword overlap ({len(overlapping)} keywords) with good semantic similarity ({score:.2f})"
        elif score > 0.7:
            return f"High semantic similarity ({score:.2f}) despite limited keyword overlap - likely related concepts"
        else:
            return f"Semantic similarity ({score:.2f}) with some keyword matches ({len(overlapping)} keywords)"

