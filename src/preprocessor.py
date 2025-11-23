"""
Document Preprocessor
Handles loading, cleaning, and metadata extraction from text documents.
"""
import os
import re
import hashlib
from pathlib import Path
from typing import List, Dict, Tuple
from dataclasses import dataclass


@dataclass
class Document:
    """Represents a processed document with metadata."""
    doc_id: str
    filename: str
    text: str
    length: int
    hash: str
    filepath: str


class DocumentPreprocessor:
    """Preprocesses text documents: cleaning, normalization, and metadata extraction."""
    
    def __init__(self, data_dir: str = "data/docs"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
    
    def clean_text(self, text: str) -> str:
        """
        Clean text: lowercase, remove HTML tags, normalize whitespace.
        
        Args:
            text: Raw text content
            
        Returns:
            Cleaned text
        """
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Strip leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def compute_hash(self, text: str) -> str:
        """
        Compute SHA256 hash of text for cache lookup.
        
        Args:
            text: Text content
            
        Returns:
            SHA256 hash string
        """
        return hashlib.sha256(text.encode('utf-8')).hexdigest()
    
    def load_document(self, filepath: Path) -> Document:
        """
        Load and preprocess a single document.
        
        Args:
            filepath: Path to the document file
            
        Returns:
            Document object with metadata
        """
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                raw_text = f.read()
            
            cleaned_text = self.clean_text(raw_text)
            doc_id = filepath.stem
            filename = filepath.name
            doc_hash = self.compute_hash(cleaned_text)
            
            return Document(
                doc_id=doc_id,
                filename=filename,
                text=cleaned_text,
                length=len(cleaned_text),
                hash=doc_hash,
                filepath=str(filepath)
            )
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            return None
    
    def load_all_documents(self) -> List[Document]:
        """
        Load and preprocess all .txt files from data directory.
        
        Returns:
            List of Document objects
        """
        documents = []
        txt_files = list(self.data_dir.glob("*.txt"))
        
        if not txt_files:
            print(f"No .txt files found in {self.data_dir}")
            return documents
        
        print(f"Found {len(txt_files)} text files")
        
        for filepath in txt_files:
            doc = self.load_document(filepath)
            if doc:
                documents.append(doc)
        
        print(f"Successfully loaded {len(documents)} documents")
        return documents
    
    def prepare_dataset_from_sklearn(self) -> List[Document]:
        """
        Download and prepare 20 Newsgroups dataset as text files.
        
        Returns:
            List of Document objects
        """
        try:
            from sklearn.datasets import fetch_20newsgroups
            
            print("Downloading 20 Newsgroups dataset...")
            dataset = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'))
            
            print(f"Dataset loaded: {len(dataset.data)} documents")
            
            documents = []
            for idx, text in enumerate(dataset.data):
                # Skip empty documents
                if not text or len(text.strip()) < 50:
                    continue
                
                cleaned_text = self.clean_text(text)
                doc_id = f"doc_{idx:03d}"
                filename = f"{doc_id}.txt"
                doc_hash = self.compute_hash(cleaned_text)
                
                # Save to file
                filepath = self.data_dir / filename
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(cleaned_text)
                
                doc = Document(
                    doc_id=doc_id,
                    filename=filename,
                    text=cleaned_text,
                    length=len(cleaned_text),
                    hash=doc_hash,
                    filepath=str(filepath)
                )
                documents.append(doc)
            
            print(f"Saved {len(documents)} documents to {self.data_dir}")
            return documents
            
        except ImportError:
            print("scikit-learn not installed. Install with: pip install scikit-learn")
            return []
        except Exception as e:
            print(f"Error preparing dataset: {e}")
            return []

