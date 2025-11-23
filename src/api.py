"""
FastAPI Application
REST API for the embedding search engine.
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import uvicorn

from .preprocessor import DocumentPreprocessor
from .embedder import EmbeddingGenerator
from .search_engine import SearchEngine


# Request/Response models
class SearchRequest(BaseModel):
    query: str
    top_k: int = 5


class SearchResult(BaseModel):
    doc_id: str
    filename: str
    score: float
    preview: str
    explanation: dict


class SearchResponse(BaseModel):
    results: List[SearchResult]
    query: str
    total_results: int


# Global instances (initialized on startup)
preprocessor = None
embedder = None
search_engine = None


app = FastAPI(
    title="Multi-document Embedding Search Engine",
    description="Embedding-based search API with caching",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    """Initialize components on startup."""
    global preprocessor, embedder, search_engine
    
    print("Initializing search engine...")
    
    # Initialize components
    preprocessor = DocumentPreprocessor(data_dir="data/docs")
    embedder = EmbeddingGenerator()
    
    # Load or prepare documents
    documents = preprocessor.load_all_documents()
    
    if not documents:
        print("No documents found. Preparing dataset from sklearn...")
        documents = preprocessor.prepare_dataset_from_sklearn()
    
    if not documents:
        raise RuntimeError("No documents available. Please add documents to data/docs/")
    
    # Generate embeddings (with caching)
    print("Generating embeddings...")
    embeddings = embedder.embed_documents(documents, use_cache=True)
    
    # Build search engine
    search_engine = SearchEngine(embedder, index_path="faiss_index.bin")
    search_engine.build_index(documents, embeddings)
    
    print(f"Search engine ready with {len(documents)} documents")


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Multi-document Embedding Search Engine API",
        "endpoints": {
            "/search": "POST - Search documents",
            "/health": "GET - Health check",
            "/stats": "GET - Cache and index statistics"
        }
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy", "documents_indexed": len(search_engine.documents) if search_engine else 0}


@app.get("/stats")
async def stats():
    """Get cache and index statistics."""
    if not embedder or not search_engine:
        raise HTTPException(status_code=503, detail="Search engine not initialized")
    
    cache_stats = embedder.cache_manager.get_cache_stats()
    
    return {
        "cache": cache_stats,
        "index": {
            "total_documents": len(search_engine.documents),
            "embedding_dimension": search_engine.dimension,
            "index_type": "FAISS IndexFlatIP"
        }
    }


@app.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest):
    """
    Search endpoint.
    
    Example request:
    {
        "query": "quantum physics basics",
        "top_k": 5
    }
    """
    if not search_engine:
        raise HTTPException(status_code=503, detail="Search engine not initialized")
    
    if not request.query or not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    if request.top_k < 1 or request.top_k > 100:
        raise HTTPException(status_code=400, detail="top_k must be between 1 and 100")
    
    try:
        results = search_engine.search(request.query, top_k=request.top_k)
        
        return SearchResponse(
            results=[SearchResult(**r) for r in results],
            query=request.query,
            total_results=len(results)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search error: {str(e)}")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

