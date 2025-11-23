# Multi-document Embedding Search Engine with Caching

A lightweight embedding-based search engine over 100-200 text documents with efficient embedding generation, local caching, vector search using FAISS, and a clean retrieval API.

## Features

- ✅ **Efficient Embedding Generation**: Uses `sentence-transformers/all-MiniLM-L6-v2` for fast, high-quality embeddings
- ✅ **Local Caching**: SQLite-based caching system that avoids recomputing embeddings for unchanged documents
- ✅ **Vector Search**: FAISS-based vector search with cosine similarity ranking
- ✅ **Clean Retrieval API**: FastAPI-based REST API with `/search` endpoint
- ✅ **Result Ranking & Explanation**: Detailed explanations for each result including keyword overlap, similarity scores, and reasoning
- ✅ **Streamlit UI**: Bonus web interface for interactive searching
- ✅ **Persistent FAISS Index**: Saves and loads FAISS index for faster startup

## Project Structure

```
AI_Assignment/
├── src/
│   ├── __init__.py
│   ├── preprocessor.py      # Document loading and text cleaning
│   ├── cache_manager.py     # SQLite-based embedding cache
│   ├── embedder.py          # Embedding generation with caching
│   ├── search_engine.py     # FAISS vector search with explanations
│   ├── api.py               # FastAPI REST API
│   └── ui.py                # Streamlit web UI (bonus)
├── data/
│   └── docs/                # Text documents (gitignored)
├── main.py                  # CLI entry point
├── requirements.txt         # Python dependencies
├── .gitignore              # Git ignore rules
├── embeddings_cache.db      # SQLite cache database (gitignored)
└── faiss_index.bin         # FAISS index file (gitignored)
```

## Installation

1. **Clone the repository** (or navigate to the project directory)

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

## How Caching Works

The caching system uses SQLite to store document embeddings with the following structure:

```python
{
    "doc_id": "doc_001",
    "embedding": [...],           # JSON-encoded numpy array
    "hash": "sha256_of_text",     # SHA256 hash of document text
    "updated_at": "timestamp",    # Unix timestamp
    "filename": "doc_001.txt",
    "doc_length": 1234
}
```

### Cache Lookup Process

1. When generating embeddings, the system first computes a SHA256 hash of the cleaned document text
2. It checks the cache for an existing embedding with matching `doc_id` and `hash`
3. If found and hash matches → **reuse cached embedding** (no recomputation)
4. If not found or hash changed → **generate new embedding** and store in cache

### Benefits

- **No recomputation**: Unchanged documents reuse cached embeddings
- **Automatic invalidation**: Changed documents (different hash) automatically get new embeddings
- **Persistent storage**: Cache survives application restarts
- **Fast lookups**: SQLite indexed by hash for O(log n) lookups

## Usage

### 1. Prepare Dataset

The system can automatically download and prepare the 20 Newsgroups dataset:

```bash
python main.py --prepare-data
```

This will:
- Download the 20 Newsgroups dataset from scikit-learn
- Clean and preprocess all documents
- Save them as `.txt` files in `data/docs/`

Alternatively, you can manually add your own `.txt` files to `data/docs/`.

### 2. Generate Embeddings and Build Index

Embeddings are generated automatically when you run the API or use the CLI. The first run will:

1. Load all documents from `data/docs/`
2. Check cache for existing embeddings
3. Generate embeddings only for new/changed documents
4. Build FAISS index
5. Save index to `faiss_index.bin`

```bash
# Rebuild index from scratch
python main.py --rebuild-index
```

### 3. Start the API Server

```bash
uvicorn src.api:app --reload
```

The API will be available at `http://localhost:8000`

#### API Endpoints

- **GET `/`**: API information and available endpoints
- **GET `/health`**: Health check
- **GET `/stats`**: Cache and index statistics
- **POST `/search`**: Search documents

#### Search Example

```bash
curl -X POST "http://localhost:8000/search" \
     -H "Content-Type: application/json" \
     -d '{"query": "quantum physics basics", "top_k": 5}'
```

Response:
```json
{
  "results": [
    {
      "doc_id": "doc_014",
      "filename": "doc_014.txt",
      "score": 0.88,
      "preview": "Quantum theory is concerned with...",
      "explanation": {
        "similarity_score": 0.88,
        "overlapping_keywords": ["quantum", "physics"],
        "keyword_overlap_count": 2,
        "overlap_ratio": 0.667,
        "document_length": 1234,
        "length_normalization": 0.81,
        "normalized_score": 0.95,
        "reason": "Strong keyword match (2 keywords) with high semantic similarity (0.88)"
      }
    }
  ],
  "query": "quantum physics basics",
  "total_results": 5
}
```

### 4. Use the Streamlit UI (Bonus)

```bash
streamlit run src/ui.py
```

This opens a web interface at `http://localhost:8501` where you can:
- Search documents interactively
- View detailed explanations for each result
- See cache statistics
- Explore keyword overlaps

### 5. CLI Testing

Test search from command line:

```bash
python main.py --query "machine learning algorithms" --top-k 5
```

## Design Choices

### 1. **Embedding Model: `sentence-transformers/all-MiniLM-L6-v2`**
   - **Why**: Fast, lightweight (80MB), good quality for semantic search
   - **Alternative**: Could use larger models (e.g., `all-mpnet-base-v2`) for better quality at cost of speed

### 2. **Caching: SQLite**
   - **Why**: 
     - Lightweight, no external dependencies
     - Persistent across restarts
     - Supports indexed lookups
     - Easy to inspect/debug
   - **Alternatives considered**: JSON (slower for large datasets), Pickle (less portable), Redis (overkill for local)

### 3. **Vector Search: FAISS IndexFlatIP**
   - **Why**: 
     - Fast exact search (inner product = cosine similarity with normalized vectors)
     - Simple, no approximation needed for 100-200 documents
     - Persistent index support
   - **Alternative**: Custom NumPy cosine similarity (slower but no FAISS dependency)

### 4. **API Framework: FastAPI**
   - **Why**: 
     - Modern, fast, async support
     - Automatic OpenAPI documentation
     - Type validation with Pydantic
   - **Alternative**: Flask (simpler but less features)

### 5. **Text Preprocessing**
   - Lowercase normalization
   - HTML tag removal
   - Whitespace normalization
   - **Why**: Consistent embedding generation, handles various document formats

### 6. **Ranking Explanation**
   - Keyword overlap analysis
   - Overlap ratio calculation
   - Document length normalization
   - Human-readable reasoning
   - **Why**: Helps users understand why documents matched, improves transparency

## Module Breakdown

### `preprocessor.py`
- `DocumentPreprocessor`: Loads, cleans, and extracts metadata from documents
- `Document`: Dataclass representing a processed document
- Methods: `clean_text()`, `compute_hash()`, `load_document()`, `load_all_documents()`

### `cache_manager.py`
- `CacheManager`: SQLite-based embedding cache
- Methods: `get_embedding()`, `store_embedding()`, `get_cache_stats()`, `clear_cache()`

### `embedder.py`
- `EmbeddingGenerator`: Generates embeddings with caching support
- Methods: `embed_text()`, `embed_document()`, `embed_documents()`, `embed_query()`

### `search_engine.py`
- `SearchEngine`: FAISS-based vector search with explanations
- Methods: `build_index()`, `search()`, `_generate_explanation()`, `_generate_reason()`

### `api.py`
- FastAPI application with `/search` endpoint
- Automatic initialization on startup
- Request/response models with Pydantic

### `ui.py`
- Streamlit web interface
- Interactive search with visualizations
- Cache statistics display

## Performance Considerations

- **First Run**: Downloads model (~80MB), generates all embeddings (may take 5-10 minutes for 200 docs)
- **Subsequent Runs**: Loads cached embeddings, builds index in seconds
- **Search Speed**: <100ms for queries on 200 documents
- **Memory**: ~500MB for model + embeddings + index

## Future Enhancements

- [ ] Query expansion using WordNet or embedding similarity
- [ ] Batch embedding with multiprocessing for faster initial generation
- [ ] Evaluation metrics (precision@k, recall@k) with test queries
- [ ] Support for other embedding models (OpenAI, Cohere, etc.)
- [ ] Advanced FAISS indices (IVF, HNSW) for larger document collections
- [ ] Document update detection and incremental indexing

## License

This project is created for the AI Engineer Intern Assignment at CodeAtRandom AI.

## Author

Created as part of the AI Engineer Intern assignment.

