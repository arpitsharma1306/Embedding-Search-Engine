"""
Main entry point for the search engine.
Can be used to initialize and test the system.
"""
import argparse
from src.preprocessor import DocumentPreprocessor
from src.embedder import EmbeddingGenerator
from src.search_engine import SearchEngine


def main():
    parser = argparse.ArgumentParser(description="Multi-document Embedding Search Engine")
    parser.add_argument("--prepare-data", action="store_true", 
                       help="Download and prepare dataset")
    parser.add_argument("--rebuild-index", action="store_true",
                       help="Rebuild the search index")
    parser.add_argument("--query", type=str, help="Test query")
    parser.add_argument("--top-k", type=int, default=5, help="Number of results")
    
    args = parser.parse_args()
    
    # Initialize components
    preprocessor = DocumentPreprocessor(data_dir="data/docs")
    embedder = EmbeddingGenerator()
    
    # Prepare data if requested
    if args.prepare_data:
        print("Preparing dataset...")
        documents = preprocessor.prepare_dataset_from_sklearn()
        print(f"Prepared {len(documents)} documents")
        return
    
    # Load documents
    documents = preprocessor.load_all_documents()
    
    if not documents:
        print("No documents found. Run with --prepare-data first.")
        return
    
    # Generate embeddings
    print("Generating embeddings...")
    embeddings = embedder.embed_documents(documents, use_cache=True)
    
    # Build search engine
    search_engine = SearchEngine(embedder, index_path="faiss_index.bin")
    
    if args.rebuild_index or not Path("faiss_index.bin").exists():
        print("Building search index...")
        search_engine.build_index(documents, embeddings)
    else:
        print("Loading existing index...")
        search_engine.load_index(documents)
    
    # Test query if provided
    if args.query:
        print(f"\nSearching for: '{args.query}'")
        results = search_engine.search(args.query, top_k=args.top_k)
        
        print(f"\nFound {len(results)} results:\n")
        for i, result in enumerate(results, 1):
            print(f"{i}. {result['filename']} (Score: {result['score']:.4f})")
            print(f"   Preview: {result['preview'][:100]}...")
            print(f"   Explanation: {result['explanation']['reason']}")
            print()
    else:
        print("\nSearch engine ready!")
        print("Use --query 'your query' to test search")
        print("Or start the API with: uvicorn src.api:app --reload")
        print("Or start the UI with: streamlit run src/ui.py")


if __name__ == "__main__":
    from pathlib import Path
    main()

