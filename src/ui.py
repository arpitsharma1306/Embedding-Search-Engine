"""
Streamlit UI
Simple web interface for the search engine.
"""
import streamlit as st
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocessor import DocumentPreprocessor
from src.embedder import EmbeddingGenerator
from src.search_engine import SearchEngine


@st.cache_resource
def initialize_search_engine():
    """Initialize and cache the search engine."""
    preprocessor = DocumentPreprocessor(data_dir="data/docs")
    embedder = EmbeddingGenerator()
    
    # Load documents
    documents = preprocessor.load_all_documents()
    
    if not documents:
        st.info("No documents found. Preparing dataset from sklearn...")
        documents = preprocessor.prepare_dataset_from_sklearn()
    
    if not documents:
        st.error("No documents available. Please add documents to data/docs/")
        return None, None
    
    # Generate embeddings
    with st.spinner("Generating embeddings (this may take a while on first run)..."):
        embeddings = embedder.embed_documents(documents, use_cache=True)
    
    # Build search engine
    search_engine = SearchEngine(embedder, index_path="faiss_index.bin")
    search_engine.build_index(documents, embeddings)
    
    return search_engine, embedder


def main():
    st.set_page_config(
        page_title="Document Search Engine",
        page_icon="🔍",
        layout="wide"
    )
    
    st.title("🔍 Multi-document Embedding Search Engine")
    st.markdown("Search through documents using semantic similarity")
    
    # Initialize search engine
    search_engine, embedder = initialize_search_engine()
    
    if search_engine is None:
        st.stop()
    
    # Sidebar with stats
    with st.sidebar:
        st.header("📊 Statistics")
        st.metric("Total Documents", len(search_engine.documents))
        st.metric("Embedding Dimension", search_engine.dimension)
        
        cache_stats = embedder.cache_manager.get_cache_stats()
        st.metric("Cached Embeddings", cache_stats["total_cached"])
        
        st.header("ℹ️ About")
        st.markdown("""
        This search engine uses:
        - **sentence-transformers** for embeddings
        - **FAISS** for vector search
        - **SQLite** for caching
        
        Results are ranked by semantic similarity with keyword overlap analysis.
        """)
    
    # Search interface
    query = st.text_input(
        "Enter your search query:",
        placeholder="e.g., quantum physics basics",
        key="search_query"
    )
    
    top_k = st.slider("Number of results:", min_value=1, max_value=20, value=5)
    
    if st.button("Search", type="primary") or query:
        if query:
            with st.spinner("Searching..."):
                results = search_engine.search(query, top_k=top_k)
            
            if results:
                st.success(f"Found {len(results)} results")
                
                for i, result in enumerate(results, 1):
                    with st.expander(
                        f"📄 {result['filename']} (Score: {result['score']:.3f})",
                        expanded=(i == 1)
                    ):
                        st.markdown(f"**Document ID:** {result['doc_id']}")
                        st.markdown(f"**Similarity Score:** {result['score']:.4f}")
                        
                        # Explanation
                        exp = result['explanation']
                        st.markdown("### Explanation")
                        st.info(exp['reason'])
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Keyword Overlap", exp['keyword_overlap_count'])
                        with col2:
                            st.metric("Overlap Ratio", f"{exp['overlap_ratio']:.2%}")
                        with col3:
                            st.metric("Doc Length", exp['document_length'])
                        
                        if exp['overlapping_keywords']:
                            st.markdown("**Overlapping Keywords:**")
                            st.write(", ".join(exp['overlapping_keywords']))
                        
                        st.markdown("### Preview")
                        st.text(result['preview'])
            else:
                st.warning("No results found. Try a different query.")
        else:
            st.info("Enter a search query to begin")


if __name__ == "__main__":
    main()

