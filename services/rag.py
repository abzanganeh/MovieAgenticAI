import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Path configuration
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INDEX_PATH = os.path.join(BASE_DIR, 'local_data', 'faiss_index')

def get_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def create_vector_db(documents):
    """
    Takes raw documents, chunks them, and creates the Index.
    """
    print(f"   [RAG] Processing {len(documents)} documents...")
    
    # 1. Chunking
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=250,
        chunk_overlap=20,
        length_function=len
    )
    chunks = text_splitter.split_documents(documents)
    print(f"   [RAG] Created {len(chunks)} chunks.")

    # 2. Indexing
    print("   [RAG] Building Vector Index...")
    embeddings = get_embeddings()
    vector_store = FAISS.from_documents(chunks, embeddings)
    vector_store.save_local(INDEX_PATH)
    print("💾 [RAG] Index saved successfully.")
    
    return vector_store

def get_vector_store():
    """
    Returns the vector store.
    AUTO-FIX: If index is missing, it runs ingestion automatically.
    """
    # 1. Check if DB exists
    if not os.path.exists(INDEX_PATH):
        print("⚠️ Vector Database not found. Initiating Auto-Ingestion...")
        
        # --- LAZY IMPORT (Prevents Circular Error) ---
        # We import ingest_movies ONLY when we actually need it
        from services.ingest import ingest_movies
        
        # Run the ingestion
        ingest_movies()
        
        print("✅ Auto-Ingestion Complete. Loading Database...")

    # 2. Load and return DB
    embeddings = get_embeddings()
    return FAISS.load_local(
        INDEX_PATH, 
        embeddings, 
        allow_dangerous_deserialization=True
    )

def get_rag_chain():
    """Returns the retriever for the Chat chain."""
    vector_store = get_vector_store()
    return vector_store.as_retriever(
        search_type="mmr", 
        search_kwargs={"k": 5, "fetch_k": 20}
    )

if __name__ == "__main__":
    # Test Block
    try:
        print("--- Testing RAG Auto-Heal ---")
        # This should trigger ingestion if you deleted the 'local_data/faiss_index' folder
        retriever = get_rag_chain()
        print("✅ RAG Chain loaded successfully!")
    except Exception as e:
        print(f"❌ Test Failed: {e}")