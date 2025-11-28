from typing import List
from loaders.pdf_loader import load_pdf, chunk_documents
from integrations.qdrant_client import create_collection, upsert_documents, get_retriever

COLLECTION_NAME = "rag_weather_cohere2"

def retrieve_documents(query: str) -> str:
    """
    Retrieves relevant documents for a given query using Qdrant.
    """
    retriever = get_retriever(COLLECTION_NAME)
    docs = retriever.invoke(query)
    
    # Concatenate document content
    return "\n\n".join([doc.page_content for doc in docs])

def index_pdf_documents(paths: List[str]):
    """
    Indexes PDF documents from the given paths.
    """
    all_docs = []
    for path in paths:
        print(f"Loading {path}...")
        raw_docs = load_pdf(path)
        chunks = chunk_documents(raw_docs)
        all_docs.extend(chunks)
    
    if not all_docs:
        print("No documents to index.")
        return

    print(f"Indexing {len(all_docs)} chunks into Qdrant collection '{COLLECTION_NAME}'...")
    # Cohere embed-english-v3.0 has 1024 dimensions
    create_collection(COLLECTION_NAME, vector_size=1024)
    upsert_documents(COLLECTION_NAME, all_docs)
    print("Indexing complete.")
