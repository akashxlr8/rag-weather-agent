import os
from typing import List
from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http import models
from integrations.embeddings import get_embeddings

def get_qdrant_client():
    url = os.getenv("QDRANT_URL")
    api_key = os.getenv("QDRANT_API_KEY")
    
    if not url:
        # Fallback to local memory for testing if no URL provided
        return QdrantClient(location=":memory:")
    
    return QdrantClient(url=url, api_key=api_key)

def create_collection(collection_name: str, vector_size: int = 1536):
    """
    Creates a Qdrant collection if it doesn't exist.
    """
    client = get_qdrant_client()
    try:
        client.get_collection(collection_name)
        print(f"Collection '{collection_name}' already exists.")
    except Exception:
        print(f"Creating collection '{collection_name}'...")
        client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE),
        )

def upsert_documents(collection_name: str, docs: List[Document]):
    """
    Upserts documents into the Qdrant collection.
    """
    client = get_qdrant_client()
    embeddings = get_embeddings()
    
    vector_store = QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=embeddings,
    )
    vector_store.add_documents(documents=docs)

def get_retriever(collection_name: str, k: int = 3, score_threshold: float = 0.5):
    """
    Returns a LangChain retriever for the Qdrant collection.
    """
    client = get_qdrant_client()
    embeddings = get_embeddings()
    
    vector_store = QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=embeddings,
    )
    return vector_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": k, "score_threshold": score_threshold}
    )
