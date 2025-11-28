import os
from langchain_cohere import CohereEmbeddings

def get_embeddings():
    """
    Returns the configured embeddings model.
    Uses CohereEmbeddings (embed-english-v3.0).
    """
    api_key = os.getenv("COHERE_API_KEY")
    if not api_key:
        # We allow missing key for import time, but it will fail at runtime if used.
        pass
        
    return CohereEmbeddings(
        cohere_api_key=api_key,
        model="embed-english-v3.0"
    )