import os

def configure_tracing():
    """
    Configures environment variables for LangSmith tracing.
    """
    # Enable tracing
    os.environ["LANGSMITH_TRACING"] = "true"
    
    # Set project name if not already set
    if not os.getenv("LANGCHAIN_PROJECT"):
        os.environ["LANGCHAIN_PROJECT"] = "rag-weather-agent"
    
    # Ensure API key is available (LangSmith uses LANGSMITH_API_KEY)
    # If LANGCHAIN_API_KEY is set but LANGSMITH_API_KEY is not, copy it over
    langchain_key = os.getenv("LANGCHAIN_API_KEY")
    if langchain_key and not os.getenv("LANGSMITH_API_KEY"):
        os.environ["LANGSMITH_API_KEY"] = langchain_key
        
    # Conversely, if LANGSMITH_API_KEY is set but LANGCHAIN_API_KEY is not, copy it over
    # (LangChain libraries often look for LANGCHAIN_API_KEY)
    langsmith_key = os.getenv("LANGSMITH_API_KEY")
    if langsmith_key and not os.getenv("LANGCHAIN_API_KEY"):
        os.environ["LANGCHAIN_API_KEY"] = langsmith_key
