"""
Advanced RAG Retriever with Document Grading and Query Rewriting.

This module implements a self-contained LangGraph sub-graph that:
1. Retrieves documents from Qdrant
2. Grades them for relevance using an LLM
3. Rewrites the query if documents are not relevant (max 2 retries)
4. Returns the relevant context or an empty string if nothing found
"""

from typing import TypedDict, Literal
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from integrations.qdrant_client import get_retriever
from tools.prompts import GRADE_PROMPT, REWRITE_PROMPT
from dotenv import load_dotenv
load_dotenv()
# Collection name (must match the one used in retriever.py)
COLLECTION_NAME = "rag_weather_cohere2"

# Maximum number of query rewrite attempts
MAX_RETRIES = 2


# --- State Schema ---

# Default message when no relevant documents found
NO_RELEVANT_DOCS_MESSAGE = "I'm sorry, I couldn't find relevant information on that topic in the knowledge base."


class AdvancedRetrieverState(TypedDict):
    """State for the advanced retriever sub-graph."""
    query: str                # Current query (may be rewritten)
    original_query: str       # Original user query (preserved)
    context: str              # Retrieved document content
    retry_count: int          # Number of rewrite attempts
    is_relevant: bool         # Whether final documents were graded as relevant


# --- Pydantic Model for Structured Output ---

class GradeDocuments(BaseModel):
    """Grade documents using a binary score for relevance check."""
    binary_score: str = Field(
        description="Relevance score: 'yes' if relevant, or 'no' if not relevant"
    )


# --- LLM Setup ---

grader_llm = ChatOpenAI(model="gpt-4.1-nano", temperature=0)
rewriter_llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)


# --- Node Functions ---

def retrieve_node(state: AdvancedRetrieverState) -> dict:
    """Retrieve documents from Qdrant based on the current query."""
    query = state["query"]
    
    # Get retriever with score threshold
    retriever = get_retriever(COLLECTION_NAME, k=3, score_threshold=0.3)
    docs = retriever.invoke(query)
    
    # Concatenate document content
    context = "\n\n".join([doc.page_content for doc in docs])
    
    return {"context": context}


def rewrite_question_node(state: AdvancedRetrieverState) -> dict:
    """Rewrite the query to improve retrieval results."""
    query = state["query"]
    retry_count = state["retry_count"]
    
    prompt = REWRITE_PROMPT.format(question=query)
    response = rewriter_llm.invoke([{"role": "user", "content": prompt}])
    
    new_query = str(response.content).strip()
    
    return {
        "query": new_query,
        "retry_count": retry_count + 1
    }


def return_context_relevant_node(state: AdvancedRetrieverState) -> dict:
    """Return the final context when documents are relevant."""
    return {"is_relevant": True}


def return_context_irrelevant_node(state: AdvancedRetrieverState) -> dict:
    """Return default message when no relevant documents found after max retries."""
    return {
        "context": NO_RELEVANT_DOCS_MESSAGE,
        "is_relevant": False
    }


# --- Conditional Edge Function ---

def grade_documents(state: AdvancedRetrieverState) -> Literal["return_context_relevant", "return_context_irrelevant", "rewrite_question"]:
    """
    Grade retrieved documents for relevance.
    Routes to 'return_context_relevant' if documents are relevant.
    Routes to 'return_context_irrelevant' if max retries reached with irrelevant docs.
    Routes to 'rewrite_question' if not relevant and retries remaining.
    """
    query = state["original_query"]
    context = state["context"]
    retry_count = state["retry_count"]
    
    # If no context retrieved, check if we should retry
    if not context or context.strip() == "":
        if retry_count >= MAX_RETRIES:
            return "return_context_irrelevant"
        return "rewrite_question"
    
    # Grade the documents using LLM
    prompt = GRADE_PROMPT.format(question=query, context=context)
    
    response = grader_llm.with_structured_output(GradeDocuments).invoke(
        [{"role": "user", "content": prompt}]
    )
    
    score = response["binary_score"].lower()
    
    if score == "yes":
        return "return_context_relevant"
    else:
        # Not relevant - check if we can retry
        if retry_count >= MAX_RETRIES:
            # Max retries reached, return default message
            return "return_context_irrelevant"
        return "rewrite_question"


# --- Build the Sub-Graph ---

def build_advanced_retriever_graph():
    """Build and compile the advanced retriever sub-graph."""
    graph_builder = StateGraph(AdvancedRetrieverState)
    
    # Add nodes
    graph_builder.add_node("retrieve", retrieve_node)
    graph_builder.add_node("rewrite_question", rewrite_question_node)
    graph_builder.add_node("return_context_relevant", return_context_relevant_node)
    graph_builder.add_node("return_context_irrelevant", return_context_irrelevant_node)
    
    # Set entry point
    graph_builder.set_entry_point("retrieve")
    
    # Add edges
    graph_builder.add_conditional_edges(
        "retrieve",
        grade_documents,
        {
            "return_context_relevant": "return_context_relevant",
            "return_context_irrelevant": "return_context_irrelevant",
            "rewrite_question": "rewrite_question"
        }
    )
    
    # After rewriting, go back to retrieve
    graph_builder.add_edge("rewrite_question", "retrieve")
    
    # Terminal nodes
    graph_builder.add_edge("return_context_relevant", END)
    graph_builder.add_edge("return_context_irrelevant", END)
    
    return graph_builder.compile()


# --- Public API ---

# Compile the graph once at module load
_retriever_graph = None

def _get_graph():
    """Lazy initialization of the retriever graph."""
    global _retriever_graph
    if _retriever_graph is None:
        _retriever_graph = build_advanced_retriever_graph()
    return _retriever_graph


def advanced_retrieve(query: str) -> str:
    """
    Retrieve relevant documents with automatic grading and query rewriting.
    
    Args:
        query: The user's query to search for.
        
    Returns:
        Retrieved document content if relevant documents found,
        empty string if no relevant documents after max retries.
    """
    graph = _get_graph()
    
    initial_state: AdvancedRetrieverState = {
        "query": query,
        "original_query": query,
        "context": "",
        "retry_count": 0,
        "is_relevant": False
    }
    
    # Run the graph to completion
    final_state = graph.invoke(initial_state)
    
    return final_state.get("context", "")
