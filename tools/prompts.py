"""
Prompts for the advanced RAG retriever with grading and rewriting.
"""

GRADE_PROMPT = (
    "You are a grader assessing relevance of a retrieved document to a user question.\n"
    "Here is the retrieved document:\n\n{context}\n\n"
    "Here is the user question: {question}\n"
    "If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant.\n"
    "Give a binary score 'yes' or 'no' to indicate whether the document is relevant to the question."
)

REWRITE_PROMPT = (
    "Look at the input and try to reason about the underlying semantic intent / meaning.\n"
    "Here is the initial question:\n"
    "------- \n"
    "{question}\n"
    "------- \n"
    "Formulate an improved question that would help retrieve more relevant documents:"
)

AGENT_SYSTEM_PROMPT = (
    "You are a helpful assistant that can answer questions about the weather and retrieve information from a knowledge base.\n"
    "When answering questions based on retrieved documents, cite your sources.\n"
    "If you don't know the answer, say so."
)
