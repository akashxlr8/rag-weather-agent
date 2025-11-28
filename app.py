"""
Streamlit Chat UI for the RAG Weather Agent.
Run with: streamlit run app.py
"""

import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage

from agents.rag_agent import build_rag_agent
from integrations.langsmith import configure_tracing

# Load environment variables
load_dotenv()

# Configure tracing
configure_tracing()

st.set_page_config(page_title="RAG Weather Agent", page_icon="🌤️")

# -----------------------------------------------------------------------------
# Setup
# -----------------------------------------------------------------------------

@st.cache_resource
def get_agent():
    """Build and cache the RAG agent."""
    return build_rag_agent()


SUGGESTIONS = [
    "What is the weather in New York?",
    "Who is Akash Kumar Shaw?",
    "What is the weather in London?",
]

# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------

def get_response(agent, user_message: str, chat_history: list) -> str:
    """
    Send a message to the agent and get the final response.
    Includes chat history for context.
    """
    # Build messages from history + new message
    messages = []
    for msg in chat_history:
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        else:
            messages.append(AIMessage(content=msg["content"]))
    messages.append(HumanMessage(content=user_message))

    inputs = {"messages": messages}

    # Run the agent (no streaming)
    response_content = ""
    for event in agent.stream(inputs):
        for key, value in event.items():
            if key == "chatbot":
                last_msg = value["messages"][-1]
                if last_msg.content:
                    response_content = last_msg.content

    return response_content


def clear_conversation():
    """Clear the chat history."""
    st.session_state.messages = []

# -----------------------------------------------------------------------------
# UI
# -----------------------------------------------------------------------------

st.title("🌤️ RAG Weather Agent")
st.caption("Ask about the weather or query the knowledge base.")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar
with st.sidebar:
    st.header("Options")
    if st.button("🗑️ Clear Chat", use_container_width=True):
        clear_conversation()
        st.rerun()

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle pending message from suggestion buttons
if "pending_message" in st.session_state:
    user_message = st.session_state.pending_message
    del st.session_state.pending_message
else:
    user_message = st.chat_input("Ask a question...")

# Process user message
if user_message:
    # Display user message
    with st.chat_message("user"):
        st.markdown(user_message)

    # Add to history
    st.session_state.messages.append({"role": "user", "content": user_message})

    # Get agent response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            agent = get_agent()
            response = get_response(agent, user_message, st.session_state.messages[:-1])

        st.markdown(response)

    # Add response to history
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.rerun()

# Show suggestions below chat input
st.caption("Try asking:")
cols = st.columns(len(SUGGESTIONS))
for i, suggestion in enumerate(SUGGESTIONS):
    if cols[i].button(suggestion, key=f"sug_{i}", use_container_width=True):
        st.session_state.pending_message = suggestion
        st.rerun()
