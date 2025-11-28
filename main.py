# Build a custom RAG agent with LangGraph

import getpass
import os
from dotenv import load_dotenv

load_dotenv()

def _set_env(key: str):
    if key not in os.environ:
        os.environ[key] = getpass.getpass(f"{key}:")

_set_env("OPENAI_API_KEY")
_set_env("OPENWEATHER_API_KEY")

# Define weather tool
from langchain.tools import tool
import requests

@tool
def get_weather(city: str) -> str:
    """Get the current weather for a given city."""
    API_KEY = os.getenv("OPENWEATHER_API_KEY")
    if not API_KEY:
        return (
            "OPENWEATHER_API_KEY is not set. Please set it in your environment "
            "or in a .env file (OPENWEATHER_API_KEY=your_key)."
        )

    try:
        geo_url = (
            f"http://api.openweathermap.org/geo/1.0/direct?q={city}&limit=1&appid={API_KEY}"
        )
        geo_resp = requests.get(geo_url, timeout=10)
    except Exception as e:
        return f"Network error while calling geocoding API: {e}"

    if geo_resp.status_code != 200:
        # Try to show useful error message returned by API
        try:
            body = geo_resp.json()
        except Exception:
            body = geo_resp.text
        return f"Geocoding API error {geo_resp.status_code}: {body}"

    try:
        geo = geo_resp.json()
    except Exception as e:
        return f"Invalid JSON from geocoding API: {e}"

    if not geo:
        return f"Could not find location for city: {city}"

    if isinstance(geo, dict):
        return f"Error in geocoding: {geo.get('message', 'Unknown error')}"

    lat = geo[0].get("lat")
    lon = geo[0].get("lon")
    if lat is None or lon is None:
        return f"Geocoding response missing coordinates: {geo[0]}"

    try:
        weather_url = (
            f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={API_KEY}&units=metric"
        )
        weather_resp = requests.get(weather_url, timeout=10)
    except Exception as e:
        return f"Network error while calling weather API: {e}"

    if weather_resp.status_code != 200:
        try:
            body = weather_resp.json()
        except Exception:
            body = weather_resp.text
        return f"Weather API error {weather_resp.status_code}: {body}"

    try:
        weather = weather_resp.json()
    except Exception as e:
        return f"Invalid JSON from weather API: {e}"

    # Safely extract fields
    try:
        description = weather["weather"][0]["description"]
        temperature = weather["main"]["temp"]
        humidity = weather["main"]["humidity"]
        wind_speed = weather.get("wind", {}).get("speed", "N/A")
    except (KeyError, TypeError) as e:
        return f"Unexpected weather data format: {e} -- {weather}"

    weather_report = (
        f"Current weather in {city}:\n"
        f"Description: {description}\n"
        f"Temperature: {temperature}°C\n"
        f"Humidity: {humidity}%\n"
        f"Wind Speed: {wind_speed} m/s"
    )
    return weather_report

# 1. Preprocess documents

from langchain_community.document_loaders import WebBaseLoader

urls = [
    "https://openweathermap.org/api",
    "https://openweathermap.org/current",
    "https://openweathermap.org/weather-conditions",
]

docs = [WebBaseLoader(url).load() for url in urls]

from langchain_text_splitters import RecursiveCharacterTextSplitter

docs_list = [item for sublist in docs for item in sublist]

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=100, chunk_overlap=50
)
doc_splits = text_splitter.split_documents(docs_list)

# 2. Create a retriever tool

from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings

vectorstore = InMemoryVectorStore.from_documents(
    documents=doc_splits, embedding=OpenAIEmbeddings()
)
retriever = vectorstore.as_retriever()

@tool
def retrieve_weather_docs(query: str) -> str:
    """Search and return information about OpenWeatherMap API and weather conditions."""
    docs = retriever.invoke(query)
    return "\n\n".join([doc.page_content for doc in docs])

# 3. Generate query

from langgraph.graph import MessagesState
from langchain.chat_models import init_chat_model

response_model = init_chat_model("gpt-4o", temperature=0)

def generate_query_or_respond(state: MessagesState):
    """Call the model to generate a response based on the current state. Given
    the question, it will decide to retrieve using the retriever tool, get weather, or simply respond to the user.
    """
    response = (
        response_model
        .bind_tools([retrieve_weather_docs, get_weather]).invoke(state["messages"])
    )
    return {"messages": [response]}

# 4. Grade documents

from pydantic import BaseModel, Field
from typing import Literal

GRADE_PROMPT = (
    "You are a grader assessing relevance of a retrieved document to a user question. \n "
    "Here is the retrieved document: \n\n {context} \n\n"
    "Here is the user question: {question} \n"
    "If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n"
    "Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."
)

class GradeDocuments(BaseModel):
    """Grade documents using a binary score for relevance check."""

    binary_score: str = Field(
        description="Relevance score: 'yes' if relevant, or 'no' if not relevant"
    )

grader_model = init_chat_model("gpt-4o", temperature=0)

def grade_documents(
    state: MessagesState,
) -> Literal["generate_answer", "rewrite_question"]:
    """Determine whether the retrieved documents are relevant to the question."""
    question = state["messages"][0].content
    context = state["messages"][-1].content

    prompt = GRADE_PROMPT.format(question=question, context=context)
    response = (
        grader_model
        .with_structured_output(GradeDocuments).invoke(
            [{"role": "user", "content": prompt}]
        )
    )
    score = response.binary_score

    if score == "yes":
        return "generate_answer"
    else:
        return "rewrite_question"

# 5. Rewrite question

from langchain.messages import HumanMessage

REWRITE_PROMPT = (
    "Look at the input and try to reason about the underlying semantic intent / meaning.\n"
    "Here is the initial question:"
    "\n ------- \n"
    "{question}"
    "\n ------- \n"
    "Formulate an improved question:"
)

def rewrite_question(state: MessagesState):
    """Rewrite the original user question."""
    messages = state["messages"]
    question = messages[0].content
    prompt = REWRITE_PROMPT.format(question=question)
    response = response_model.invoke([{"role": "user", "content": prompt}])
    return {"messages": [HumanMessage(content=response.content)]}

# 6. Generate an answer

GENERATE_PROMPT = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer the question. "
    "If you don't know the answer, just say that you don't know. "
    "Use three sentences maximum and keep the answer concise.\n"
    "Question: {question} \n"
    "Context: {context}"
)

def generate_answer(state: MessagesState):
    """Generate an answer."""
    question = state["messages"][0].content
    context = state["messages"][-1].content
    prompt = GENERATE_PROMPT.format(question=question, context=context)
    response = response_model.invoke([{"role": "user", "content": prompt}])
    return {"messages": [response]}

# 7. Assemble the graph

from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition

workflow = StateGraph(MessagesState)

# Define the nodes we will cycle between
workflow.add_node(generate_query_or_respond)
workflow.add_node("retrieve", ToolNode([retrieve_weather_docs, get_weather]))
workflow.add_node(rewrite_question)
workflow.add_node(generate_answer)

workflow.add_edge(START, "generate_query_or_respond")

# Decide whether to retrieve
workflow.add_conditional_edges(
    "generate_query_or_respond",
    # Assess LLM decision (call `retriever_tool` tool or respond to the user)
    tools_condition,
    {
        # Translate the condition outputs to nodes in our graph
        "tools": "retrieve",
        END: END,
    },
)

# Edges taken after the `action` node is called.
workflow.add_conditional_edges(
    "retrieve",
    # Assess agent decision
    grade_documents,
)
workflow.add_edge("generate_answer", END)
workflow.add_edge("rewrite_question", "generate_query_or_respond")

# Compile
graph = workflow.compile()

# 8. Run the agentic RAG

if __name__ == "__main__":
    for chunk in graph.stream(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "What is the weather in Paris? Also, what does OpenWeatherMap API provide?",
                }
            ]
        }
    ):
        for node, update in chunk.items():
            print("Update from node", node)
            update["messages"][-1].pretty_print()
            print("\n\n")
