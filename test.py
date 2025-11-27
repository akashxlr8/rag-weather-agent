# Step 1: Define tools and model

from langchain.tools import tool
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
import requests
load_dotenv()
import os

model = init_chat_model(
    "gpt-4.1-mini",
    temperature=0
)


# Define tools
@tool
def multiply(a: int, b: int) -> int:
    """Multiply `a` and `b`.

    Args:
        a: First int
        b: Second int
    """
    return a * b


@tool
def add(a: int, b: int) -> int:
    """Adds `a` and `b`.

    Args:
        a: First int
        b: Second int
    """
    return a + b


@tool
def divide(a: int, b: int) -> float:
    """Divide `a` and `b`.

    Args:
        a: First int
        b: Second int
    """
    return a / b

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
# Augment the LLM with tools
tools = [add, multiply, divide, get_weather]
tools_by_name = {tool.name: tool for tool in tools}
model_with_tools = model.bind_tools(tools)

# Step 2: Define state

from langchain.messages import AnyMessage
from typing_extensions import TypedDict, Annotated
import operator


class MessagesState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    llm_calls: int

# Step 3: Define model node
from langchain.messages import SystemMessage


def llm_call(state: dict):
    """LLM decides whether to call a tool or not"""

    return {
        "messages": [
            model_with_tools.invoke(
                [
                    SystemMessage(
                        content="You are a helpful assistant tasked with performing arithmetic on a set of inputs."
                    )
                ]
                + state["messages"]
            )
        ],
        "llm_calls": state.get('llm_calls', 0) + 1
    }


# Step 4: Define tool node

from langchain.messages import ToolMessage


def tool_node(state: dict):
    """Performs the tool call"""

    result = []
    for tool_call in state["messages"][-1].tool_calls:
        tool = tools_by_name[tool_call["name"]]
        observation = tool.invoke(tool_call["args"])
        result.append(ToolMessage(content=observation, tool_call_id=tool_call["id"]))
    return {"messages": result}

# Step 5: Define logic to determine whether to end

from typing import Literal
from langgraph.graph import StateGraph, START, END


# Conditional edge function to route to the tool node or end based upon whether the LLM made a tool call
def should_continue(state: MessagesState) -> Literal["tool_node", END]:
    """Decide if we should continue the loop or stop based upon whether the LLM made a tool call"""

    messages = state["messages"]
    last_message = messages[-1]

    # If the LLM makes a tool call, then perform an action
    if last_message.tool_calls:
        return "tool_node"

    # Otherwise, we stop (reply to the user)
    return END

# Step 6: Build agent

# Build workflow
agent_builder = StateGraph(MessagesState)

# Add nodes
agent_builder.add_node("llm_call", llm_call)
agent_builder.add_node("tool_node", tool_node)

# Add edges to connect nodes
agent_builder.add_edge(START, "llm_call")
agent_builder.add_conditional_edges(
    "llm_call",
    should_continue,
    ["tool_node", END]
)
agent_builder.add_edge("tool_node", "llm_call")

# Compile the agent
agent = agent_builder.compile()


# from IPython.display import Image, display
# # Show the agent
# display(Image(agent.get_graph(xray=True).draw_mermaid_png()))

# Invoke
from langchain.messages import HumanMessage
messages = [HumanMessage(content="What is the weather in Paris? Also, what is 12 multiplied by 7, then add 10 and divide by 2?")]
messages = agent.invoke({"messages": messages})
for m in messages["messages"]:
    m.pretty_print()