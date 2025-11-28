from typing import Annotated, Literal, TypedDict
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from tools.weather import get_weather
from tools.retriever import retrieve_documents

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

def build_rag_agent():
    """
    Builds the RAG agent graph.
    """
    # Define tools
    @tool
    def weather_tool(city: str):
        """Get the weather for a city."""
        return get_weather(city)

    @tool
    def retriever_tool(query: str):
        """Retrieve information from documents."""
        return retrieve_documents(query)

    tools = [weather_tool, retriever_tool]
    
    # Initialize LLM with tools
    llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)
    llm_with_tools = llm.bind_tools(tools)

    # Define nodes
    def chatbot(state: AgentState):
        return {"messages": [llm_with_tools.invoke(state["messages"])]}

    def tools_node(state: AgentState):
        # Simple tool execution node (in a real app, use ToolNode from langgraph.prebuilt)
        last_message = state["messages"][-1]
        if not isinstance(last_message, AIMessage) or not last_message.tool_calls:
            return {}
        
        results = []
        for tool_call in last_message.tool_calls:
            if tool_call["name"] == "weather_tool":
                res = weather_tool.invoke(tool_call["args"])
                results.append(res)
            elif tool_call["name"] == "retriever_tool":
                res = retriever_tool.invoke(tool_call["args"])
                results.append(res)
        
        # For simplicity in Phase 1, we just append the result as a message
        # In a full implementation, we'd create ToolMessages
        from langchain_core.messages import ToolMessage
        tool_messages = []
        for tool_call, res in zip(last_message.tool_calls, results):
             tool_messages.append(ToolMessage(tool_call_id=tool_call["id"], content=str(res)))
        
        return {"messages": tool_messages}

    # Define conditional edge
    def route_tools(state: AgentState) -> Literal["tools", "__end__"]:
        last_message = state["messages"][-1]
        if isinstance(last_message, AIMessage) and last_message.tool_calls:
            return "tools"
        return "__end__"

    # Build graph
    graph_builder = StateGraph(AgentState)
    graph_builder.add_node("chatbot", chatbot)
    graph_builder.add_node("tools", tools_node)

    graph_builder.add_edge("tools", "chatbot") # Loop back to chatbot after tools
    graph_builder.set_entry_point("chatbot")
    
    graph_builder.add_conditional_edges(
        "chatbot",
        route_tools
    )

    return graph_builder.compile()
