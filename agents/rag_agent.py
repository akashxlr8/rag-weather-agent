from typing import Annotated, Literal, TypedDict
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from tools.weather import get_weather
from tools.advanced_retriever import advanced_retrieve
from tools.prompts import AGENT_SYSTEM_PROMPT

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

def build_rag_agent():
    """
    Builds the RAG agent graph.
    """
    # Define tools
    @tool
    def weather_tool(city: str):
        """Get the weather for a city.
            Arg:
                city: Name of the city to get the weather for.
        
        """
        return get_weather(city)

    @tool
    def retriever_tool(query: str):
        """Retrieve information from documents with automatic relevance grading and query rewriting.
            Arg:
                query: The user's query to search for. 
                
                
        Always use this tool for document retrieval from the knowledge base.
        Try to provide relevant query string based on user question rebuilding from the context and previous interactions.
        
        Example:
            if user asks: "Who is Akash and what is his role?"
            call retriever_tool with query: "Who is Akash? What is his role? What are his responsibilities? Information about Akash."
        """
        return advanced_retrieve(query)

    tools = [weather_tool, retriever_tool]
    
    # Initialize LLM with tools
    llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0, max_completion_tokens=2000)
    llm_with_tools = llm.bind_tools(tools)

    # Define nodes
    def chatbot(state: AgentState):
        messages = state["messages"]
    
        messages = [SystemMessage(content=AGENT_SYSTEM_PROMPT)] + messages
        return {"messages": [llm_with_tools.invoke(messages)]}

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
