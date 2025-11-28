import os
from dotenv import load_dotenv
from agents.rag_agent import build_rag_agent
from langchain_core.messages import HumanMessage

def main():
    # Load environment variables
    load_dotenv()
    
    # Check for API keys
    if not os.getenv("OPENAI_API_KEY"):
        print("Warning: OPENAI_API_KEY not found in environment variables.")
    if not os.getenv("OPENWEATHER_API_KEY"):
        print("Warning: OPENWEATHER_API_KEY not found in environment variables.")

    # Build the agent
    print("Building RAG Agent...")
    agent = build_rag_agent()

    # Interactive loop
    print("Agent ready! Type 'exit' to quit.")
    while True:
        try:
            user_input = input("User: ")
            if user_input.lower() in ["exit", "quit"]:
                break
            
            inputs = {"messages": [HumanMessage(content=user_input)]}
            
            # Stream the output
            for event in agent.stream(inputs):
                for key, value in event.items():
                    if key == "chatbot":
                        # The chatbot node returns the AIMessage
                        last_msg = value['messages'][-1]
                        if last_msg.content:
                            print(f"Assistant: {last_msg.content}")
                    elif key == "tools":
                        # Tools node output
                        pass
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
