import unittest
from unittest.mock import patch, MagicMock
from agents.rag_agent import build_rag_agent
from langchain_core.messages import HumanMessage, AIMessage

class TestGraphFlow(unittest.TestCase):

    @patch('agents.rag_agent.ChatOpenAI')
    def test_build_rag_agent(self, mock_chat_openai):
        mock_llm = MagicMock()
        mock_chat_openai.return_value = mock_llm
        mock_llm.bind_tools.return_value = mock_llm
        
        agent = build_rag_agent()
        self.assertIsNotNone(agent)

    # Testing the full graph flow is complex because it involves LLM calls.
    # We can test the nodes individually if we refactor them out, 
    # or use LangGraph's testing utilities if available.
    # For now, we just ensure the graph builds without error.

if __name__ == '__main__':
    unittest.main()
