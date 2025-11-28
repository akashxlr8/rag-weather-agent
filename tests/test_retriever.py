import unittest
from unittest.mock import patch, MagicMock
from tools.retriever import retrieve_documents, index_pdf_documents

class TestRetriever(unittest.TestCase):

    @patch('tools.retriever.get_retriever')
    def test_retrieve_documents(self, mock_get_retriever):
        mock_retriever = MagicMock()
        mock_doc = MagicMock()
        mock_doc.page_content = "This is a test document."
        mock_retriever.invoke.return_value = [mock_doc]
        mock_get_retriever.return_value = mock_retriever

        result = retrieve_documents("test query")
        self.assertIn("This is a test document.", result)

    @patch('tools.retriever.load_pdf')
    @patch('tools.retriever.chunk_documents')
    @patch('tools.retriever.create_collection')
    @patch('tools.retriever.upsert_documents')
    def test_index_pdf_documents(self, mock_upsert, mock_create, mock_chunk, mock_load):
        mock_load.return_value = ["raw_doc"]
        mock_chunk.return_value = ["chunk1", "chunk2"]

        index_pdf_documents(["test.pdf"])

        mock_load.assert_called_with("test.pdf")
        mock_chunk.assert_called()
        mock_create.assert_called()
        mock_upsert.assert_called()

if __name__ == '__main__':
    unittest.main()
