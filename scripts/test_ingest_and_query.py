import sys
import os

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dotenv import load_dotenv
from tools.retriever import index_pdf_documents, retrieve_documents

def main():
    load_dotenv()
    
    pdf_path = "data/test_weather.pdf"
    if not os.path.exists(pdf_path):
        print(f"Error: {pdf_path} not found. Run scripts/create_test_pdf.py first.")
        return

    print(f"--- Indexing {pdf_path} ---")
    try:
        index_pdf_documents([pdf_path])
    except Exception as e:
        print(f"Error during indexing: {e}")
        return

    print("\n--- Testing Retrieval ---")
    query = "What is San Francisco known for?"
    print(f"Query: {query}")
    try:
        result = retrieve_documents(query)
        print(f"Result:\n{result}")
    except Exception as e:
        print(f"Error during retrieval: {e}")

if __name__ == "__main__":
    main()
