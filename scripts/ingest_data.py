import os
import glob
import sys
from dotenv import load_dotenv

# Add project root to sys.path to allow imports from tools
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tools.retriever import index_pdf_documents

def main():
    # Load environment variables
    load_dotenv()
    
    data_dir = "data"
    if not os.path.exists(data_dir):
        print(f"Directory '{data_dir}' does not exist.")
        return

    # Find all PDF files in the data directory
    pdf_files = glob.glob(os.path.join(data_dir, "*.pdf"))
    
    if not pdf_files:
        print(f"No PDF files found in '{data_dir}'.")
        return

    print(f"Found {len(pdf_files)} PDF file(s):")
    for f in pdf_files:
        print(f" - {f}")

    print("\nStarting ingestion...")
    try:
        index_pdf_documents(pdf_files)
        print("\nIngestion complete!")
    except Exception as e:
        print(f"\nError during ingestion: {e}")

if __name__ == "__main__":
    main()
