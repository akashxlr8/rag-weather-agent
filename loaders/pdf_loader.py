from typing import List
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_pdf(path: str) -> List[Document]:
    """
    Loads a PDF file and returns a list of Documents.
    """
    loader = PyPDFLoader(path)
    return loader.load()

def chunk_documents(docs: List[Document], chunk_size: int = 1000, overlap: int = 200) -> List[Document]:
    """
    Chunks a list of Documents into smaller pieces.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap
    )
    return text_splitter.split_documents(docs)
