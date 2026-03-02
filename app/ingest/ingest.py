"""
High-level ingestion helpers for the exam assistant.
"""

from app.ingest.loaders import load_documents, load_image
from app.config import get_embeddings
from langchain_community.vectorstores import Chroma

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter


def ingest_file(path: str):
    """
    Ingest a single file and return a list of Document objects.

    All supported formats use the same return type.
    """
    if path.endswith((".pdf", ".docx")):
        return load_documents(path)
    if path.endswith(".png"):
        return load_image(path)
    raise ValueError("Unsupported file type")


def create_vector_store(documents):
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = splitter.split_documents(documents)
    # Azure OpenAI rejects empty or non-string input; keep only non-empty string content
    from langchain_core.documents import Document
    safe_chunks = []
    for doc in chunks:
        text = getattr(doc, "page_content", None) or ""
        if isinstance(text, str) and text.strip():
            safe_chunks.append(Document(page_content=text.strip(), metadata=getattr(doc, "metadata", None) or {}))
    if not safe_chunks:
        raise ValueError("No valid text chunks to embed. Check that your documents contain readable text.")
    return Chroma.from_documents(
        safe_chunks,
        get_embeddings(),
        persist_directory="data/vectorstore",
    )
