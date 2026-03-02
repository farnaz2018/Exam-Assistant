"""
Document loaders for PDF, Word, and image (PNG) files.
"""

from PIL import Image
import pytesseract
from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader
from langchain_core.documents import Document


def load_documents(file_path: str):
    """
    Load documents from a given file path.

    Returns a list of LangChain `Document` objects.
    """
    if file_path.endswith(".pdf"):
        return PyPDFLoader(file_path).load()
    if file_path.endswith(".docx"):
        return Docx2txtLoader(file_path).load()
    if file_path.endswith(".png"):
        return load_image(file_path)
    raise ValueError("Unsupported file type")


def load_image(file_path: str):
    text = pytesseract.image_to_string(Image.open(file_path))
    return [Document(page_content=text)]
