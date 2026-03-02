"""
LLM and embeddings configuration. Uses Azure OpenAI when Azure env vars are set.
"""

import os
import re


def _use_azure() -> bool:
    return bool(os.getenv("AZURE_OPENAI_ENDPOINT") or os.getenv("AZURE_OPENAI_API_KEY"))


def _clean_text_for_embedding(text: str) -> str:
    """Ensure Azure/OpenAI embeddings receive plain strings (no control chars)."""
    if not isinstance(text, str):
        text = str(text)
    # Strip and replace control characters / problematic bytes
    text = text.strip()
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", " ", text)
    return text[:8192] if len(text) > 8192 else text  # stay under typical token limit


class _AzureEmbeddingsOneByOne:
    """Wraps Azure embeddings to send one string per request; avoids 'Unsupported data type' batch issues."""

    def __init__(self, embeddings):
        self._embeddings = embeddings

    def embed_documents(self, texts):
        if not texts:
            return []
        # Ensure clean strings; embed one at a time so Azure always gets a single string
        out = []
        for t in texts:
            s = _clean_text_for_embedding(t)
            if not s:
                s = " "  # Azure needs at least one character
            vec = self._embeddings.embed_documents([s])
            out.append(vec[0])
        return out

    def embed_query(self, text: str):
        s = _clean_text_for_embedding(text)
        return self._embeddings.embed_query(s)
    
    def __getattr__(self, name):
        return getattr(self._embeddings, name)


def get_llm():
    """Return chat model (Azure OpenAI or OpenAI) from env."""
    if _use_azure():
        from langchain_openai import AzureChatOpenAI
        return AzureChatOpenAI(
            azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
            temperature=0,
        )
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(model="gpt-4o-mini", temperature=0)


def get_embeddings():
    """Return embeddings (Azure OpenAI or OpenAI) from env. Use same for ingest and retriever."""
    if _use_azure():
        from langchain_openai import AzureOpenAIEmbeddings
        base = AzureOpenAIEmbeddings(
            azure_deployment=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-large"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
        )
        return _AzureEmbeddingsOneByOne(base)
    from langchain_openai import OpenAIEmbeddings
    return OpenAIEmbeddings()
