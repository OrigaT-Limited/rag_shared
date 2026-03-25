"""rag_shared — Reusable RAG component backed by RAGAnything + Milvus + Redis."""

from .adapters import EmbeddingAdapter, LLMAdapter, UsageCallback, VLMAdapter
from .config import EMBED_DIM
from .rag_service import RAGService, RetrievalResult

__all__ = [
    "RAGService",
    "RetrievalResult",
    "LLMAdapter",
    "VLMAdapter",
    "EmbeddingAdapter",
    "UsageCallback",
    "EMBED_DIM",
]
