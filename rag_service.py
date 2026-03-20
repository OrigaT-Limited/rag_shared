"""RAGAnything service — reusable RAG component.

Wraps RAGAnything (which wraps LightRAG) to provide:
  - Content ingestion (text, images, tables, equations) → KG + 3 Milvus collections
  - Hybrid query (KG + vector retrieval, optionally VLM-enhanced)

The caller provides its own ``llm_model_func`` (e.g. from AI-ModelLayer).
VLM and embedding are handled internally via adapters.
"""

from __future__ import annotations

import os
from typing import Any, Callable, Coroutine, Optional

from raganything import RAGAnything, RAGAnythingConfig
from lightrag.utils import EmbeddingFunc

from .adapters import VLMAdapter, EmbeddingAdapter
from .config import EMBED_DIM


class RAGService:
    """High-level RAG service backed by RAGAnything + Milvus standalone.

    Parameters
    ----------
    llm_model_func:
        Async function matching LightRAG's signature::

            async def llm_model_func(
                prompt: str,
                system_prompt: str | None = None,
                history_messages: list = [],
                **kwargs,
            ) -> str

        The caller (e.g. AI-ModelLayer) provides this.

    working_dir:
        Local directory for LightRAG's KV stores (graph JSON, doc status, cache).
        Also used as the default workspace prefix for Milvus collection names
        unless ``workspace`` or ``MILVUS_WORKSPACE`` env var is set.

    workspace:
        Milvus collection name prefix for data isolation.
        LightRAG creates 3 collections per workspace:
          - ``{workspace}_entities``
          - ``{workspace}_relationships``
          - ``{workspace}_chunks``
        If empty string, falls back to ``MILVUS_WORKSPACE`` env var,
        then namespace alone (no prefix).

    vlm_adapter:
        Optional pre-configured VLMAdapter. Created with defaults if not given.

    embedding_adapter:
        Optional pre-configured EmbeddingAdapter. Created with defaults if not given.

    embedding_dim:
        Embedding vector dimension. Must match the embedding model output.

    cosine_threshold:
        Minimum cosine similarity for a vector match to be considered relevant.

    enable_image_processing:
        Whether RAGAnything should process image content via VLM.

    enable_table_processing:
        Whether RAGAnything should process table content.

    enable_equation_processing:
        Whether RAGAnything should process equation content.
    """

    def __init__(
        self,
        llm_model_func: Callable[..., Coroutine[Any, Any, str]],
        working_dir: str = "rag_workdir",
        workspace: str = "",
        vlm_adapter: Optional[VLMAdapter] = None,
        embedding_adapter: Optional[EmbeddingAdapter] = None,
        embedding_dim: int = EMBED_DIM,
        cosine_threshold: float = 0.5,
        enable_image_processing: bool = True,
        enable_table_processing: bool = True,
        enable_equation_processing: bool = True,
    ):
        self.llm_model_func = llm_model_func
        self.working_dir = working_dir
        self.workspace = workspace
        self.vlm = vlm_adapter or VLMAdapter()
        self.emb = embedding_adapter or EmbeddingAdapter()
        self.embedding_dim = embedding_dim
        self.cosine_threshold = cosine_threshold
        self.enable_image_processing = enable_image_processing
        self.enable_table_processing = enable_table_processing
        self.enable_equation_processing = enable_equation_processing

        self._rag: Optional[RAGAnything] = None

    async def _ensure_initialized(self) -> RAGAnything:
        """Lazily initialize RAGAnything on first use."""
        if self._rag is not None:
            return self._rag

        emb_adapter = self.emb

        async def _embed(texts):
            if isinstance(texts, str):
                texts = [texts]
            return await emb_adapter.embed(texts)

        config = RAGAnythingConfig(
            working_dir=self.working_dir,
            enable_image_processing=self.enable_image_processing,
            enable_table_processing=self.enable_table_processing,
            enable_equation_processing=self.enable_equation_processing,
        )

        lightrag_kwargs: dict[str, Any] = {
            # Vector storage → Milvus standalone (3 collections per workspace)
            "vector_storage": "MilvusVectorDBStorage",
            "vector_db_storage_cls_kwargs": {
                "cosine_better_than_threshold": self.cosine_threshold,
            },
            # KV storage → Redis
            "kv_storage": "RedisKVStorage",
            "doc_status_storage": "RedisDocStatusStorage",
        }
        if self.workspace:
            lightrag_kwargs["workspace"] = self.workspace

        self._rag = RAGAnything(
            config=config,
            llm_model_func=self.llm_model_func,
            vision_model_func=self.vlm,
            embedding_func=EmbeddingFunc(
                embedding_dim=self.embedding_dim,
                max_token_size=8192,
                func=_embed,
            ),
            lightrag_kwargs=lightrag_kwargs,
        )
        return self._rag

    async def insert(
        self,
        content_list: list[dict[str, Any]],
        file_path: str,
        *,
        display_stats: bool = False,
    ) -> dict:
        """Ingest content into the knowledge graph + vector store.

        Parameters
        ----------
        content_list:
            List of content dicts. Each must have a ``"type"`` key.
            Supported types: ``"text"``, ``"image"``, ``"table"``, ``"equation"``.

            Text example::

                {"type": "text", "text": "...", "page_idx": 0}

            Image example::

                {
                    "type": "image",
                    "img_path": "/abs/path/to/image.jpg",
                    "image_caption": ["description"],
                    "image_footnote": [],
                    "page_idx": 0,
                }

        file_path:
            Identifier for the source document (used for dedup/tracking).

        display_stats:
            Whether to print ingestion statistics.

        Returns
        -------
        dict
            Result from RAGAnything's ``insert_content_list``.
        """
        rag = await self._ensure_initialized()
        return await rag.insert_content_list(
            content_list=content_list,
            file_path=file_path,
            display_stats=display_stats,
        )

    async def insert_document(
        self,
        file_path: str,
        output_dir: str | None = None,
        *,
        display_stats: bool = False,
    ) -> dict:
        """Ingest a document file (PDF, DOCX, etc.) via RAGAnything's built-in parser.

        Requires MinerU or Docling parser to be installed.

        Parameters
        ----------
        file_path:
            Path to the document file.
        output_dir:
            Directory for parser output. Defaults to ``{working_dir}/parsed``.
        display_stats:
            Whether to print ingestion statistics.
        """
        rag = await self._ensure_initialized()
        if output_dir is None:
            output_dir = os.path.join(self.working_dir, "parsed")
        return await rag.process_document_complete(
            file_path=file_path,
            output_dir=output_dir,
            display_stats=display_stats,
        )

    async def query(
        self,
        question: str,
        *,
        mode: str = "hybrid",
    ) -> str:
        """Query the knowledge graph + vector store.

        Parameters
        ----------
        question:
            The user's question.
        mode:
            LightRAG query mode: ``"naive"``, ``"local"``, ``"global"``,
            ``"hybrid"`` (default), or ``"mix"``.

        Returns
        -------
        str
            The generated answer.
        """
        rag = await self._ensure_initialized()
        return await rag.aquery(question, mode=mode)
