"""RAGAnything service — reusable RAG component.

Wraps RAGAnything (which wraps LightRAG) to provide:
  - Content ingestion (text, images, tables, equations) → KG + 3 Milvus collections
  - Hybrid query (KG + vector retrieval, optionally VLM-enhanced)

The caller provides its own ``llm_model_func`` (e.g. from AI-ModelLayer).
VLM and embedding are handled internally via adapters.
"""

from __future__ import annotations

import asyncio
from collections.abc import Collection
from dataclasses import dataclass, field
import os
from typing import Any, Optional

import numpy as np

from lightrag import QueryParam
from raganything import RAGAnything, RAGAnythingConfig
from lightrag.utils import EmbeddingFunc

from .adapters import LLMAdapter, VLMAdapter, EmbeddingAdapter
from .config import EMBED_DIM, GRAPH_STORAGE


@dataclass(slots=True)
class RetrievalResult:
    """Structured retrieval context returned by ``RAGService.retrieve()``."""

    query: str
    mode: str
    chunks: list[dict[str, Any]] = field(default_factory=list)
    entities: list[dict[str, Any]] = field(default_factory=list)
    relationships: list[dict[str, Any]] = field(default_factory=list)

    def format_for_llm(self, max_chunks: int = 10) -> str:
        """Render retrieval results as compact prompt context."""
        parts: list[str] = []

        if self.chunks:
            parts.append("## Relevant document passages")
            for chunk in self.chunks[:max_chunks]:
                source = chunk.get("file_path", "unknown")
                content = chunk.get("content", "")
                parts.append(f"[{source}]\n{content}")

        if self.entities:
            parts.append("## Knowledge graph entities")
            for entity in self.entities[:20]:
                entity_name = entity.get("entity_name", "")
                entity_type = entity.get("entity_type", "")
                description = entity.get("description", "")
                parts.append(f"- {entity_name} ({entity_type}): {description}")

        if self.relationships:
            parts.append("## Knowledge graph relationships")
            for relationship in self.relationships[:20]:
                src_id = relationship.get("src_id", "")
                tgt_id = relationship.get("tgt_id", "")
                description = relationship.get("description", "")
                parts.append(f"- {src_id} -> {tgt_id}: {description}")

        return "\n\n".join(parts) if parts else "No relevant context found."


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

        Defaults to ``LLMAdapter()`` (DashScope Qwen).
        The caller (e.g. AI-ModelLayer) can override this.

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
        llm_model_func: Optional[LLMAdapter] = None,
        working_dir: str = "rag_workdir",
        workspace: str = "",
        vlm_adapter: Optional[VLMAdapter] = None,
        embedding_adapter: Optional[EmbeddingAdapter] = None,
        embedding_dim: int = EMBED_DIM,
        cosine_threshold: float = 0.5,
        enable_image_processing: bool = True,
        enable_table_processing: bool = True,
        enable_equation_processing: bool = False,
        excluded_content_types: Collection[str] | None = None,
    ):
        self.llm_model_func = llm_model_func or LLMAdapter()
        self.working_dir = working_dir
        self.workspace = workspace
        self.vlm = vlm_adapter or VLMAdapter()
        self.emb = embedding_adapter or EmbeddingAdapter()
        self.embedding_dim = embedding_dim
        self.cosine_threshold = cosine_threshold
        self.enable_image_processing = enable_image_processing
        self.enable_table_processing = enable_table_processing
        self.enable_equation_processing = enable_equation_processing
        self.excluded_content_types = self._normalize_content_types(
            excluded_content_types
        )

        self._rag: Optional[RAGAnything] = None
        self._init_lock = asyncio.Lock()

    @staticmethod
    def _normalize_content_types(
        content_types: Collection[str] | None,
    ) -> frozenset[str]:
        if not content_types:
            return frozenset()

        return frozenset(
            content_type.strip().lower()
            for content_type in content_types
            if isinstance(content_type, str) and content_type.strip()
        )

    def _resolve_excluded_content_types(
        self,
        excluded_content_types: Collection[str] | None,
    ) -> frozenset[str]:
        if excluded_content_types is None:
            return self.excluded_content_types

        return self._normalize_content_types(excluded_content_types)

    def _filter_content_list(
        self,
        content_list: list[dict[str, Any]],
        *,
        excluded_content_types: Collection[str] | None = None,
    ) -> list[dict[str, Any]]:
        resolved_excluded_types = self._resolve_excluded_content_types(
            excluded_content_types
        )
        if not resolved_excluded_types:
            return content_list

        return [
            item
            for item in content_list
            if not (
                isinstance(item, dict)
                and isinstance(item.get("type"), str)
                and item["type"].strip().lower() in resolved_excluded_types
            )
        ]

    def _build_rag_instance(self) -> RAGAnything:
        """Construct the upstream RAGAnything instance off the event loop."""
        emb_adapter = self.emb

        async def _embed(texts):
            if isinstance(texts, str):
                texts = [texts]
            result = await emb_adapter.embed(texts)
            return np.array(result)

        config = RAGAnythingConfig(
            working_dir=self.working_dir,
            enable_image_processing=self.enable_image_processing,
            enable_table_processing=self.enable_table_processing,
            enable_equation_processing=self.enable_equation_processing,
        )

        lightrag_kwargs: dict[str, Any] = {
            # Vector storage -> Milvus standalone (3 collections per workspace)
            "vector_storage": "MilvusVectorDBStorage",
            "vector_db_storage_cls_kwargs": {
                "cosine_better_than_threshold": self.cosine_threshold,
            },
            # KV storage -> Redis
            "kv_storage": "RedisKVStorage",
            "doc_status_storage": "RedisDocStatusStorage",
            # Graph storage -> configurable (NetworkXStorage | Neo4JStorage)
            # Neo4JStorage reads connection settings from env vars directly.
            "graph_storage": GRAPH_STORAGE,
        }
        if self.workspace:
            lightrag_kwargs["workspace"] = self.workspace

        return RAGAnything(
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

    async def _ensure_initialized(self) -> RAGAnything:
        """Lazily initialize RAGAnything on first use."""
        if self._rag is not None:
            return self._rag

        async with self._init_lock:
            if self._rag is not None:
                return self._rag

            self._rag = await asyncio.to_thread(self._build_rag_instance)
        return self._rag

    async def _ensure_lightrag_ready(self, operation: str) -> RAGAnything:
        """Ensure the internal LightRAG instance exists for query-style operations."""
        rag = await self._ensure_initialized()
        if rag.lightrag is None:
            rag._parser_installation_checked = True
            result = await rag._ensure_lightrag_initialized()
            if isinstance(result, dict) and not result.get("success", True):
                raise RuntimeError(
                    f"Failed to initialize LightRAG for {operation}: {result.get('error')}"
                )
        return rag

    async def warmup(self) -> None:
        """Pre-initialize both RAGAnything and LightRAG for this workspace."""
        await self._ensure_lightrag_ready("startup warmup")

    async def insert(
        self,
        content_list: list[dict[str, Any]],
        file_path: str,
        *,
        display_stats: bool = False,
        excluded_content_types: Collection[str] | None = None,
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

        excluded_content_types:
            Content block types to skip before chunk generation. For example,
            pass ``{"footer"}`` to exclude footer blocks from ingestion.

        Returns
        -------
        dict
            Result from RAGAnything's ``insert_content_list``.
        """
        rag = await self._ensure_initialized()
        filtered_content_list = self._filter_content_list(
            content_list,
            excluded_content_types=excluded_content_types,
        )
        if not filtered_content_list:
            raise ValueError("All content blocks were excluded before ingestion.")

        return await rag.insert_content_list(
            content_list=filtered_content_list,
            file_path=file_path,
            display_stats=display_stats,
        )

    async def insert_document(
        self,
        file_path: str,
        output_dir: str | None = None,
        *,
        display_stats: bool = False,
        parse_method: str | None = None,
        split_by_character: str | None = None,
        split_by_character_only: bool = False,
        doc_id: str | None = None,
        excluded_content_types: Collection[str] | None = None,
        **parser_kwargs: Any,
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
        parse_method:
            Parser method forwarded to ``RAGAnything.parse_document()``.
        split_by_character:
            Optional character passed to LightRAG chunk splitting.
        split_by_character_only:
            Whether LightRAG should split only by ``split_by_character``.
        doc_id:
            Optional stable document id for deduplication and tracking.
        excluded_content_types:
            Content block types to skip before chunk generation. For example,
            pass ``{"footer"}`` to exclude footer blocks from ingestion.
        parser_kwargs:
            Additional parser-specific options forwarded to
            ``RAGAnything.parse_document()``.
        """
        rag = await self._ensure_initialized()
        if output_dir is None:
            output_dir = os.path.join(self.working_dir, "parsed")

        content_list, parsed_doc_id = await rag.parse_document(
            file_path=file_path,
            output_dir=output_dir,
            display_stats=display_stats,
            parse_method=parse_method,
            **parser_kwargs,
        )
        filtered_content_list = self._filter_content_list(
            content_list,
            excluded_content_types=excluded_content_types,
        )
        if not filtered_content_list:
            raise ValueError("All parsed content blocks were excluded before ingestion.")

        effective_doc_id = doc_id
        if effective_doc_id is None:
            effective_doc_id = (
                parsed_doc_id
                if len(filtered_content_list) == len(content_list)
                else rag._generate_content_based_doc_id(filtered_content_list)
            )

        return await rag.insert_content_list(
            content_list=filtered_content_list,
            file_path=file_path,
            split_by_character=split_by_character,
            split_by_character_only=split_by_character_only,
            doc_id=effective_doc_id,
            display_stats=display_stats,
        )

    async def query(
        self,
        question: str,
        *,
        mode: str = "hybrid",
        vlm_enhanced: bool | None = None,
    ) -> str:
        """Query the knowledge graph + vector store.

        Parameters
        ----------
        question:
            The user's question.
        mode:
            LightRAG query mode: ``"naive"``, ``"local"``, ``"global"``,
            ``"hybrid"`` (default), ``"mix"``, or ``"bypass"``.
        vlm_enhanced:
            Whether to replace image paths in retrieved context with encoded
            images for VLM-assisted answering. ``None`` defers to RAGAnything's
            default behavior, which enables this automatically when a vision
            model is available.

        Returns
        -------
        str
            The generated answer.
        """
        rag = await self._ensure_lightrag_ready("query")
        query_kwargs: dict[str, Any] = {}
        if vlm_enhanced is not None:
            query_kwargs["vlm_enhanced"] = vlm_enhanced
        return await rag.aquery(question, mode=mode, **query_kwargs)

    async def retrieve(
        self,
        question: str,
        *,
        param: QueryParam | None = None,
    ) -> RetrievalResult:
        """Retrieve structured context without running answer generation.

        Parameters
        ----------
        question:
            The user's question.
        param:
            Optional LightRAG query parameters. Defaults to ``QueryParam(mode="hybrid")``.
        """
        rag = await self._ensure_lightrag_ready("retrieval")
        query_param = param or QueryParam(mode="hybrid")
        raw = await rag.lightrag.aquery_data(question, param=query_param)
        data = raw.get("data", {}) if isinstance(raw, dict) else {}
        return RetrievalResult(
            query=question,
            mode=query_param.mode,
            chunks=data.get("chunks", []),
            entities=data.get("entities", []),
            relationships=data.get("relationships", []),
        )
 