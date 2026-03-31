"""Environment-driven configuration for rag_shared.

Only VLM, embedding, and Milvus settings — no LLM chat config.
The chat/LLM layer is owned by the caller (e.g. AI-ModelLayer).
"""

import os
from pathlib import Path
from urllib.parse import urlparse

from dotenv import dotenv_values, load_dotenv


def _iter_dotenv_candidates() -> list[Path]:
    candidates: list[Path] = []
    seen: set[Path] = set()

    for base in [Path.cwd(), *Path(__file__).resolve().parents]:
        candidate = base / ".env"
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.is_file():
            candidates.append(candidate)

    return candidates


def _has_valid_redis_scheme(value: str | None) -> bool:
    if not value:
        return False
    return urlparse(value).scheme in {"redis", "rediss", "unix"}


def _load_environment() -> None:
    dotenv_candidates = _iter_dotenv_candidates()
    for dotenv_path in dotenv_candidates:
        load_dotenv(dotenv_path, override=False)

    current_redis_uri = os.getenv("REDIS_URI")
    if _has_valid_redis_scheme(current_redis_uri):
        return

    for dotenv_path in dotenv_candidates:
        dotenv_redis_uri = dotenv_values(dotenv_path).get("REDIS_URI")
        if _has_valid_redis_scheme(dotenv_redis_uri):
            os.environ["REDIS_URI"] = dotenv_redis_uri
            return


_load_environment()

# --- DashScope API ---
DASHSCOPE_API_KEY: str = os.getenv("DASHSCOPE_API_KEY", "")

# --- LLM (Text) ---
LLM_MODEL: str = os.getenv("LLM_MODEL", "qwen-plus")
LLM_URL: str = os.getenv(
    "LLM_URL",
    "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
)

# --- VLM (Vision Language Model) ---
VLM_MODEL: str = os.getenv("VLM_MODEL", "qwen-vl-max")
VLM_URL: str = os.getenv(
    "VLM_URL",
    "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
)

# --- Embedding ---
EMBED_MODEL: str = os.getenv("EMBED_MODEL", "text-embedding-v4")
EMBED_URL: str = os.getenv(
    "EMBED_URL",
    "https://dashscope.aliyuncs.com/compatible-mode/v1",
)
EMBED_DIM: int = int(os.getenv("EMBED_DIM", "1024"))

# --- Milvus Standalone ---
# LightRAG reads MILVUS_URI, MILVUS_USER, MILVUS_PASSWORD, MILVUS_TOKEN,
# MILVUS_DB_NAME directly from env vars for its MilvusVectorDBStorage.
MILVUS_URI: str = os.getenv("MILVUS_URI", "http://localhost:19530")

# --- Redis (KV + doc-status storage) ---
# LightRAG reads REDIS_URI directly from env vars for RedisKVStorage.
REDIS_URI: str = os.getenv("REDIS_URI", "redis://localhost:6379")

# --- Graph Storage ---
# LightRAG graph backend: "NetworkXStorage" (local, default) or "Neo4JStorage".
# When Neo4JStorage is selected, LightRAG reads NEO4J_URI, NEO4J_USERNAME,
# NEO4J_PASSWORD, NEO4J_DATABASE, and NEO4J_WORKSPACE directly from env vars.
# Default the username to "neo4j" for the standard Docker image.
_SUPPORTED_GRAPH_STORAGES = {"NetworkXStorage", "Neo4JStorage"}

GRAPH_STORAGE: str = os.getenv("GRAPH_STORAGE", "NetworkXStorage")
if GRAPH_STORAGE not in _SUPPORTED_GRAPH_STORAGES:
    raise ValueError(
        f"Unsupported GRAPH_STORAGE={GRAPH_STORAGE!r}. "
        f"Choose from: {', '.join(sorted(_SUPPORTED_GRAPH_STORAGES))}"
    )

if GRAPH_STORAGE == "Neo4JStorage":
    os.environ.setdefault("NEO4J_USERNAME", "neo4j")
    _neo4j_required = {"NEO4J_URI", "NEO4J_PASSWORD"}
    _neo4j_missing = [v for v in sorted(_neo4j_required) if not os.getenv(v)]
    if _neo4j_missing:
        raise RuntimeError(
            f"GRAPH_STORAGE=Neo4JStorage requires env vars: {', '.join(_neo4j_missing)}"
        )

# --- Timeouts ---
TIMEOUT: int = int(os.getenv("RAG_TIMEOUT", "120"))
