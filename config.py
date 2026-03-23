"""Environment-driven configuration for rag_shared.

Only VLM, embedding, and Milvus settings — no LLM chat config.
The chat/LLM layer is owned by the caller (e.g. AI-ModelLayer).
"""

import os
from dotenv import load_dotenv

load_dotenv()

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

# --- Timeouts ---
TIMEOUT: int = int(os.getenv("RAG_TIMEOUT", "120"))
