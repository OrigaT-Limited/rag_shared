"""VLM and Embedding adapters for DashScope (Qwen) OpenAI-compatible APIs.

No chat/LLM adapter — that responsibility belongs to the caller.
"""

import asyncio
import base64
from typing import List, Optional

import aiohttp
from openai import OpenAI

from .config import (
    DASHSCOPE_API_KEY,
    VLM_MODEL,
    VLM_URL,
    EMBED_MODEL,
    EMBED_URL,
    EMBED_DIM,
    TIMEOUT,
)

_HEADERS = {
    "Authorization": f"Bearer {DASHSCOPE_API_KEY}",
    "Content-Type": "application/json",
}


class VLMAdapter:
    """Vision Language Model adapter for RAGAnything.

    Implements the three call patterns RAGAnything expects from vision_model_func:
      1. Image captioning:  (prompt, image_data=base64, system_prompt=...)
      2. Text-only fallback: (prompt, system_prompt=...)
      3. Full messages list: ("", messages=[{role, content}, ...])
    """

    def __init__(
        self,
        model: str = VLM_MODEL,
        url: str = VLM_URL,
        api_key: str = DASHSCOPE_API_KEY,
    ):
        self.model = model
        self.url = url
        self._headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

    async def __call__(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        history_messages: Optional[list] = None,
        image_data: Optional[str] = None,
        messages: Optional[list] = None,
        **kwargs,
    ) -> str:
        """RAGAnything-compatible vision model function."""
        if messages:
            payload = {
                "model": self.model,
                "messages": [m for m in messages if m is not None],
                "max_tokens": kwargs.get("max_tokens", 4096),
                "temperature": 0.2,
                "enable_thinking": False,
            }
        elif image_data:
            user_content = [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}},
            ]
            msgs = []
            if system_prompt:
                msgs.append({"role": "system", "content": system_prompt})
            msgs.append({"role": "user", "content": user_content})
            payload = {
                "model": self.model,
                "messages": msgs,
                "max_tokens": kwargs.get("max_tokens", 4096),
                "temperature": 0.2,
                "enable_thinking": False,
            }
        else:
            msgs = []
            if system_prompt:
                msgs.append({"role": "system", "content": system_prompt})
            msgs.append({"role": "user", "content": prompt})
            payload = {
                "model": self.model,
                "messages": msgs,
                "max_tokens": kwargs.get("max_tokens", 4096),
                "temperature": 0.5,
                "enable_thinking": False,
            }

        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=TIMEOUT)) as session:
            async with session.post(self.url, json=payload, headers=self._headers) as resp:
                resp.raise_for_status()
                data = await resp.json()
                return data["choices"][0]["message"]["content"]


class EmbeddingAdapter:
    """Embedding adapter using OpenAI-compatible DashScope endpoint."""

    def __init__(
        self,
        model: str = EMBED_MODEL,
        base_url: str = EMBED_URL,
        api_key: str = DASHSCOPE_API_KEY,
        dim: int = EMBED_DIM,
    ):
        self.model = model
        self.dim = dim
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    async def embed(self, texts: str | list[str]) -> list[float] | list[list[float]]:
        """Embed a single string or list of strings.

        Returns a single embedding (list[float]) for a str input,
        or a list of embeddings (list[list[float]]) for a list input.
        """
        single = isinstance(texts, str)
        inputs = [texts] if single else texts
        resp = await asyncio.to_thread(
            self.client.embeddings.create,
            model=self.model,
            input=inputs,
            dimensions=self.dim,
        )
        embeddings = [item.embedding for item in resp.data]
        return embeddings[0] if single else embeddings
