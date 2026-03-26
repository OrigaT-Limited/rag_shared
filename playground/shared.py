"""Shared utilities for the RAG Playground GUI."""

from __future__ import annotations

import asyncio
import threading


# ---------------------------------------------------------------------------
# Token usage tracker
# ---------------------------------------------------------------------------
class TokenTracker:
    """Accumulates token usage reported by adapters, split by phase."""

    def __init__(self):
        self._lock = threading.Lock()
        self._phase: str = "insert"  # "insert", "query", or "retrieve"
        # {phase: {adapter: {field: count}}}
        self._counts: dict[str, dict[str, dict[str, int]]] = {
            "insert": {},
            "query": {},
            "retrieve": {},
        }

    def set_phase(self, phase: str):
        self._phase = phase

    def on_usage(self, adapter_name: str, usage: dict):
        with self._lock:
            phase_data = self._counts[self._phase]
            if adapter_name not in phase_data:
                phase_data[adapter_name] = {}
            for key, val in usage.items():
                if isinstance(val, int):
                    phase_data[adapter_name][key] = (
                        phase_data[adapter_name].get(key, 0) + val
                    )

    def summary(self, phase: str) -> str:
        with self._lock:
            phase_data = self._counts.get(phase, {})
        if not phase_data:
            return "No usage recorded"
        lines = []
        total_all = 0
        for adapter in sorted(phase_data):
            fields = phase_data[adapter]
            total = fields.get("total_tokens", 0)
            prompt = fields.get("prompt_tokens", 0)
            completion = fields.get("completion_tokens", 0)
            total_all += total
            lines.append(f"  {adapter}: prompt={prompt:,}  completion={completion:,}  total={total:,}")
        lines.insert(0, f"Total tokens: {total_all:,}")
        return "\n".join(lines)

    def reset(self, phase: str):
        with self._lock:
            self._counts[phase] = {}


# ---------------------------------------------------------------------------
# Async bridge — run coroutines from the tkinter (sync) world
# ---------------------------------------------------------------------------
_loop: asyncio.AbstractEventLoop | None = None


def _get_loop() -> asyncio.AbstractEventLoop:
    global _loop
    if _loop is None or _loop.is_closed():
        _loop = asyncio.new_event_loop()
        t = threading.Thread(target=_loop.run_forever, daemon=True)
        t.start()
    return _loop


def run_coro(coro):
    """Schedule *coro* on the background event loop, return a Future."""
    return asyncio.run_coroutine_threadsafe(coro, _get_loop())
