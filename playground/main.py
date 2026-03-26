"""Playground GUI — select files via macOS Finder and insert them into RAG.

Run:
    python playground/main.py

Requires:
    - rag_shared installed (or run from repo root so it's importable)
    - Milvus standalone running at MILVUS_URI
    - Redis running at REDIS_URI
    - DASHSCOPE_API_KEY set in .env or environment
"""

from __future__ import annotations

import sys
import tkinter as tk
from tkinter import ttk
from pathlib import Path

# Ensure rag_shared is importable when running from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from rag_shared import RAGService, LLMAdapter, VLMAdapter, EmbeddingAdapter

from shared import TokenTracker  # noqa: E402
from tabs import IngestTab, QueryTab, RetrieveTab, ChunksTab  # noqa: E402


# ---------------------------------------------------------------------------
# GUI Application
# ---------------------------------------------------------------------------
class RAGPlaygroundApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        root.title("RAG Playground")
        root.geometry("800x700")
        root.minsize(700, 580)

        self.tracker = TokenTracker()

        llm = LLMAdapter(on_usage=self.tracker.on_usage)
        vlm = VLMAdapter(on_usage=self.tracker.on_usage)
        emb = EmbeddingAdapter(on_usage=self.tracker.on_usage)

        self.service = RAGService(
            llm_model_func=llm,
            vlm_adapter=vlm,
            embedding_adapter=emb,
            workspace="default",
        )

        self._build_ui()
        self.root.after(0, self.chunks_tab.refresh)

    def _build_ui(self):
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        # --- Tab 1: Chunks (created first so ingest can reference it) ---
        chunks_frame = ttk.Frame(notebook, padding=8)
        self.chunks_tab = ChunksTab(
            chunks_frame, self.root, self.service, self.tracker,
        )

        # --- Tab 2: Ingest ---
        ingest_frame = ttk.Frame(notebook, padding=8)
        self.ingest_tab = IngestTab(
            ingest_frame, self.root, self.service, self.tracker,
            on_ingest_done=self.chunks_tab.refresh,
        )

        # --- Tab 3: Query ---
        query_frame = ttk.Frame(notebook, padding=8)
        self.query_tab = QueryTab(
            query_frame, self.root, self.service, self.tracker,
        )

        # --- Tab 4: Retrieve ---
        retrieve_frame = ttk.Frame(notebook, padding=8)
        self.retrieve_tab = RetrieveTab(
            retrieve_frame, self.root, self.service, self.tracker,
        )

        # Add tabs in display order
        notebook.add(ingest_frame, text="Ingest")
        notebook.add(query_frame, text="Query")
        notebook.add(retrieve_frame, text="Retrieve")
        notebook.add(chunks_frame, text="Chunks")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    root = tk.Tk()
    RAGPlaygroundApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
