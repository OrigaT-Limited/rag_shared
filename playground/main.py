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

import asyncio
import sys
import os
import threading
import tkinter as tk
from tkinter import filedialog, scrolledtext, ttk
from pathlib import Path

# Ensure rag_shared is importable when running from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from rag_shared import RAGService


# ---------------------------------------------------------------------------
# Stub LLM — caller normally provides this; playground uses a placeholder
# ---------------------------------------------------------------------------
async def _stub_llm(
    prompt: str,
    system_prompt: str | None = None,
    history_messages: list | None = None,
    **kwargs,
) -> str:
    """Minimal LLM stub so RAGService can initialise.

    In production the caller (AI-ModelLayer) injects a real LLM function.
    For ingestion-only use the LLM is called by LightRAG to extract KG
    entities/relations.  Replace this stub with a real model call if you
    need high-quality extraction.
    """
    return ""


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


def _run_coro(coro):
    """Schedule *coro* on the background event loop, return a Future."""
    return asyncio.run_coroutine_threadsafe(coro, _get_loop())


# ---------------------------------------------------------------------------
# GUI Application
# ---------------------------------------------------------------------------
class RAGIngestApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        root.title("RAG File Ingestion — Playground")
        root.geometry("720x520")
        root.minsize(600, 400)

        self.service = RAGService(llm_model_func=_stub_llm, workspace="playground")
        self._pending: list[str] = []

        self._build_ui()

    # ---- UI construction ---------------------------------------------------
    def _build_ui(self):
        # Top frame — buttons
        btn_frame = ttk.Frame(self.root, padding=8)
        btn_frame.pack(fill=tk.X)

        self.btn_select = ttk.Button(
            btn_frame, text="Select Files…", command=self._on_select_files
        )
        self.btn_select.pack(side=tk.LEFT, padx=(0, 6))

        self.btn_insert = ttk.Button(
            btn_frame, text="Insert into RAG", command=self._on_insert, state=tk.DISABLED
        )
        self.btn_insert.pack(side=tk.LEFT, padx=(0, 6))

        self.btn_clear = ttk.Button(
            btn_frame, text="Clear", command=self._on_clear
        )
        self.btn_clear.pack(side=tk.LEFT)

        # File list
        list_frame = ttk.LabelFrame(self.root, text="Selected files", padding=6)
        list_frame.pack(fill=tk.BOTH, expand=False, padx=8, pady=(0, 4))

        self.file_listbox = tk.Listbox(list_frame, height=6, selectmode=tk.EXTENDED)
        self.file_listbox.pack(fill=tk.BOTH, expand=True)

        # Log area
        log_frame = ttk.LabelFrame(self.root, text="Log", padding=6)
        log_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 8))

        self.log = scrolledtext.ScrolledText(log_frame, state=tk.DISABLED, height=10)
        self.log.pack(fill=tk.BOTH, expand=True)

        # Progress bar
        self.progress = ttk.Progressbar(self.root, mode="determinate")
        self.progress.pack(fill=tk.X, padx=8, pady=(0, 8))

    # ---- Logging helper ----------------------------------------------------
    def _log(self, msg: str):
        self.log.configure(state=tk.NORMAL)
        self.log.insert(tk.END, msg + "\n")
        self.log.see(tk.END)
        self.log.configure(state=tk.DISABLED)

    # ---- Callbacks ---------------------------------------------------------
    def _on_select_files(self):
        paths = filedialog.askopenfilenames(
            title="Choose documents to ingest",
            filetypes=[
                ("Documents", "*.pdf *.docx *.doc *.pptx *.txt *.md *.html *.htm"),
                ("PDF", "*.pdf"),
                ("Word", "*.docx *.doc"),
                ("PowerPoint", "*.pptx"),
                ("Text / Markdown", "*.txt *.md"),
                ("HTML", "*.html *.htm"),
                ("All files", "*.*"),
            ],
        )
        if not paths:
            return
        for p in paths:
            if p not in self._pending:
                self._pending.append(p)
                self.file_listbox.insert(tk.END, p)
        self.btn_insert.configure(state=tk.NORMAL)
        self._log(f"Added {len(paths)} file(s).")

    def _on_clear(self):
        self._pending.clear()
        self.file_listbox.delete(0, tk.END)
        self.btn_insert.configure(state=tk.DISABLED)
        self.progress["value"] = 0
        self._log("Cleared file list.")

    def _on_insert(self):
        if not self._pending:
            return
        self.btn_insert.configure(state=tk.DISABLED)
        self.btn_select.configure(state=tk.DISABLED)
        files = list(self._pending)
        self.progress["maximum"] = len(files)
        self.progress["value"] = 0
        self._log(f"Starting ingestion of {len(files)} file(s)…")

        # Run ingestion in background so the GUI stays responsive
        threading.Thread(target=self._ingest_worker, args=(files,), daemon=True).start()

    # ---- Background worker -------------------------------------------------
    def _ingest_worker(self, files: list[str]):
        for i, fpath in enumerate(files, 1):
            name = os.path.basename(fpath)
            self.root.after(0, self._log, f"[{i}/{len(files)}] Ingesting {name}…")
            try:
                future = _run_coro(
                    self.service.insert_document(fpath, display_stats=True)
                )
                future.result()  # block until done
                self.root.after(0, self._log, f"  ✓ {name} done.")
            except Exception as exc:
                self.root.after(0, self._log, f"  ✗ {name} failed: {exc}")
            self.root.after(0, self._step_progress)

        self.root.after(0, self._ingest_done)

    def _step_progress(self):
        self.progress["value"] = self.progress["value"] + 1

    def _ingest_done(self):
        self._log("Ingestion complete.")
        self._pending.clear()
        self.file_listbox.delete(0, tk.END)
        self.btn_select.configure(state=tk.NORMAL)
        self.progress["value"] = 0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    root = tk.Tk()
    RAGIngestApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
