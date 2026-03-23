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
class RAGPlaygroundApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        root.title("RAG Playground")
        root.geometry("800x620")
        root.minsize(700, 500)

        self.service = RAGService(workspace="playground")
        self._pending: list[str] = []

        self._build_ui()

    # ---- UI construction ---------------------------------------------------
    def _build_ui(self):
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        # --- Tab 1: Ingest ---
        ingest_tab = ttk.Frame(notebook, padding=8)
        notebook.add(ingest_tab, text="Ingest")
        self._build_ingest_tab(ingest_tab)

        # --- Tab 2: Query ---
        query_tab = ttk.Frame(notebook, padding=8)
        notebook.add(query_tab, text="Query")
        self._build_query_tab(query_tab)

    def _build_ingest_tab(self, parent: ttk.Frame):
        # Top frame — buttons
        btn_frame = ttk.Frame(parent)
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
        list_frame = ttk.LabelFrame(parent, text="Selected files", padding=6)
        list_frame.pack(fill=tk.BOTH, expand=False, pady=(8, 4))

        self.file_listbox = tk.Listbox(list_frame, height=6, selectmode=tk.EXTENDED)
        self.file_listbox.pack(fill=tk.BOTH, expand=True)

        # Log area
        log_frame = ttk.LabelFrame(parent, text="Log", padding=6)
        log_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 4))

        self.log = scrolledtext.ScrolledText(log_frame, state=tk.DISABLED, height=10)
        self.log.pack(fill=tk.BOTH, expand=True)

        # Progress bar
        self.progress = ttk.Progressbar(parent, mode="determinate")
        self.progress.pack(fill=tk.X, pady=(0, 4))

    def _build_query_tab(self, parent: ttk.Frame):
        # --- Prompt input ---
        prompt_frame = ttk.LabelFrame(parent, text="Prompt", padding=6)
        prompt_frame.pack(fill=tk.BOTH, expand=False, pady=(0, 4))

        self.prompt_text = scrolledtext.ScrolledText(prompt_frame, height=5, wrap=tk.WORD)
        self.prompt_text.pack(fill=tk.BOTH, expand=True)

        # --- Controls ---
        ctrl_frame = ttk.Frame(parent)
        ctrl_frame.pack(fill=tk.X, pady=4)

        ttk.Label(ctrl_frame, text="Mode:").pack(side=tk.LEFT, padx=(0, 4))
        self.query_mode = tk.StringVar(value="hybrid")
        mode_combo = ttk.Combobox(
            ctrl_frame,
            textvariable=self.query_mode,
            values=["naive", "local", "global", "hybrid", "mix"],
            state="readonly",
            width=10,
        )
        mode_combo.pack(side=tk.LEFT, padx=(0, 10))

        self.btn_query = ttk.Button(
            ctrl_frame, text="Send Query", command=self._on_query
        )
        self.btn_query.pack(side=tk.LEFT, padx=(0, 6))

        self.btn_clear_query = ttk.Button(
            ctrl_frame, text="Clear", command=self._on_clear_query
        )
        self.btn_clear_query.pack(side=tk.LEFT)

        self.query_spinner = ttk.Progressbar(ctrl_frame, mode="indeterminate", length=80)
        self.query_spinner.pack(side=tk.RIGHT)

        # --- Response output ---
        resp_frame = ttk.LabelFrame(parent, text="Response", padding=6)
        resp_frame.pack(fill=tk.BOTH, expand=True, pady=(4, 0))

        self.response_text = scrolledtext.ScrolledText(
            resp_frame, state=tk.DISABLED, height=12, wrap=tk.WORD
        )
        self.response_text.pack(fill=tk.BOTH, expand=True)

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

    # ---- Query callbacks ---------------------------------------------------
    def _on_query(self):
        prompt = self.prompt_text.get("1.0", tk.END).strip()
        if not prompt:
            return
        self.btn_query.configure(state=tk.DISABLED)
        self.query_spinner.start(10)
        self.response_text.configure(state=tk.NORMAL)
        self.response_text.delete("1.0", tk.END)
        self.response_text.insert(tk.END, "Querying…\n")
        self.response_text.configure(state=tk.DISABLED)

        mode = self.query_mode.get()
        threading.Thread(
            target=self._query_worker, args=(prompt, mode), daemon=True
        ).start()

    def _query_worker(self, prompt: str, mode: str):
        try:
            future = _run_coro(self.service.query(prompt, mode=mode))
            result = future.result()
            self.root.after(0, self._show_response, result)
        except Exception as exc:
            self.root.after(0, self._show_response, f"Error: {exc}")

    def _show_response(self, text: str):
        self.response_text.configure(state=tk.NORMAL)
        self.response_text.delete("1.0", tk.END)
        self.response_text.insert(tk.END, text)
        self.response_text.configure(state=tk.DISABLED)
        self.query_spinner.stop()
        self.btn_query.configure(state=tk.NORMAL)

    def _on_clear_query(self):
        self.prompt_text.delete("1.0", tk.END)
        self.response_text.configure(state=tk.NORMAL)
        self.response_text.delete("1.0", tk.END)
        self.response_text.configure(state=tk.DISABLED)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    root = tk.Tk()
    RAGPlaygroundApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
