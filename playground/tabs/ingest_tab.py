from __future__ import annotations

import os
import threading
import tkinter as tk
from tkinter import filedialog, scrolledtext, ttk
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from rag_shared import RAGService
    from shared import TokenTracker

from shared import run_coro


class IngestTab:
    """File selection + RAG ingestion tab."""

    def __init__(
        self,
        parent: ttk.Frame,
        root: tk.Tk,
        service: RAGService,
        tracker: TokenTracker,
        *,
        on_ingest_done: Callable[[], None] | None = None,
    ):
        self.root = root
        self.service = service
        self.tracker = tracker
        self._on_ingest_done = on_ingest_done
        self._pending: list[str] = []

        self._build(parent)

    # ---- UI ----------------------------------------------------------------
    def _build(self, parent: ttk.Frame):
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

        list_frame = ttk.LabelFrame(parent, text="Selected files", padding=6)
        list_frame.pack(fill=tk.BOTH, expand=False, pady=(8, 4))

        self.file_listbox = tk.Listbox(list_frame, height=6, selectmode=tk.EXTENDED)
        self.file_listbox.pack(fill=tk.BOTH, expand=True)

        log_frame = ttk.LabelFrame(parent, text="Log", padding=6)
        log_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 4))

        self.log = scrolledtext.ScrolledText(log_frame, state=tk.DISABLED, height=10)
        self.log.pack(fill=tk.BOTH, expand=True)

        self.progress = ttk.Progressbar(parent, mode="determinate")
        self.progress.pack(fill=tk.X, pady=(0, 4))

        self.ingest_token_label = ttk.Label(parent, text="Tokens: —", anchor=tk.W)
        self.ingest_token_label.pack(fill=tk.X, pady=(0, 2))

    # ---- Logging -----------------------------------------------------------
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
        self.tracker.reset("insert")
        self.ingest_token_label.configure(text="Tokens: —")
        self._log("Cleared file list.")

    def _on_insert(self):
        if not self._pending:
            return
        self.btn_insert.configure(state=tk.DISABLED)
        self.btn_select.configure(state=tk.DISABLED)
        files = list(self._pending)
        self.progress["maximum"] = len(files)
        self.progress["value"] = 0
        self.tracker.reset("insert")
        self.tracker.set_phase("insert")
        self._log(f"Starting ingestion of {len(files)} file(s)…")

        threading.Thread(target=self._ingest_worker, args=(files,), daemon=True).start()

    # ---- Background worker -------------------------------------------------
    def _ingest_worker(self, files: list[str]):
        for i, fpath in enumerate(files, 1):
            name = os.path.basename(fpath)
            self.root.after(0, self._log, f"[{i}/{len(files)}] Ingesting {name}…")
            try:
                future = run_coro(
                    self.service.insert_document(fpath, display_stats=True)
                )
                future.result()
                self.root.after(0, self._log, f"  ✓ {name} done.")
            except Exception as exc:
                self.root.after(0, self._log, f"  ✗ {name} failed: {exc}")
            self.root.after(0, self._step_progress)

        self.root.after(0, self._ingest_done)

    def _step_progress(self):
        self.progress["value"] = self.progress["value"] + 1
        self.ingest_token_label.configure(
            text=self.tracker.summary("insert")
        )

    def _ingest_done(self):
        self._log("Ingestion complete.")
        self.ingest_token_label.configure(
            text=self.tracker.summary("insert")
        )
        self._pending.clear()
        self.file_listbox.delete(0, tk.END)
        self.btn_select.configure(state=tk.NORMAL)
        self.progress["value"] = 0
        if self._on_ingest_done:
            self._on_ingest_done()
