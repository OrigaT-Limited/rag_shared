from __future__ import annotations

import json
import threading
import tkinter as tk
from tkinter import scrolledtext, ttk
from typing import TYPE_CHECKING

from lightrag import QueryParam

if TYPE_CHECKING:
    from rag_shared import RAGService
    from shared import TokenTracker

from shared import run_coro


class RetrieveTab:
    """Raw retrieve tab — returns structured chunks/entities/relationships."""

    def __init__(
        self,
        parent: ttk.Frame,
        root: tk.Tk,
        service: RAGService,
        tracker: TokenTracker,
    ):
        self.root = root
        self.service = service
        self.tracker = tracker

        self._build(parent)

    # ---- UI ----------------------------------------------------------------
    def _build(self, parent: ttk.Frame):
        prompt_frame = ttk.LabelFrame(parent, text="Prompt", padding=6)
        prompt_frame.pack(fill=tk.BOTH, expand=False, pady=(0, 4))

        self.retrieve_prompt_text = scrolledtext.ScrolledText(
            prompt_frame, height=5, wrap=tk.WORD,
        )
        self.retrieve_prompt_text.pack(fill=tk.BOTH, expand=True)

        ctrl_frame = ttk.Frame(parent)
        ctrl_frame.pack(fill=tk.X, pady=4)

        ttk.Label(ctrl_frame, text="Mode:").pack(side=tk.LEFT, padx=(0, 4))
        self.retrieve_mode = tk.StringVar(value="hybrid")
        retrieve_mode_combo = ttk.Combobox(
            ctrl_frame,
            textvariable=self.retrieve_mode,
            values=["naive", "local", "global", "hybrid", "mix", "bypass"],
            state="readonly",
            width=10,
        )
        retrieve_mode_combo.pack(side=tk.LEFT, padx=(0, 10))

        self.btn_retrieve = ttk.Button(
            ctrl_frame, text="Run Retrieve", command=self._on_retrieve,
        )
        self.btn_retrieve.pack(side=tk.LEFT, padx=(0, 6))

        self.btn_clear_retrieve = ttk.Button(
            ctrl_frame, text="Clear", command=self._on_clear,
        )
        self.btn_clear_retrieve.pack(side=tk.LEFT)

        self.retrieve_spinner = ttk.Progressbar(
            ctrl_frame, mode="indeterminate", length=80,
        )
        self.retrieve_spinner.pack(side=tk.RIGHT)

        self.retrieve_token_label = ttk.Label(parent, text="Tokens: —", anchor=tk.W)
        self.retrieve_token_label.pack(fill=tk.X, pady=(0, 2))

        result_frame = ttk.LabelFrame(parent, text="Retrieve Result", padding=6)
        result_frame.pack(fill=tk.BOTH, expand=True, pady=(4, 0))

        self.retrieve_response_text = scrolledtext.ScrolledText(
            result_frame, state=tk.DISABLED, height=12, wrap=tk.WORD,
        )
        self.retrieve_response_text.pack(fill=tk.BOTH, expand=True)

    # ---- Callbacks ---------------------------------------------------------
    def _on_retrieve(self):
        prompt = self.retrieve_prompt_text.get("1.0", tk.END).strip()
        if not prompt:
            return
        self.btn_retrieve.configure(state=tk.DISABLED)
        self.retrieve_spinner.start(10)
        self.retrieve_response_text.configure(state=tk.NORMAL)
        self.retrieve_response_text.delete("1.0", tk.END)
        self.retrieve_response_text.insert(tk.END, "Retrieving…\n")
        self.retrieve_response_text.configure(state=tk.DISABLED)

        self.tracker.reset("retrieve")
        self.tracker.set_phase("retrieve")
        mode = self.retrieve_mode.get()
        threading.Thread(
            target=self._retrieve_worker,
            args=(prompt, mode),
            daemon=True,
        ).start()

    def _retrieve_worker(self, prompt: str, mode: str):
        try:
            future = run_coro(
                self.service.retrieve(
                    prompt,
                    param=QueryParam(mode=mode),
                )
            )
            result = future.result()
            self.root.after(0, self._show_response, result)
        except Exception as exc:
            self.root.after(0, self._show_error, str(exc))

    def _show_response(self, result):
        summary_lines = [
            f"Query: {result.query}",
            f"Mode: {result.mode}",
            f"Chunks: {len(result.chunks)}",
            f"Entities: {len(result.entities)}",
            f"Relationships: {len(result.relationships)}",
        ]
        payload = {
            "query": result.query,
            "mode": result.mode,
            "chunks": result.chunks,
            "entities": result.entities,
            "relationships": result.relationships,
        }
        sections = [
            "\n".join(summary_lines),
            "Formatted context:\n" + result.format_for_llm(),
            "Raw JSON:\n" + json.dumps(payload, indent=2, ensure_ascii=False),
        ]

        self.retrieve_response_text.configure(state=tk.NORMAL)
        self.retrieve_response_text.delete("1.0", tk.END)
        self.retrieve_response_text.insert(tk.END, "\n\n".join(sections))
        self.retrieve_response_text.configure(state=tk.DISABLED)
        self.retrieve_spinner.stop()
        self.btn_retrieve.configure(state=tk.NORMAL)
        self.retrieve_token_label.configure(text=self.tracker.summary("retrieve"))

    def _show_error(self, error_text: str):
        self.retrieve_response_text.configure(state=tk.NORMAL)
        self.retrieve_response_text.delete("1.0", tk.END)
        self.retrieve_response_text.insert(tk.END, f"Error: {error_text}")
        self.retrieve_response_text.configure(state=tk.DISABLED)
        self.retrieve_spinner.stop()
        self.btn_retrieve.configure(state=tk.NORMAL)
        self.retrieve_token_label.configure(text=self.tracker.summary("retrieve"))

    def _on_clear(self):
        self.retrieve_prompt_text.delete("1.0", tk.END)
        self.retrieve_response_text.configure(state=tk.NORMAL)
        self.retrieve_response_text.delete("1.0", tk.END)
        self.retrieve_response_text.configure(state=tk.DISABLED)
        self.tracker.reset("retrieve")
        self.retrieve_token_label.configure(text="Tokens: —")
