from __future__ import annotations

import threading
import tkinter as tk
from tkinter import scrolledtext, ttk
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rag_shared import RAGService
    from shared import TokenTracker

from shared import run_coro


class QueryTab:
    """LLM query tab with mode selection and VLM toggle."""

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

        self.prompt_text = scrolledtext.ScrolledText(prompt_frame, height=5, wrap=tk.WORD)
        self.prompt_text.pack(fill=tk.BOTH, expand=True)

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

        self.query_vlm_enhanced = tk.BooleanVar(value=True)
        self.query_vlm_enhanced_toggle = ttk.Checkbutton(
            ctrl_frame,
            text="VLM enhanced",
            variable=self.query_vlm_enhanced,
        )
        self.query_vlm_enhanced_toggle.pack(side=tk.LEFT, padx=(0, 10))

        self.btn_query = ttk.Button(
            ctrl_frame, text="Send Query", command=self._on_query
        )
        self.btn_query.pack(side=tk.LEFT, padx=(0, 6))

        self.btn_clear_query = ttk.Button(
            ctrl_frame, text="Clear", command=self._on_clear
        )
        self.btn_clear_query.pack(side=tk.LEFT)

        self.query_spinner = ttk.Progressbar(ctrl_frame, mode="indeterminate", length=80)
        self.query_spinner.pack(side=tk.RIGHT)

        self.query_token_label = ttk.Label(parent, text="Tokens: —", anchor=tk.W)
        self.query_token_label.pack(fill=tk.X, pady=(0, 2))

        resp_frame = ttk.LabelFrame(parent, text="Response", padding=6)
        resp_frame.pack(fill=tk.BOTH, expand=True, pady=(4, 0))

        self.response_text = scrolledtext.ScrolledText(
            resp_frame, state=tk.DISABLED, height=12, wrap=tk.WORD
        )
        self.response_text.pack(fill=tk.BOTH, expand=True)

    # ---- Callbacks ---------------------------------------------------------
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

        self.tracker.reset("query")
        self.tracker.set_phase("query")
        mode = self.query_mode.get()
        vlm_enhanced = self.query_vlm_enhanced.get()
        threading.Thread(
            target=self._query_worker,
            args=(prompt, mode, vlm_enhanced),
            daemon=True,
        ).start()

    def _query_worker(self, prompt: str, mode: str, vlm_enhanced: bool):
        try:
            future = run_coro(
                self.service.query(
                    prompt,
                    mode=mode,
                    vlm_enhanced=vlm_enhanced,
                )
            )
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
        self.query_token_label.configure(
            text=self.tracker.summary("query")
        )

    def _on_clear(self):
        self.prompt_text.delete("1.0", tk.END)
        self.response_text.configure(state=tk.NORMAL)
        self.response_text.delete("1.0", tk.END)
        self.response_text.configure(state=tk.DISABLED)
        self.tracker.reset("query")
        self.query_token_label.configure(text="Tokens: —")
