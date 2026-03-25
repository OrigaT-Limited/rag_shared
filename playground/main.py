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
import json
import os
import sys
import threading
import tkinter as tk
from tkinter import filedialog, scrolledtext, ttk
from pathlib import Path
from typing import Any

from redis import Redis

# Ensure rag_shared is importable when running from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from rag_shared import RAGService, LLMAdapter, VLMAdapter, EmbeddingAdapter
from rag_shared.config import REDIS_URI


# ---------------------------------------------------------------------------
# Token usage tracker
# ---------------------------------------------------------------------------
class TokenTracker:
    """Accumulates token usage reported by adapters, split by phase."""

    def __init__(self):
        self._lock = threading.Lock()
        self._phase: str = "insert"  # "insert" or "query"
        # {phase: {adapter: {field: count}}}
        self._counts: dict[str, dict[str, dict[str, int]]] = {
            "insert": {},
            "query": {},
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
            workspace="playground",
        )
        self._pending: list[str] = []
        self._chunk_rows: list[dict[str, Any]] = []
        self._chunk_data_by_id: dict[str, dict[str, Any]] = {}

        self._build_ui()
        self.root.after(0, self._refresh_chunks)

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

        # --- Tab 3: Chunks ---
        chunks_tab = ttk.Frame(notebook, padding=8)
        notebook.add(chunks_tab, text="Chunks")
        self._build_chunks_tab(chunks_tab)

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

        # Token usage
        self.ingest_token_label = ttk.Label(parent, text="Tokens: —", anchor=tk.W)
        self.ingest_token_label.pack(fill=tk.X, pady=(0, 2))

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

        # --- Token usage ---
        self.query_token_label = ttk.Label(parent, text="Tokens: —", anchor=tk.W)
        self.query_token_label.pack(fill=tk.X, pady=(0, 2))

        # --- Response output ---
        resp_frame = ttk.LabelFrame(parent, text="Response", padding=6)
        resp_frame.pack(fill=tk.BOTH, expand=True, pady=(4, 0))

        self.response_text = scrolledtext.ScrolledText(
            resp_frame, state=tk.DISABLED, height=12, wrap=tk.WORD
        )
        self.response_text.pack(fill=tk.BOTH, expand=True)

    def _build_chunks_tab(self, parent: ttk.Frame):
        ctrl_frame = ttk.Frame(parent)
        ctrl_frame.pack(fill=tk.X, pady=(0, 6))

        self.btn_refresh_chunks = ttk.Button(
            ctrl_frame, text="Refresh", command=self._refresh_chunks
        )
        self.btn_refresh_chunks.pack(side=tk.LEFT, padx=(0, 6))

        ttk.Label(ctrl_frame, text="Filter:").pack(side=tk.LEFT, padx=(0, 4))
        self.chunk_filter_var = tk.StringVar()
        self.chunk_filter_entry = ttk.Entry(
            ctrl_frame, textvariable=self.chunk_filter_var, width=32
        )
        self.chunk_filter_entry.pack(side=tk.LEFT, padx=(0, 6))
        self.chunk_filter_entry.bind("<Return>", lambda _event: self._refresh_chunks())

        self.btn_clear_chunk_filter = ttk.Button(
            ctrl_frame, text="Clear", command=self._clear_chunk_filter
        )
        self.btn_clear_chunk_filter.pack(side=tk.LEFT)

        self.chunk_status_label = ttk.Label(ctrl_frame, text="Idle", anchor=tk.E)
        self.chunk_status_label.pack(side=tk.RIGHT)

        paned = ttk.PanedWindow(parent, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True)

        list_frame = ttk.LabelFrame(paned, text="Chunk list", padding=6)
        detail_frame = ttk.LabelFrame(paned, text="Chunk details", padding=6)
        paned.add(list_frame, weight=3)
        paned.add(detail_frame, weight=2)

        columns = ("file", "order", "page", "tokens", "preview")
        self.chunk_tree = ttk.Treeview(
            list_frame,
            columns=columns,
            show="headings",
            selectmode="browse",
        )
        self.chunk_tree.heading("file", text="File")
        self.chunk_tree.heading("order", text="Order")
        self.chunk_tree.heading("page", text="Page")
        self.chunk_tree.heading("tokens", text="Tokens")
        self.chunk_tree.heading("preview", text="Preview")
        self.chunk_tree.column("file", width=180, anchor=tk.W)
        self.chunk_tree.column("order", width=70, anchor=tk.CENTER)
        self.chunk_tree.column("page", width=70, anchor=tk.CENTER)
        self.chunk_tree.column("tokens", width=80, anchor=tk.CENTER)
        self.chunk_tree.column("preview", width=360, anchor=tk.W)
        self.chunk_tree.bind("<<TreeviewSelect>>", self._on_select_chunk)

        chunk_tree_scroll = ttk.Scrollbar(
            list_frame, orient=tk.VERTICAL, command=self.chunk_tree.yview
        )
        self.chunk_tree.configure(yscrollcommand=chunk_tree_scroll.set)
        self.chunk_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        chunk_tree_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        self.chunk_detail_text = scrolledtext.ScrolledText(
            detail_frame, state=tk.DISABLED, wrap=tk.WORD
        )
        self.chunk_detail_text.pack(fill=tk.BOTH, expand=True)

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
        self._refresh_chunks()

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

        self.tracker.reset("query")
        self.tracker.set_phase("query")
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
        self.query_token_label.configure(
            text=self.tracker.summary("query")
        )

    def _on_clear_query(self):
        self.prompt_text.delete("1.0", tk.END)
        self.response_text.configure(state=tk.NORMAL)
        self.response_text.delete("1.0", tk.END)
        self.response_text.configure(state=tk.DISABLED)
        self.tracker.reset("query")
        self.query_token_label.configure(text="Tokens: —")

    # ---- Chunk browser ----------------------------------------------------
    def _clear_chunk_filter(self):
        self.chunk_filter_var.set("")
        self._refresh_chunks()

    def _refresh_chunks(self):
        self.btn_refresh_chunks.configure(state=tk.DISABLED)
        self.chunk_status_label.configure(text="Loading chunks...")
        threading.Thread(target=self._load_chunks_worker, daemon=True).start()

    def _load_chunks_worker(self):
        try:
            rows = self._fetch_chunk_rows(self.chunk_filter_var.get().strip())
            self.root.after(0, self._populate_chunk_rows, rows)
        except Exception as exc:
            self.root.after(0, self._show_chunk_load_error, str(exc))

    def _fetch_chunk_rows(self, filter_text: str) -> list[dict[str, Any]]:
        client = Redis.from_url(REDIS_URI, decode_responses=True)
        namespace = self._get_chunk_namespace()
        pattern = f"{namespace}:*"
        normalized_filter = filter_text.lower()
        rows: list[dict[str, Any]] = []

        try:
            for key in client.scan_iter(match=pattern, count=200):
                payload = client.get(key)
                if not payload:
                    continue
                try:
                    data = json.loads(payload)
                except json.JSONDecodeError:
                    continue

                chunk_id = data.get("_id") or key.rsplit(":", 1)[-1]
                content = data.get("content", "")
                file_path = data.get("file_path", "")
                preview = " ".join(content.split())
                row = {
                    "chunk_id": chunk_id,
                    "file_path": file_path,
                    "file_name": Path(file_path).name if file_path else "—",
                    "chunk_order_index": data.get("chunk_order_index"),
                    "page_idx": data.get("page_idx"),
                    "tokens": data.get("tokens", 0),
                    "preview": preview[:140],
                    "data": data,
                }

                if normalized_filter:
                    haystack = "\n".join(
                        [
                            str(chunk_id),
                            str(file_path),
                            preview,
                            str(data.get("full_doc_id", "")),
                        ]
                    ).lower()
                    if normalized_filter not in haystack:
                        continue

                rows.append(row)
        finally:
            client.close()

        rows.sort(
            key=lambda row: (
                row["file_path"],
                self._sort_number(row["page_idx"]),
                self._sort_number(row["chunk_order_index"]),
                row["chunk_id"],
            )
        )
        return rows

    def _get_chunk_namespace(self) -> str:
        workspace = os.environ.get("REDIS_WORKSPACE", "").strip() or self.service.workspace
        if workspace:
            return f"{workspace}_text_chunks"
        return "text_chunks"

    @staticmethod
    def _sort_number(value: Any) -> int:
        return value if isinstance(value, int) else -1

    def _populate_chunk_rows(self, rows: list[dict[str, Any]]):
        self._chunk_rows = rows
        self._chunk_data_by_id = {row["chunk_id"]: row for row in rows}

        for item_id in self.chunk_tree.get_children():
            self.chunk_tree.delete(item_id)

        for row in rows:
            page_idx = row["page_idx"] if row["page_idx"] is not None else "—"
            chunk_order = (
                row["chunk_order_index"]
                if row["chunk_order_index"] is not None
                else "—"
            )
            self.chunk_tree.insert(
                "",
                tk.END,
                iid=row["chunk_id"],
                values=(
                    row["file_name"],
                    chunk_order,
                    page_idx,
                    row["tokens"],
                    row["preview"],
                ),
            )

        self._set_chunk_details("")
        self.chunk_status_label.configure(text=f"{len(rows)} chunk(s)")
        self.btn_refresh_chunks.configure(state=tk.NORMAL)

        if rows:
            first_id = rows[0]["chunk_id"]
            self.chunk_tree.selection_set(first_id)
            self.chunk_tree.focus(first_id)
            self._show_chunk_details(first_id)

    def _show_chunk_load_error(self, error_text: str):
        self._chunk_rows = []
        self._chunk_data_by_id = {}
        for item_id in self.chunk_tree.get_children():
            self.chunk_tree.delete(item_id)
        self._set_chunk_details(f"Failed to load chunks from Redis.\n\n{error_text}")
        self.chunk_status_label.configure(text="Load failed")
        self.btn_refresh_chunks.configure(state=tk.NORMAL)

    def _on_select_chunk(self, _event=None):
        selection = self.chunk_tree.selection()
        if not selection:
            return
        self._show_chunk_details(selection[0])

    def _show_chunk_details(self, chunk_id: str):
        row = self._chunk_data_by_id.get(chunk_id)
        if not row:
            self._set_chunk_details("")
            return

        data = row["data"]
        metadata_lines = [
            f"Chunk ID: {row['chunk_id']}",
            f"File: {data.get('file_path', '—')}",
            f"Full doc ID: {data.get('full_doc_id', '—')}",
            f"Page: {data.get('page_idx', '—')}",
            f"Order: {data.get('chunk_order_index', '—')}",
            f"Tokens: {data.get('tokens', '—')}",
            f"Type: {data.get('original_type', 'text')}",
            f"Multimodal: {data.get('is_multimodal', False)}",
            f"Created: {data.get('create_time', '—')}",
            f"Updated: {data.get('update_time', '—')}",
        ]
        extra_fields = {
            key: value
            for key, value in data.items()
            if key not in {
                "_id",
                "content",
                "file_path",
                "full_doc_id",
                "page_idx",
                "chunk_order_index",
                "tokens",
                "original_type",
                "is_multimodal",
                "create_time",
                "update_time",
            }
        }
        detail_text = "\n".join(metadata_lines)
        if extra_fields:
            detail_text += "\n\nExtra fields:\n"
            detail_text += json.dumps(extra_fields, indent=2, ensure_ascii=False)
        detail_text += "\n\nContent:\n"
        detail_text += data.get("content", "")
        self._set_chunk_details(detail_text)

    def _set_chunk_details(self, text: str):
        self.chunk_detail_text.configure(state=tk.NORMAL)
        self.chunk_detail_text.delete("1.0", tk.END)
        self.chunk_detail_text.insert(tk.END, text)
        self.chunk_detail_text.configure(state=tk.DISABLED)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    root = tk.Tk()
    RAGPlaygroundApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
