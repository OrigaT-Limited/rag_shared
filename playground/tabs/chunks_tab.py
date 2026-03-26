from __future__ import annotations

import json
import os
import threading
import tkinter as tk
from tkinter import messagebox, scrolledtext, ttk
from pathlib import Path
from typing import TYPE_CHECKING, Any

from redis import Redis

if TYPE_CHECKING:
    from rag_shared import RAGService
    from shared import TokenTracker

from rag_shared.config import REDIS_URI


class ChunksTab:
    """Redis chunk browser with filter, detail view, and deletion."""

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
        self._chunk_rows: list[dict[str, Any]] = []
        self._chunk_data_by_id: dict[str, dict[str, Any]] = {}

        self._build(parent)

    # ---- UI ----------------------------------------------------------------
    def _build(self, parent: ttk.Frame):
        ctrl_frame = ttk.Frame(parent)
        ctrl_frame.pack(fill=tk.X, pady=(0, 6))

        self.btn_refresh_chunks = ttk.Button(
            ctrl_frame, text="Refresh", command=self.refresh
        )
        self.btn_refresh_chunks.pack(side=tk.LEFT, padx=(0, 6))

        ttk.Label(ctrl_frame, text="Filter:").pack(side=tk.LEFT, padx=(0, 4))
        self.chunk_filter_var = tk.StringVar()
        self.chunk_filter_entry = ttk.Entry(
            ctrl_frame, textvariable=self.chunk_filter_var, width=32
        )
        self.chunk_filter_entry.pack(side=tk.LEFT, padx=(0, 6))
        self.chunk_filter_entry.bind("<Return>", lambda _event: self.refresh())

        self.btn_clear_chunk_filter = ttk.Button(
            ctrl_frame, text="Clear", command=self._clear_filter
        )
        self.btn_clear_chunk_filter.pack(side=tk.LEFT)

        self.btn_delete_chunk = ttk.Button(
            ctrl_frame,
            text="Delete selected",
            command=self._on_delete_chunks,
            state=tk.DISABLED,
        )
        self.btn_delete_chunk.pack(side=tk.LEFT, padx=(6, 0))

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
            selectmode="extended",
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
        self.chunk_tree.bind("<Delete>", self._on_delete_chunks)
        self.chunk_tree.bind("<BackSpace>", self._on_delete_chunks)

        self.chunk_detail_text = scrolledtext.ScrolledText(
            detail_frame, state=tk.DISABLED, wrap=tk.WORD
        )
        self.chunk_detail_text.pack(fill=tk.BOTH, expand=True)

    # ---- Public ------------------------------------------------------------
    def refresh(self):
        """Reload chunks from Redis (can be called externally)."""
        self.btn_refresh_chunks.configure(state=tk.DISABLED)
        self.chunk_status_label.configure(text="Loading chunks...")
        threading.Thread(target=self._load_chunks_worker, daemon=True).start()

    # ---- Internal ----------------------------------------------------------
    def _clear_filter(self):
        self.chunk_filter_var.set("")
        self.refresh()

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
                    "redis_key": key,
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
        self._update_chunk_action_state()

        if rows:
            first_id = rows[0]["chunk_id"]
            self.chunk_tree.selection_set(first_id)
            self.chunk_tree.focus(first_id)
            self._show_chunk_details(first_id)
            self._update_chunk_action_state()

    def _show_chunk_load_error(self, error_text: str):
        self._chunk_rows = []
        self._chunk_data_by_id = {}
        for item_id in self.chunk_tree.get_children():
            self.chunk_tree.delete(item_id)
        self._set_chunk_details(f"Failed to load chunks from Redis.\n\n{error_text}")
        self.chunk_status_label.configure(text="Load failed")
        self.btn_refresh_chunks.configure(state=tk.NORMAL)
        self._update_chunk_action_state()

    def _on_select_chunk(self, _event=None):
        selection = self.chunk_tree.selection()
        self._update_chunk_action_state()
        if not selection:
            return
        self._show_chunk_details(selection[0])

    def _update_chunk_action_state(self):
        state = tk.NORMAL if self.chunk_tree.selection() else tk.DISABLED
        self.btn_delete_chunk.configure(state=state)

    def _on_delete_chunks(self, _event=None):
        selection = list(self.chunk_tree.selection())
        if not selection:
            return

        selected_rows = [
            self._chunk_data_by_id[item_id]
            for item_id in selection
            if item_id in self._chunk_data_by_id
        ]
        if not selected_rows:
            return

        if len(selected_rows) == 1:
            row = selected_rows[0]
            prompt = (
                "Delete the selected chunk from Redis?\n\n"
                f"Chunk ID: {row['chunk_id']}\n"
                f"File: {row['file_path'] or '—'}"
            )
        else:
            prompt = (
                f"Delete {len(selected_rows)} selected chunks from Redis?\n\n"
                "This only removes the Redis chunk records shown in this tab."
            )

        if not messagebox.askyesno("Delete chunk", prompt, parent=self.root):
            return

        self.btn_delete_chunk.configure(state=tk.DISABLED)
        self.btn_refresh_chunks.configure(state=tk.DISABLED)
        self.chunk_status_label.configure(text="Deleting chunks...")
        threading.Thread(
            target=self._delete_chunks_worker,
            args=(selected_rows,),
            daemon=True,
        ).start()

    def _delete_chunks_worker(self, rows: list[dict[str, Any]]):
        deleted_count = 0
        errors: list[str] = []
        client = Redis.from_url(REDIS_URI, decode_responses=True)

        try:
            redis_keys = [
                row["redis_key"]
                for row in rows
                if isinstance(row.get("redis_key"), str) and row["redis_key"]
            ]
            if redis_keys:
                deleted = client.delete(*redis_keys)
                deleted_count = deleted if isinstance(deleted, int) else 0
        except Exception as exc:
            errors.append(str(exc))
        finally:
            client.close()

        self.root.after(
            0,
            self._finish_chunk_deletion,
            [row["chunk_id"] for row in rows],
            deleted_count,
            errors,
        )

    def _finish_chunk_deletion(
        self,
        chunk_ids: list[str],
        deleted_count: int,
        errors: list[str],
    ):
        if errors:
            self.chunk_status_label.configure(text="Delete failed")
            self.btn_refresh_chunks.configure(state=tk.NORMAL)
            self._update_chunk_action_state()
            messagebox.showerror(
                "Delete chunk",
                "Failed to delete chunk(s) from Redis.\n\n" + "\n".join(errors),
                parent=self.root,
            )
            return

        deleted_ids = set(chunk_ids)
        self._chunk_rows = [
            row for row in self._chunk_rows if row["chunk_id"] not in deleted_ids
        ]
        self._chunk_data_by_id = {
            row["chunk_id"]: row for row in self._chunk_rows
        }

        for chunk_id in chunk_ids:
            if self.chunk_tree.exists(chunk_id):
                self.chunk_tree.delete(chunk_id)

        if self._chunk_rows:
            first_id = self._chunk_rows[0]["chunk_id"]
            self.chunk_tree.selection_set(first_id)
            self.chunk_tree.focus(first_id)
            self._show_chunk_details(first_id)
        else:
            self._set_chunk_details("")

        self.chunk_status_label.configure(
            text=f"Deleted {deleted_count} chunk(s); {len(self._chunk_rows)} remaining"
        )
        self.btn_refresh_chunks.configure(state=tk.NORMAL)
        self._update_chunk_action_state()

        if deleted_count != len(chunk_ids):
            messagebox.showwarning(
                "Delete chunk",
                "Some selected chunks were not found in Redis. The list has been refreshed locally.",
                parent=self.root,
            )

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
