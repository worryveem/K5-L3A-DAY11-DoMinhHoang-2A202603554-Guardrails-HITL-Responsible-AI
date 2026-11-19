"""
Assignment 11 — Audit log with request correlation and elapsed time.

Records every interaction for forensics. Never blocks by itself —
other layers catch attacks; this layer makes them reviewable.
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path


def default_audit_log_path() -> str:
    """Always resolve to <repo>/outputs/… (safe when cwd is src/)."""
    repo_root = Path(__file__).resolve().parents[2]
    return str(repo_root / "outputs" / "audit_log.json")


class AuditLogPlugin:
    """Framework-agnostic audit logger (wire into ADK callbacks or your pipeline)."""

    def __init__(self):
        self.name = "audit_log"
        self.logs: list[dict] = []
        self._open: dict[str, dict] = {}

    def record_input(self, *, user_id: str, text: str, request_id: str | None = None):
        """Start a request; explicit IDs support concurrent requests by one user."""
        key = request_id or user_id
        if key in self._open:
            raise ValueError("An audit request with this ID is already open")
        self._open[key] = {
            "request_id": key,
            "user_id": user_id,
            "input": text,
            "started_at": utc_now_iso(),
            "_started_monotonic": time.monotonic(),
        }
        return key

    def record_output(
        self,
        *,
        user_id: str,
        text: str,
        blocked: bool = False,
        layer: str | None = None,
        request_id: str | None = None,
    ):
        """Finish a matching request, including guardrail and provider failures."""
        key = request_id or user_id
        entry = self._open.get(key)
        if entry is None or entry["user_id"] != user_id:
            raise ValueError("No matching audit input for this request and user")
        self._open.pop(key)
        started = entry.pop("_started_monotonic")
        entry.update({
            "output": text,
            "blocked": blocked,
            "layer": layer,
            "completed_at": utc_now_iso(),
            "latency_ms": round(max(0.0, time.monotonic() - started) * 1000, 3),
        })
        self.logs.append(entry)
        return entry

    def export_json(self, filepath: str | None = None):
        """Write logs to disk (JSON array) under repo-root ``outputs/`` by default."""
        path = Path(filepath or default_audit_log_path())
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.logs, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return str(path)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
