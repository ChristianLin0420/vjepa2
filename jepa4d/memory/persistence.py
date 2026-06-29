"""SQLite persistence for structured Phase 0 records."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path


class MemoryPersistence:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.path) as connection:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS records (kind TEXT, record_id TEXT PRIMARY KEY, payload TEXT)"
            )

    def upsert(self, kind: str, record_id: str, payload: dict) -> None:
        with sqlite3.connect(self.path) as connection:
            connection.execute(
                "INSERT INTO records VALUES (?, ?, ?) ON CONFLICT(record_id) DO UPDATE SET kind=excluded.kind,payload=excluded.payload",
                (kind, record_id, json.dumps(payload)),
            )

    def get(self, record_id: str) -> dict | None:
        with sqlite3.connect(self.path) as connection:
            row = connection.execute("SELECT payload FROM records WHERE record_id=?", (record_id,)).fetchone()
        return None if row is None else json.loads(row[0])

    def list_kind(self, kind: str) -> list[dict]:
        with sqlite3.connect(self.path) as connection:
            rows = connection.execute(
                "SELECT payload FROM records WHERE kind=? ORDER BY record_id", (kind,)
            ).fetchall()
        return [json.loads(row[0]) for row in rows]
