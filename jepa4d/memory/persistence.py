"""Transactional SQLite storage for records, event sourcing, and snapshots."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable
from pathlib import Path
from typing import Any


class MemoryPersistence:
    schema_version = 2

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=NORMAL")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS records (
                    kind TEXT NOT NULL,
                    record_id TEXT PRIMARY KEY,
                    payload TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS records_kind_idx ON records(kind);
                CREATE TABLE IF NOT EXISTS event_log (
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    operation TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    record_id TEXT NOT NULL,
                    payload TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS event_timestamp_idx ON event_log(timestamp);
                CREATE TABLE IF NOT EXISTS snapshots (
                    revision INTEGER PRIMARY KEY,
                    timestamp REAL NOT NULL,
                    payload TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
                """
            )
            connection.execute(
                "INSERT INTO metadata(key,value) VALUES('schema_version',?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (str(self.schema_version),),
            )

    def upsert(self, kind: str, record_id: str, payload: dict[str, Any]) -> None:
        self.apply_batch([(kind, record_id, payload)], timestamp=float(payload.get("last_seen_time", 0.0)))

    def apply_batch(
        self,
        records: Iterable[tuple[str, str, dict[str, Any]]],
        *,
        timestamp: float,
        operation: str = "upsert",
    ) -> int:
        return self.commit_update(records, timestamp=timestamp, operation=operation)

    def commit_update(
        self,
        records: Iterable[tuple[str, str, dict[str, Any]]],
        *,
        timestamp: float,
        operation: str = "upsert",
        revision: int | None = None,
        snapshot_payload: dict[str, Any] | None = None,
    ) -> int:
        """Commit records, their event-log entries, and an optional snapshot atomically."""
        values = list(records)
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            for kind, record_id, payload in values:
                encoded = json.dumps(payload, separators=(",", ":"), allow_nan=False)
                connection.execute(
                    "INSERT INTO records(kind,record_id,payload) VALUES(?,?,?) "
                    "ON CONFLICT(record_id) DO UPDATE SET kind=excluded.kind,payload=excluded.payload",
                    (kind, record_id, encoded),
                )
                connection.execute(
                    "INSERT INTO event_log(timestamp,operation,kind,record_id,payload) VALUES(?,?,?,?,?)",
                    (timestamp, operation, kind, record_id, encoded),
                )
            if revision is not None and snapshot_payload is not None:
                encoded_snapshot = json.dumps(snapshot_payload, separators=(",", ":"), allow_nan=False)
                connection.execute(
                    "INSERT INTO snapshots(revision,timestamp,payload) VALUES(?,?,?) "
                    "ON CONFLICT(revision) DO UPDATE SET timestamp=excluded.timestamp,payload=excluded.payload",
                    (revision, timestamp, encoded_snapshot),
                )
        return len(values)

    def append_event(
        self, timestamp: float, operation: str, kind: str, record_id: str, payload: dict[str, Any]
    ) -> int:
        encoded = json.dumps(payload, separators=(",", ":"), allow_nan=False)
        with self._connect() as connection:
            cursor = connection.execute(
                "INSERT INTO event_log(timestamp,operation,kind,record_id,payload) VALUES(?,?,?,?,?)",
                (timestamp, operation, kind, record_id, encoded),
            )
            if cursor.lastrowid is None:
                raise RuntimeError("SQLite did not return an event sequence")
            return int(cursor.lastrowid)

    def get(self, record_id: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute("SELECT payload FROM records WHERE record_id=?", (record_id,)).fetchone()
        return None if row is None else json.loads(row["payload"])

    def list_kind(self, kind: str) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT payload FROM records WHERE kind=? ORDER BY record_id", (kind,)
            ).fetchall()
        return [json.loads(row["payload"]) for row in rows]

    def list_events(
        self, *, since_sequence: int = 0, kind: str | None = None, limit: int | None = None
    ) -> list[dict[str, Any]]:
        query = "SELECT sequence,timestamp,operation,kind,record_id,payload FROM event_log WHERE sequence>?"
        parameters: list[Any] = [since_sequence]
        if kind is not None:
            query += " AND kind=?"
            parameters.append(kind)
        query += " ORDER BY sequence"
        if limit is not None:
            query += " LIMIT ?"
            parameters.append(limit)
        with self._connect() as connection:
            rows = connection.execute(query, parameters).fetchall()
        return [
            {
                "sequence": row["sequence"],
                "timestamp": row["timestamp"],
                "operation": row["operation"],
                "kind": row["kind"],
                "record_id": row["record_id"],
                "payload": json.loads(row["payload"]),
            }
            for row in rows
        ]

    def save_snapshot(self, revision: int, timestamp: float, payload: dict[str, Any]) -> None:
        encoded = json.dumps(payload, separators=(",", ":"), allow_nan=False)
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO snapshots(revision,timestamp,payload) VALUES(?,?,?) "
                "ON CONFLICT(revision) DO UPDATE SET timestamp=excluded.timestamp,payload=excluded.payload",
                (revision, timestamp, encoded),
            )

    def load_latest_snapshot(self) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT revision,timestamp,payload FROM snapshots ORDER BY revision DESC LIMIT 1"
            ).fetchone()
        if row is None:
            return None
        return {"revision": row["revision"], "timestamp": row["timestamp"], "payload": json.loads(row["payload"])}

    def stats(self) -> dict[str, int]:
        with self._connect() as connection:
            records = int(connection.execute("SELECT COUNT(*) FROM records").fetchone()[0])
            events = int(connection.execute("SELECT COUNT(*) FROM event_log").fetchone()[0])
            snapshots = int(connection.execute("SELECT COUNT(*) FROM snapshots").fetchone()[0])
        return {"records": records, "events": events, "snapshots": snapshots, "schema_version": self.schema_version}
