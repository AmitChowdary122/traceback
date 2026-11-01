"""SQLite store with provenance on every row (CONTRACTS.md v1.2 sections 2 and 4)."""
from __future__ import annotations

import hashlib
import re
import sqlite3
from typing import Any, Iterable

PROVENANCE = ("artifact_id", "source_path", "source_row", "ts_utc", "ts_original", "tz_basis", "raw_json")

# Table-specific columns, in contract order, with their SQLite types. These are the citable fields.
FAMILY: dict[str, dict[str, str]] = {
    "messages": {"app": "TEXT", "thread_id": "TEXT", "direction": "TEXT", "sender": "TEXT", "sender_number": "TEXT",
                 "recipient": "TEXT", "recipient_number": "TEXT", "body": "TEXT", "attachment_media_id": "TEXT"},
    "calls": {"app": "TEXT", "direction": "TEXT", "number": "TEXT", "contact_name": "TEXT", "duration_s": "REAL",
              "transcript": "TEXT"},
    "locations": {"source": "TEXT", "lat": "REAL", "lon": "REAL", "accuracy_m": "REAL"},
    "media": {"app": "TEXT", "kind": "TEXT", "file_path": "TEXT", "mime_type": "TEXT", "sha256": "TEXT",
              "size_bytes": "INTEGER", "lat": "REAL", "lon": "REAL"},
    "transcripts": {"media_id": "TEXT", "model": "TEXT", "language": "TEXT", "text": "TEXT", "duration_s": "REAL"},
    "image_tags": {"media_id": "TEXT", "model": "TEXT", "label": "TEXT", "score": "REAL", "box": "TEXT",
                   "prompt_set": "TEXT"},
}
PREFIX = {"messages": "msg", "calls": "call", "locations": "loc", "media": "med", "transcripts": "trn",
          "image_tags": "tag"}
TABLE_BY_PREFIX = {v: k for k, v in PREFIX.items()}
CITABLE: dict[str, set[str]] = {t: {"ts_utc", "ts_local", "source_path", *cols} for t, cols in FAMILY.items()}
CONTACT_COLUMNS = (("messages", "sender"), ("messages", "recipient"), ("calls", "contact_name"))

_ID_RE = re.compile(r"(msg|call|loc|med|trn|tag)_[0-9a-f]{10}")


def check_rel_path(path: str) -> str:
    """source_path must be a POSIX path relative to the report (or extraction) root."""
    if (not path or path.startswith("/") or "\\" in path or re.match(r"[A-Za-z]:", path)
            or ".." in path.split("/")):
        raise ValueError(f"source_path must be relative POSIX: {path!r}")
    return path


def artifact_id(table: str, source_path: str, row_key: str) -> str:
    check_rel_path(source_path)
    digest = hashlib.sha1(f"{source_path}|{row_key}".encode()).hexdigest()[:10]
    return f"{PREFIX[table]}_{digest}"


def _ddl(table: str) -> str:
    cols = ["artifact_id TEXT PRIMARY KEY", "source_path TEXT NOT NULL", "source_row TEXT NOT NULL", "ts_utc TEXT",
            "ts_original TEXT", "tz_basis TEXT NOT NULL", "raw_json TEXT NOT NULL"]
    cols += [f"{c} {t}" for c, t in FAMILY[table].items()]
    return f"CREATE TABLE IF NOT EXISTS {table} ({', '.join(cols)})"


class Store:
    def __init__(self, path: str):
        self.path = str(path)
        self.conn = sqlite3.connect(self.path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        with self.conn:
            for table in FAMILY:
                self.conn.execute(_ddl(table))
                self.conn.execute(f"CREATE INDEX IF NOT EXISTS ix_{table}_ts ON {table} (ts_utc, artifact_id)")
            self.conn.execute("CREATE TABLE IF NOT EXISTS case_meta (key TEXT PRIMARY KEY, value TEXT)")

    # -- writes ------------------------------------------------------------------------------
    def insert(self, table: str, row: dict) -> None:
        self.insert_many(table, [row])

    def insert_many(self, table: str, rows: Iterable[dict]) -> None:
        with self.conn:
            self._insert_rows(table, rows)

    def replace_rows(self, table: str, where: str, params: Iterable[Any], rows: Iterable[dict]) -> None:
        """Delete the rows matching `where`, then insert `rows`, in one transaction.

        Used for derived tables (transcripts, image_tags), so enriching twice gives the same rows."""
        if table not in FAMILY:
            raise ValueError(f"unknown table: {table!r}")
        with self.conn:
            self.conn.execute(f"DELETE FROM {table} WHERE {where}", list(params))
            self._insert_rows(table, rows)

    def _insert_rows(self, table: str, rows: Iterable[dict]) -> None:
        if table not in FAMILY:
            raise ValueError(f"unknown table: {table!r}")
        allowed = set(PROVENANCE) | set(FAMILY[table])
        cols = list(PROVENANCE) + list(FAMILY[table])
        sql = f"INSERT INTO {table} ({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))})"
        for row in rows:
            extra = set(row) - allowed
            if extra:
                raise ValueError(f"{table}: unknown columns {sorted(extra)}")
            for c in ("artifact_id", "source_path", "source_row", "tz_basis", "raw_json"):
                if row.get(c) in (None, ""):
                    raise ValueError(f"{table}: provenance column {c} is required")
            check_rel_path(row["source_path"])
            try:
                self.conn.execute(sql, [row.get(c) for c in cols])
            except sqlite3.IntegrityError as e:
                raise ValueError(f"artifact_id collision in {table}: {row['artifact_id']}") from e

    def set_meta(self, key: str, value: str) -> None:
        with self.conn:
            self.conn.execute("INSERT OR REPLACE INTO case_meta (key, value) VALUES (?, ?)", (key, value))

    # -- reads -------------------------------------------------------------------------------
    def meta(self) -> dict[str, str]:
        return {r["key"]: r["value"] for r in self.conn.execute("SELECT key, value FROM case_meta")}

    @property
    def case_tz(self) -> str:
        r = self.conn.execute("SELECT value FROM case_meta WHERE key = 'device_tz'").fetchone()
        return r["value"] if r and r["value"] else "UTC"

    def table_of(self, artifact_id: str) -> str | None:
        if not isinstance(artifact_id, str) or not _ID_RE.fullmatch(artifact_id):
            return None
        return TABLE_BY_PREFIX[artifact_id.split("_", 1)[0]]

    def get(self, artifact_id: str) -> dict | None:
        table = self.table_of(artifact_id)
        if table is None:
            return None
        r = self.conn.execute(f"SELECT * FROM {table} WHERE artifact_id = ?", (artifact_id,)).fetchone()
        if r is None:
            return None
        return {**dict(r), "table": table}

    def known_contacts(self) -> set[str]:
        out: set[str] = set()
        for table, col in CONTACT_COLUMNS:
            for (v,) in self.conn.execute(f"SELECT DISTINCT {col} FROM {table} WHERE {col} IS NOT NULL"):
                v = str(v).strip()
                if v:
                    out.add(v)
        return out

    def query(self, sql: str, params: Iterable[Any] = ()) -> list[dict]:
        return [dict(r) for r in self.conn.execute(sql, list(params))]

    def count(self, table: str) -> int:
        return self.conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]

    def close(self) -> None:
        self.conn.close()
