"""Versioned application databases with explicit transaction ownership."""

import sqlite3
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path

from shared.errors import AgentError


class Database:
    def __init__(self, path: Path, app_id: int, schema: Sequence[str]) -> None:
        if str(path) == ":memory:":
            raise AgentError("Choose a persistent database path.")
        self.path = path
        try:
            path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        except OSError as exc:
            raise AgentError("Cannot create the database directory.") from exc
        with self.connect(write=True) as conn:
            version = conn.execute("PRAGMA user_version").fetchone()[0]
            identity = conn.execute("PRAGMA application_id").fetchone()[0]
            if version == 1 and identity == app_id:
                return
            tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
            if version or identity or tables:
                raise AgentError("Database belongs to another application or schema version.")
            for statement in schema:
                conn.execute(statement)
            conn.execute(f"PRAGMA application_id = {int(app_id)}")
            conn.execute("PRAGMA user_version = 1")

    @contextmanager
    def connect(self, *, write: bool = False) -> Iterator[sqlite3.Connection]:
        conn = None
        try:
            conn = sqlite3.connect(self.path, timeout=5, isolation_level=None)
            conn.row_factory = sqlite3.Row
            conn.execute("BEGIN IMMEDIATE" if write else "BEGIN")
            yield conn
            conn.commit()
        except sqlite3.Error as exc:
            raise AgentError("Database is unavailable, locked, or invalid.") from exc
        finally:
            if conn is not None:
                conn.close()
