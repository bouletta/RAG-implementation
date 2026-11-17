"""Small transaction boundary; only Agent 022 databases are accepted."""

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from shared.errors import AgentError


class Database:
    def __init__(self, path: Path) -> None:
        if str(path) == ":memory:":
            raise AgentError("Long-term memory requires an on-disk database.")
        self.path = path
        try:
            path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        except OSError as exc:
            raise AgentError("Cannot create the facts directory.") from exc
        with self.connect(write=True) as connection:
            version = connection.execute("PRAGMA user_version").fetchone()[0]
            app_id = connection.execute("PRAGMA application_id").fetchone()[0]
            if version == 1 and app_id == 22022:
                return
            tables = connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
            if version or app_id or tables:
                raise AgentError("Choose an empty database dedicated to Agent 022.")
            connection.execute(
                """CREATE TABLE facts (
                    profile TEXT NOT NULL,
                    key TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    PRIMARY KEY(profile, key)
                )"""
            )
            connection.execute("PRAGMA application_id = 22022")
            connection.execute("PRAGMA user_version = 1")

    @contextmanager
    def connect(self, *, write: bool = False) -> Iterator[sqlite3.Connection]:
        connection = None
        try:
            connection = sqlite3.connect(self.path, timeout=5, isolation_level=None)
            connection.execute("BEGIN IMMEDIATE" if write else "BEGIN")
            yield connection
            connection.commit()
        except sqlite3.Error as exc:
            raise AgentError("Facts database is unavailable, locked, or invalid.") from exc
        finally:
            if connection is not None:
                connection.close()
