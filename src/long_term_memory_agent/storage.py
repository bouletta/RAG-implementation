"""Explicit writes with compare-and-swap revisions and expiry-aware reads."""

import logging
from collections.abc import Callable
from datetime import datetime, timezone
from uuid import uuid4

from pydantic import ValidationError

from shared.errors import AgentError

from .database import Database
from .models import Fact, Forget, Remember, Scope, Selection

logger = logging.getLogger(__name__)
MAX_FACTS = 100


class MemoryConflict(AgentError):
    """Refresh the record before trying another change."""


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def active(fact: Fact, now: datetime) -> bool:
    return fact.expires_at is None or fact.expires_at > now


class FactStore:
    def __init__(self, database: Database, clock: Callable[[], datetime] = utc_now) -> None:
        self.database = database
        self.clock = clock

    @staticmethod
    def _decode(payload: str) -> Fact:
        try:
            return Fact.model_validate_json(payload)
        except ValidationError as exc:
            raise AgentError("Stored fact is invalid; restore a trusted backup.") from exc

    def list_facts(self, scope: Scope) -> list[Fact]:
        with self.database.connect() as connection:
            rows = connection.execute(
                "SELECT payload FROM facts WHERE profile=? ORDER BY key LIMIT ?",
                (scope.profile, MAX_FACTS + 1),
            ).fetchall()
        if len(rows) > MAX_FACTS:
            raise AgentError("Profile exceeds the supported fact limit.")
        return [self._decode(row[0]) for row in rows]

    def select(self, selection: Selection) -> list[Fact]:
        now = self.clock()
        with self.database.connect() as connection:
            records = []
            for key in selection.keys:
                row = connection.execute(
                    "SELECT payload FROM facts WHERE profile=? AND key=?",
                    (selection.profile, key),
                ).fetchone()
                if row is None:
                    raise AgentError(
                        "A selected memory is missing or expired; inspect the profile."
                    )
                fact = self._decode(row[0])
                if not active(fact, now):
                    raise AgentError(
                        "A selected memory is missing or expired; inspect the profile."
                    )
                records.append(fact)
        return records

    def remember(self, request: Remember) -> Fact:
        now = self.clock()
        if request.expires_at is not None and request.expires_at <= now:
            raise AgentError("Memory expiry must be in the future.")
        with self.database.connect(write=True) as connection:
            row = connection.execute(
                "SELECT payload FROM facts WHERE profile=? AND key=?",
                (request.profile, request.key),
            ).fetchone()
            previous = self._decode(row[0]) if row else None
            if (previous.revision if previous else None) != request.expected_revision:
                raise MemoryConflict(
                    "Memory changed or already exists; inspect its current revision."
                )
            if previous is None:
                count = connection.execute(
                    "SELECT COUNT(*) FROM facts WHERE profile=?", (request.profile,)
                ).fetchone()[0]
                if count >= MAX_FACTS:
                    raise AgentError("Profile fact limit reached; forget unused records first.")
            fact = Fact(
                profile=request.profile,
                key=request.key,
                value=request.value,
                revision=uuid4().hex,
                created_at=previous.created_at if previous else now,
                updated_at=now,
                expires_at=request.expires_at,
            )
            connection.execute(
                "INSERT INTO facts(profile,key,payload) VALUES (?,?,?) "
                "ON CONFLICT(profile,key) DO UPDATE SET payload=excluded.payload",
                (fact.profile, fact.key, fact.model_dump_json()),
            )
        logger.info("fact_remembered")
        return fact

    def forget(self, request: Forget) -> None:
        with self.database.connect(write=True) as connection:
            row = connection.execute(
                "SELECT payload FROM facts WHERE profile=? AND key=?",
                (request.profile, request.key),
            ).fetchone()
            if row is None or self._decode(row[0]).revision != request.expected_revision:
                raise MemoryConflict(
                    "Memory changed or no longer exists; inspect before forgetting."
                )
            connection.execute(
                "DELETE FROM facts WHERE profile=? AND key=?", (request.profile, request.key)
            )
        logger.info("fact_forgotten")
