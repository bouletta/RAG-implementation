"""Immutable application snapshots over a trusted, operator-owned Chroma directory."""

import logging
import re
import struct
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from chunking_agent.models import Build
from embedding_rag_agent.embeddings import preflight, prepare
from embedding_rag_agent.models import Match, Usage
from embedding_rag_agent.search import ranked, validate_search
from pydantic import ValidationError
from semantic_memory_agent.embeddings import Embedder, normalize

from shared.errors import AgentError
from shared.structured import fingerprint

from .config import Settings
from .models import IngestResult, Snapshot

logger = logging.getLogger(__name__)


def validate_name(name: str) -> None:
    if not re.fullmatch(r"[a-z][a-z0-9_-]{2,62}", name):
        raise AgentError("Snapshot name must be 3–63 lowercase letters, digits, _ or -.")


def validate_build(built: Build) -> None:
    if not built.corpus.passages:
        raise AgentError("Cannot persist an empty corpus.")
    if built.manifest.corpus_sha256 != fingerprint(built.corpus.model_dump(mode="json")):
        raise AgentError("Corpus does not match its chunk manifest.")
    if [c.id for c in built.manifest.chunks] != [p.id for p in built.corpus.passages]:
        raise AgentError("Manifest chunk mapping does not match the corpus.")


def float32_rows(rows: tuple[tuple[float, ...], ...]) -> list[list[float]]:
    return [[struct.unpack("f", struct.pack("f", value))[0] for value in row] for row in rows]


class Store:
    def __init__(self, path: Path) -> None:
        try:
            import chromadb
            from chromadb.config import Settings as ChromaSettings
        except ImportError as exc:
            raise AgentError("Install agents/035_chroma_rag/requirements.txt.") from exc
        try:
            self.client = chromadb.PersistentClient(
                path=str(path), settings=ChromaSettings(anonymized_telemetry=False)
            )
        except Exception as exc:
            raise AgentError("Could not open the local Chroma store.") from exc

    def _collection(self, name: str) -> Any:
        validate_name(name)
        try:
            return self.client.get_collection(name, embedding_function=None)
        except Exception as exc:
            raise AgentError("Snapshot is unavailable; ingest it first.") from exc

    def ingest(
        self, name: str, built: Build, settings: Settings, embedder: Embedder
    ) -> IngestResult:
        validate_name(name)
        validate_build(built)
        preflight(built.corpus, [], settings)
        try:
            from chromadb.errors import NotFoundError

            existing = self.client.get_collection(name, embedding_function=None)
        except NotFoundError:
            existing = None
        except Exception as exc:
            raise AgentError("Could not inspect the snapshot name.") from exc
        if existing is not None:
            snapshot = self.load(name, settings)
            if snapshot.build != built:
                raise AgentError("Snapshot name already belongs to another corpus; use a new name.")
            return self._receipt(snapshot, Usage(), reused=True)
        # Reserve the name before paid work. Failed builds remain unqueryable, never overwritten.
        try:
            collection = self.client.create_collection(
                name,
                embedding_function=None,
                configuration={"hnsw": {"space": "cosine", "num_threads": 1}},
                metadata={"state": "building"},
            )
        except Exception as exc:
            raise AgentError("Could not reserve the snapshot; choose an unused name.") from exc
        data, usage = prepare(built.corpus, embedder, settings)
        rows = float32_rows(data.vectors)
        snapshot = Snapshot(
            name=name,
            build=built,
            embedding_model=settings.embedding_model,
            dimensions=settings.dimensions,
            vectors_sha256=fingerprint(rows),
        )
        try:
            for start in range(0, len(rows), 32):
                passages = built.corpus.passages[start : start + 32]
                batch: list[Sequence[float] | Sequence[int]] = list(rows[start : start + 32])
                collection.add(
                    ids=[p.id for p in passages],
                    documents=[p.text for p in passages],
                    embeddings=batch,
                )
            self._validate_records(collection, snapshot)
            collection.modify(metadata={"state": "ready", "snapshot": snapshot.model_dump_json()})
        except AgentError:
            raise
        except Exception as exc:
            raise AgentError(
                "Snapshot publication failed; it must not be used. Use a new name."
            ) from exc
        logger.info("chroma_snapshot_published")
        return self._receipt(snapshot, usage, reused=False)

    @staticmethod
    def _receipt(snapshot: Snapshot, usage: Usage, *, reused: bool) -> IngestResult:
        return IngestResult(
            snapshot=snapshot.name,
            corpus_sha256=snapshot.build.manifest.corpus_sha256,
            passages=len(snapshot.build.corpus.passages),
            reused=reused,
            embedding_usage=usage,
        )

    def load(self, name: str, settings: Settings) -> Snapshot:
        collection = self._collection(name)
        metadata = collection.metadata or {}
        raw = metadata.get("snapshot")
        if metadata.get("state") != "ready" or not isinstance(raw, str):
            raise AgentError("Snapshot is incomplete; use a new name to rebuild.")
        if len(raw.encode("utf-8")) > 8_000_000:
            raise AgentError("Snapshot metadata exceeds the supported size.")
        try:
            snapshot = Snapshot.model_validate_json(raw)
        except ValidationError as exc:
            raise AgentError("Snapshot metadata is invalid.") from exc
        if snapshot.name != name:
            raise AgentError("Snapshot name does not match its metadata.")
        if (snapshot.embedding_model, snapshot.dimensions) != (
            settings.embedding_model,
            settings.dimensions,
        ):
            raise AgentError("Embedding model or dimensions differ from the stored snapshot.")
        validate_build(snapshot.build)
        try:
            self._validate_records(collection, snapshot)
        except AgentError:
            raise
        except Exception as exc:
            raise AgentError("Snapshot integrity validation failed.") from exc
        return snapshot

    @staticmethod
    def _validate_records(collection: Any, snapshot: Snapshot) -> None:
        if collection.configuration.get("hnsw", {}).get("space") != "cosine":
            raise AgentError("Snapshot must use cosine distance.")
        passages = snapshot.build.corpus.passages
        if collection.count() != len(passages):
            raise AgentError("Snapshot record count is inconsistent.")
        records = collection.get(limit=201, include=["documents", "embeddings"])
        ids = records["ids"]
        if len(ids) != len(set(ids)) or set(ids) != {p.id for p in passages}:
            raise AgentError("Snapshot passage IDs are inconsistent.")
        documents, vectors = records["documents"], records["embeddings"]
        if documents is None or vectors is None:
            raise AgentError("Snapshot is missing documents or vectors.")
        lookup = {id_: i for i, id_ in enumerate(ids)}
        rows = []
        for passage in passages:
            i = lookup[passage.id]
            if documents[i] != passage.text:
                raise AgentError("Stored text differs from the snapshot corpus.")
            row = [float(value) for value in vectors[i]]
            normalize(row, snapshot.dimensions)
            rows.append(row)
        if fingerprint(rows) != snapshot.vectors_sha256:
            raise AgentError("Stored vectors differ from the published snapshot.")

    def search(
        self, snapshot: Snapshot, vector: tuple[float, ...], top_k: int, minimum: float
    ) -> list[Match]:
        validate_search(top_k, minimum)
        unit = normalize(list(vector), snapshot.dimensions)
        try:
            result = self._collection(snapshot.name).query(
                query_embeddings=[unit],
                n_results=min(top_k, len(snapshot.build.corpus.passages)),
                include=["distances"],
            )
            ids, distances = result["ids"][0], result["distances"][0]
            if len(ids) != len(distances) or not set(ids) <= {
                p.id for p in snapshot.build.corpus.passages
            }:
                raise AgentError("Chroma returned an invalid passage mapping.")
            return ranked(
                [
                    (id_, 1.0 - float(distance))
                    for id_, distance in zip(ids, distances, strict=True)
                ],
                top_k,
                minimum,
            )
        except AgentError:
            raise
        except Exception as exc:
            raise AgentError("Chroma search failed.") from exc
