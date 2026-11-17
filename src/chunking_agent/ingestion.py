import hashlib
import os
import stat
from pathlib import Path

from shared.errors import AgentError
from shared.structured import fingerprint

from .config import Settings
from .models import Document


def read_document(path: Path, limit: int) -> bytes:
    """Open explicit regular files without following the final symlink or blocking on FIFOs."""
    if path.suffix.lower() not in {".txt", ".md"}:
        raise AgentError("Only UTF-8 .txt and .md documents are supported.")
    if not hasattr(os, "O_NOFOLLOW"):
        raise AgentError("Document ingestion requires POSIX no-follow file support.")
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW)
        with os.fdopen(fd, "rb") as stream:
            if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
                raise AgentError("Document inputs must be regular files.")
            data = stream.read(limit + 1)
    except OSError as exc:
        raise AgentError("Could not open a selected document safely.") from exc
    if len(data) > limit:
        raise AgentError("Document or total byte budget exceeded.")
    return data


def load_documents(paths: list[Path], settings: Settings) -> tuple[Document, ...]:
    if not 1 <= len(paths) <= 20:
        raise AgentError("Select between 1 and 20 documents.")
    if len({p.name for p in paths}) != len(paths):
        raise AgentError("Document basenames must be unique within a run.")
    documents = []
    consumed = 0
    for path in paths:
        data = read_document(
            path, min(settings.max_document_bytes, settings.max_total_bytes - consumed)
        )
        consumed += len(data)
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise AgentError("Documents must contain valid UTF-8 text.") from exc
        if not text.strip() or "\x00" in text:
            raise AgentError("Documents must be nonblank text without NUL characters.")
        digest = hashlib.sha256(data).hexdigest()
        documents.append(
            Document(
                id="d_" + fingerprint({"name": path.name, "sha256": digest})[:32],
                name=path.name,
                sha256=digest,
                text=text,
            )
        )
    return tuple(sorted(documents, key=lambda document: document.id))
