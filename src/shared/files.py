"""Bounded reads for files explicitly selected by the operator."""

import os
import stat
from pathlib import Path

from shared.errors import ToolError


def read_limited(path: Path, max_bytes: int) -> bytes:
    try:
        with path.open("rb") as source:
            if not stat.S_ISREG(os.fstat(source.fileno()).st_mode):
                raise ToolError("Only regular files are supported.")
            data = source.read(max_bytes + 1)
    except OSError as exc:
        raise ToolError("Could not read the selected file.") from exc
    if len(data) > max_bytes:
        raise ToolError("The selected file exceeds the byte limit.")
    return data
