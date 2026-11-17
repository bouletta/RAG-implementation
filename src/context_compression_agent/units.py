"""Segment exact slices without merging text across omitted source ranges."""

import re

from basic_rag_agent.models import Passage, Source

from shared.errors import AgentError
from shared.structured import fingerprint

from .config import Settings
from .models import Unit


def segment(parents: list[Source], settings: Settings) -> list[Unit]:
    if len(parents) > settings.top_k or len({p.id for p in parents}) != len(parents):
        raise AgentError("Compression requires distinct retrieved parents within top-k.")
    units = []
    for parent in parents:
        text = parent.text
        boundaries = [m.end() for m in re.finditer(r"[.!?]+(?=\s|$)|\r?\n", text)]
        if not boundaries or boundaries[-1] != len(text):
            boundaries.append(len(text))
        start = 0
        for end in boundaries:
            for left in range(start, end, settings.unit_chars):
                right = min(left + settings.unit_chars, end)
                while left < right and text[left].isspace():
                    left += 1
                while right > left and text[right - 1].isspace():
                    right -= 1
                if left == right:
                    continue
                identity = {
                    "parent": parent.id,
                    "sha256": parent.sha256,
                    "start": left,
                    "end": right,
                    "version": "exact_units_v1",
                }
                passage = Passage(
                    id="u_" + fingerprint(identity)[:32],
                    document_id=parent.document_id,
                    title=parent.title,
                    locator=f"parent={parent.id}; chars={left}:{right}",
                    text=text[left:right],
                )
                units.append(
                    Unit(
                        source=Source(
                            **passage.model_dump(),
                            sha256=fingerprint(passage.model_dump(mode="json")),
                        ),
                        parent_id=parent.id,
                        parent_sha256=parent.sha256,
                        parent_locator=parent.locator,
                        start=left,
                        end=right,
                    )
                )
                if len(units) > settings.max_units:
                    raise AgentError("Compression unit count exceeds its configured budget.")
            start = end
    return units
