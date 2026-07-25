from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from dedupe.db.base import Hit


@dataclass
class Verdict:
    candidate_id: str
    score: float  # 0..1, above resolver threshold => merge
    review: bool = False  # route to human queue instead of auto-merge
    reason: str = ""


@runtime_checkable
class Resolver(Protocol):
    """Second-stage adjudicator: query profile vs retrieved candidates."""

    name: str
    threshold: float

    async def score(self, query_doc: dict, candidates: list[Hit]) -> list[Verdict]: ...
