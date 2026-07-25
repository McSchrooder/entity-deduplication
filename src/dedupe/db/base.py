"""The universal db contract plus the retrieval entry point every consumer
(benchmark, blocking, mining, stream) goes through."""

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from dedupe.config import SearchParams, settings
from dedupe.embedding import Embedder


@dataclass
class Hit:
    id: str
    doc: dict
    score: float
    extra: dict = field(default_factory=dict)


@runtime_checkable
class DB(Protocol):
    """Minimal contract a store must satisfy to plug into the pipeline.

    lexical_search is optional capability, dbs without a text index report
    supports_lexical=False and retrieval degrades to vector-only.
    """

    name: str
    supports_lexical: bool

    def init(self) -> None:
        """Bring the store to queryable,
        Assumes the container is already up (docker/<name>/compose.yml)."""
        ...

    def ingest(self, docs: list[dict], embeddings: list[list[float]]) -> int: ...

    def vector_search(self, vector: list[float], k: int) -> list[Hit]: ...

    def lexical_search(self, query_doc: dict, k: int) -> list[Hit]: ...

    def count(self) -> int: ...


def rrf_fuse(*result_lists: list[Hit], k: int = 60) -> list[Hit]:
    scores: dict[str, float] = {}
    hits: dict[str, Hit] = {}
    for results in result_lists:
        for rank, hit in enumerate(results):
            scores[hit.id] = scores.get(hit.id, 0.0) + 1.0 / (k + rank + 1)
            hits.setdefault(hit.id, hit)
    fused = sorted(hits.values(), key=lambda h: scores[h.id], reverse=True)
    for h in fused:
        h.extra["rrf_score"] = scores[h.id]
    return fused


def search(
    db: DB,
    embedder: Embedder | None,
    query_doc: dict,
    *,
    mode: str | None = None,
    k: int | None = None,
    params: SearchParams | None = None,
) -> list[Hit]:
    """Candidate retrieval."""
    params = params or settings.search
    mode = mode or params.mode
    k = k or params.retrieval_k

    if mode == "lexical":
        if not db.supports_lexical:
            raise ValueError(f"{db.name} has no lexical index")
        return db.lexical_search(query_doc, k)

    vector = embedder.encode_doc(query_doc)
    vec_hits = db.vector_search(vector, k)
    if mode == "vector" or not db.supports_lexical:
        return vec_hits
    return rrf_fuse(vec_hits, db.lexical_search(query_doc, k), k=params.rrf_k)
