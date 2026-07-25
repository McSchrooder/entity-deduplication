from dataclasses import dataclass, field

from dedupe.config import SearchParams, settings
from dedupe.db.base import DB, search
from dedupe.embedding import Embedder
from dedupe.resolver.base import Resolver
from dedupe.serialization import doc_id


@dataclass
class Resolution:
    record_id: str
    action: str  # "new" | "merged" | "review"
    entity_id: str | None
    matched: list[str] = field(default_factory=list)
    review: list[str] = field(default_factory=list)
    reason: str = ""


class StreamProcessor:
    """Streaming shell around the same search->resolve ports the benchmark uses."""

    def __init__(
        self,
        db: DB,
        resolver: Resolver,
        embedder: Embedder | None,
        *,
        params: SearchParams | None = None,
        max_entity_size: int = 20,
        ingest: bool = True,
    ):
        self.db = db
        self.resolver = resolver
        self.embedder = embedder
        self.params = params or settings.search
        self.max_entity_size = max_entity_size
        self.ingest = ingest
        self._entity_of: dict[str, str] = {}
        self._members: dict[str, list[str]] = {}
        self.audit: list[Resolution] = []

    def seed(self, entities: dict[str, list[str]]) -> None:
        """Preload resolved state, e.g. the incumbent's clusters at cutover."""
        for eid, members in entities.items():
            self._members[eid] = list(members)
            for m in members:
                self._entity_of[m] = eid

    def entity(self, record_id: str) -> str | None:
        return self._entity_of.get(record_id)

    def members(self, entity_id: str) -> list[str]:
        return list(self._members.get(entity_id, []))

    def _entity_for_candidate(self, record_id: str) -> str:
        if record_id not in self._entity_of:
            self._entity_of[record_id] = record_id
            self._members[record_id] = [record_id]
        return self._entity_of[record_id]

    async def resolve(self, doc: dict) -> Resolution:
        rid = doc_id(doc)
        hits = search(self.db, self.embedder, doc, params=self.params)
        candidates = [h for h in hits if h.id != rid][: self.params.rerank_k]

        verdicts = await self.resolver.score(doc, candidates) if candidates else []
        matched = [v.candidate_id for v in verdicts if not v.review and v.score > self.resolver.threshold]
        review = [v.candidate_id for v in verdicts if v.review]

        if not matched:
            res = Resolution(
                rid,
                "review" if review else "new",
                None,
                matched,
                review,
                "resolver flagged candidates" if review else "no candidate cleared the threshold",
            )
            if res.action == "new":
                self._entity_of[rid] = rid
                self._members[rid] = [rid]
                res.entity_id = rid
        else:
            entities = {self._entity_for_candidate(m) for m in matched}
            if len(entities) > 1:
                res = Resolution(
                    rid,
                    "review",
                    None,
                    matched,
                    review,
                    f"matches span {len(entities)} entities — refusing to bridge",
                )
            else:
                eid = entities.pop()
                if len(self._members[eid]) >= self.max_entity_size:
                    res = Resolution(
                        rid,
                        "review",
                        None,
                        matched,
                        review,
                        f"entity {eid} at size cap {self.max_entity_size}",
                    )
                else:
                    self._entity_of[rid] = eid
                    self._members[eid].append(rid)
                    res = Resolution(rid, "merged", eid, matched, review, "joined entity")

        if self.ingest:
            vec = self.embedder.encode_doc(doc) if self.embedder else None
            self.db.ingest([doc], [vec] if vec is not None else [])

        self.audit.append(res)
        return res
