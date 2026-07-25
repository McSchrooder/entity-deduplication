import asyncio

from dedupe.db.base import Hit
from dedupe.resolver.base import Verdict
from dedupe.stream import StreamProcessor


class FakeDB:
    name = "fake"
    supports_lexical = False

    def __init__(self, hits):
        self.hits = hits
        self.ingested = []

    def init(self):
        pass

    def ingest(self, docs, embeddings):
        self.ingested.extend(docs)
        return len(docs)

    def vector_search(self, vector, k):
        return self.hits

    def lexical_search(self, query_doc, k):
        return []

    def count(self):
        return len(self.ingested)


class FakeEmbedder:
    def encode_doc(self, doc):
        return [0.0]


class FakeResolver:
    name = "fake"
    threshold = 0.5

    def __init__(self, scores, review=()):
        self.scores = scores
        self.review = set(review)

    async def score(self, query_doc, candidates):
        return [Verdict(c.id, self.scores.get(c.id, 0.0), review=c.id in self.review) for c in candidates]


def hit(code):
    return Hit(id=code, doc={"_id": {"code": code}}, score=1.0)


def stream(hits, scores, review=(), **kw):
    return StreamProcessor(FakeDB(hits), FakeResolver(scores, review), FakeEmbedder(), **kw)


def run(s, code):
    return asyncio.run(s.resolve({"_id": {"code": code}}))


def test_no_match_creates_entity():
    s = stream([hit("a")], {"a": 0.1})
    res = run(s, "n1")
    assert res.action == "new"
    assert res.entity_id == "n1"
    assert s.entity("n1") == "n1"


def test_match_joins_lazy_singleton():
    s = stream([hit("a")], {"a": 0.9})
    res = run(s, "n1")
    assert res.action == "merged"
    assert res.entity_id == "a"
    assert sorted(s.members("a")) == ["a", "n1"]


def test_bridge_goes_to_review_not_merge():
    s = stream([hit("a"), hit("b")], {"a": 0.9, "b": 0.9})
    s.seed({"e1": ["a"], "e2": ["b"]})
    res = run(s, "n1")
    assert res.action == "review"
    assert "refusing to bridge" in res.reason
    assert s.entity("n1") is None  # no state change


def test_size_cap_diverts_to_review():
    s = stream([hit("a")], {"a": 0.9}, max_entity_size=2)
    s.seed({"e1": ["a", "b"]})
    res = run(s, "n1")
    assert res.action == "review"
    assert "size cap" in res.reason


def test_review_verdict_blocks_new_entity():
    s = stream([hit("a")], {"a": 0.4}, review=("a",))
    res = run(s, "n1")
    assert res.action == "review"
    assert res.entity_id is None


def test_resolved_record_is_ingested():
    s = stream([], {})
    run(s, "n1")
    assert [d["_id"]["code"] for d in s.db.ingested] == ["n1"]
    s2 = stream([], {}, ingest=False)
    run(s2, "n1")
    assert s2.db.ingested == []


def test_audit_trail_grows_per_record():
    s = stream([hit("a")], {"a": 0.9})
    run(s, "n1")
    run(s, "n2")
    assert [x.record_id for x in s.audit] == ["n1", "n2"]
