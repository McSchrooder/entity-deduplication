import asyncio

from dedupe.benchmark import BenchmarkResult, build_gt_map, run_benchmark
from dedupe.db.base import Hit
from dedupe.resolver.base import Verdict


class FakeDB:
    name = "fake"
    supports_lexical = False

    def __init__(self, hits_by_query):
        self.hits_by_query = hits_by_query
        self._current = None

    def vector_search(self, vector, k):
        return self.hits_by_query.get(self._current, [])

    def lexical_search(self, query_doc, k):
        return []


class FakeEmbedder:
    def encode_doc(self, doc):
        return [0.0]


class FakeResolver:
    name = "fake"
    threshold = 0.5

    def __init__(self, verdicts):
        self.verdicts = verdicts

    async def score(self, query_doc, candidates):
        return [self.verdicts[h.id] for h in candidates]


def test_metrics_math():
    r = BenchmarkResult(db="x", resolver="y", tp=8, fp=2, fn=4, fn_caught_in_review=3)
    assert r.precision == 0.8
    assert abs(r.recall - 8 / 12) < 1e-9
    assert abs(r.operational_recall - 11 / 12) < 1e-9


def test_build_gt_map_drops_self():
    gt = [{"query_id": {"code": "q1"}, "expected_matches": [{"code": "q1"}, {"code": "m1"}]}]
    assert build_gt_map(gt) == {"q1": {"m1"}}


def test_review_saves_counted():
    queries = [{"_id": {"code": "q1"}, "first_name": "A"}]
    gt = [{"query_id": {"code": "q1"}, "expected_matches": [{"code": "m1"}, {"code": "m2"}]}]
    hits = [Hit(id="m1", doc={"code": "m1"}, score=1.0), Hit(id="m2", doc={"code": "m2"}, score=0.9)]

    db = FakeDB({None: hits})
    resolver = FakeResolver(
        {
            "m1": Verdict(candidate_id="m1", score=0.9),  # merged: TP
            "m2": Verdict(candidate_id="m2", score=0.2, review=True),  # missed but flagged
        }
    )

    result = asyncio.run(run_benchmark(db, resolver, FakeEmbedder(), queries, gt))
    assert (result.tp, result.fp, result.fn) == (1, 0, 1)
    assert result.fn_caught_in_review == 1
    assert result.operational_recall == 1.0
