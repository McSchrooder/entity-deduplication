import json

from dedupe.db.base import Hit
from dedupe.mining import mine_pairs


class FakeDB:
    name = "fake"
    supports_lexical = False

    def __init__(self, hits):
        self.hits = hits

    def vector_search(self, vector, k):
        return self.hits


class FakeEmbedder:
    def encode_doc(self, doc):
        return [0.0]


def test_mining_excludes_same_cluster(tmp_path):
    db_docs = [
        {"_id": {"code": "pos1"}, "code": "pos1", "first_name": "Ana", "meta_unified_id": "cluster-A"},
        {"_id": {"code": "neg1"}, "code": "neg1", "first_name": "Ivana", "meta_unified_id": "cluster-B"},
        {"_id": {"code": "neg2"}, "code": "neg2", "first_name": "Anka", "meta_unified_id": "cluster-C"},
    ]
    queries = [{"_id": {"code": "q1"}, "code": "q1", "first_name": "Ana"}]
    gt = [{"query_id": {"code": "q1"}, "cluster_id": "cluster-A", "expected_matches": [{"code": "pos1"}]}]

    hits = [Hit(id=d["code"], doc=d, score=1.0) for d in db_docs]
    out = tmp_path / "pairs.jsonl"

    n = mine_pairs(FakeDB(hits), FakeEmbedder(), queries, gt, db_docs, out)
    assert n == 1

    [record] = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert record["query"]["first_name"] == "Ana"
    assert [p["first_name"] for p in record["pos"]] == ["Ana"]
    # same-cluster doc must not appear as negative
    assert {n["first_name"] for n in record["neg"]} == {"Ivana", "Anka"}
    # docs are stripped to evidence fields, no ids, no labels
    for doc in [record["query"], *record["pos"], *record["neg"]]:
        assert "_id" not in doc
        assert "code" not in doc
        assert "meta_unified_id" not in doc


def test_cluster_zero_still_excluded(tmp_path):
    """Integer cluster ids start at 0 — a falsy check would mine the cluster against itself."""
    db_docs = [
        {"_id": {"code": "pos1"}, "code": "pos1", "first_name": "Ana", "meta_unified_id": 0},
        {"_id": {"code": "neg1"}, "code": "neg1", "first_name": "Ivana", "meta_unified_id": 1},
    ]
    queries = [{"_id": {"code": "q1"}, "code": "q1", "first_name": "Ana"}]
    gt = [{"query_id": {"code": "q1"}, "cluster_id": 0, "expected_matches": [{"code": "pos1"}]}]

    hits = [Hit(id=d["code"], doc=d, score=1.0) for d in db_docs]
    out = tmp_path / "pairs.jsonl"
    mine_pairs(FakeDB(hits), FakeEmbedder(), queries, gt, db_docs, out)

    [record] = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert all(n["first_name"] != "Ana" for n in record["neg"])
