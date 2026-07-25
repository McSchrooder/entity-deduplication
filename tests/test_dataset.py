import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from dedupe.dataset import COLUMNS, build_benchmark, duplicate_tags

# (code, person_id, first, last, source, blob)
ROWS = [
    ("A1", 1, "Ivan", "Horvat", "1", "blob-a"),
    ("A2", 1, "Ivan", "Horvat", "1", "blob-a"),
    ("B1", 2, "Jürgen", "Müller", "1", "blob-b1"),
    ("B2", 2, "Jurgen", "Mueller", "2", "blob-b2"),
    ("C1", 3, "Ana", "Babić", "1", "blob-c"),
]


@pytest.fixture
def corpus(tmp_path):
    cols = {c: [None] * len(ROWS) for c in COLUMNS}
    cols["_identityBlob"] = [r[5] for r in ROWS]
    cols["CODE"] = [r[0] for r in ROWS]
    cols["FIRST_NAME"] = [r[2] for r in ROWS]
    cols["LAST_NAME"] = [r[3] for r in ROWS]
    cols["SOURCE"] = [r[4] for r in ROWS]
    guests = tmp_path / "guests.parquet"
    pq.write_table(pa.table({k: pa.array(v, pa.string()) for k, v in cols.items()}), guests)

    pmap = tmp_path / "map.parquet"
    pq.write_table(
        pa.table(
            {
                "CODE": pa.array([r[0] for r in ROWS], pa.string()),
                "person_id": pa.array([r[1] for r in ROWS], pa.int64()),
            }
        ),
        pmap,
    )
    return guests, pmap


def test_query_held_out_and_matches_indexed(corpus):
    guests, pmap = corpus
    index, queries, gt = build_benchmark(guests, pmap, index_size=10, num_queries=10)

    indexed = {d["code"] for d in index}
    assert len(queries) == 2
    for item in gt:
        q = str(item["query_id"]["code"])
        assert q not in indexed
        assert {m["code"] for m in item["expected_matches"]} <= indexed


def test_singleton_indexed_as_distractor(corpus):
    guests, pmap = corpus
    index, _, _ = build_benchmark(guests, pmap, index_size=10, num_queries=10)
    assert "C1" in {d["code"] for d in index}


def test_cluster_id_carried_for_mining(corpus):
    guests, pmap = corpus
    index, _, gt = build_benchmark(guests, pmap, index_size=10, num_queries=10)
    assert all("meta_unified_id" in d for d in index)
    assert all(isinstance(item["cluster_id"], int) for item in gt)


def test_labeler_key_never_indexed(corpus):
    guests, pmap = corpus
    index, queries, _ = build_benchmark(guests, pmap, index_size=10, num_queries=10)
    assert all("_identityBlob" not in d for d in index + queries)


def test_deterministic(corpus):
    guests, pmap = corpus
    a = build_benchmark(guests, pmap, index_size=10, num_queries=10, seed=7)
    b = build_benchmark(guests, pmap, index_size=10, num_queries=10, seed=7)
    assert a == b


def test_tags():
    docs = {
        "x": {"first_name": "Ivan", "last_name": "Horvat", "source": "1"},
        "y": {"first_name": "Ivan", "last_name": "Horvat", "source": "1"},
        "z": {"first_name": "Jurgen", "last_name": "Mueller", "source": "2"},
    }
    assert duplicate_tags(["x", "y"], docs, {"x": "b", "y": "b"}) == "Exact"
    tags = duplicate_tags(["x", "z"], docs, {"x": "b1", "z": "b2"})
    assert tags == "Fuzzy + CrossSystem + NameDiff"


def test_diacritics_do_not_count_as_name_difference():
    docs = {
        "x": {"first_name": "Ana", "last_name": "Babić", "source": "1"},
        "y": {"first_name": "Ana", "last_name": "Babic", "source": "1"},
    }
    assert duplicate_tags(["x", "y"], docs, {"x": "b1", "y": "b2"}) == "Fuzzy"
