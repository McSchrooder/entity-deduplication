import asyncio

from dedupe.db.base import Hit
from dedupe.resolver.heuristic import HeuristicResolver, score_dob


def run(coro):
    return asyncio.run(coro)


def _score(query, cand):
    resolver = HeuristicResolver()
    [verdict] = run(resolver.score(query, [Hit(id="c1", doc=cand, score=0.0)]))
    return verdict


def test_exact_match_merges():
    q = {"first_name": "Ivan", "last_name": "Horvat", "birth_date": "1980-05-05", "city": "Split"}
    v = _score(q, dict(q))
    assert v.score > HeuristicResolver().threshold
    assert not v.review


def test_dob_conflict_kills_match():
    q = {"first_name": "Ivan", "last_name": "Horvat", "birth_date": "1980-05-05"}
    c = {**q, "birth_date": "1975-02-02"}
    assert _score(q, c).score < 0.85


def test_diacritics_ignored():
    q = {"first_name": "Šimun", "last_name": "Babić", "birth_date": "1960-03-03"}
    c = {"first_name": "Simun", "last_name": "Babic", "birth_date": "1960-03-03"}
    assert _score(q, c).score > 0.9


def test_score_dob_variants():
    assert score_dob("1990-01-01", "1990-01-01")[0] == 25
    assert score_dob("1990-05-12", "1990-12-05")[0] == 15
    assert score_dob("1990-01-01", "1991-01-01")[0] == 10
    assert score_dob("1990-01-01", "1980-01-01")[0] == 5
    assert score_dob("1990-01-01", "1993-06-06")[0] == -50
    assert score_dob(None, "1990-01-01")[0] == 0
