from dedupe.db.base import Hit, rrf_fuse


def _hits(*ids):
    return [Hit(id=i, doc={"code": i}, score=1.0) for i in ids]


def test_rrf_rewards_presence_in_both_lists():
    fused = rrf_fuse(_hits("a", "b", "c"), _hits("c", "d"))
    assert fused[0].id == "c"
    assert {h.id for h in fused} == {"a", "b", "c", "d"}


def test_rrf_scores_monotonic():
    fused = rrf_fuse(_hits("a", "b"), _hits("a", "b"))
    scores = [h.extra["rrf_score"] for h in fused]
    assert scores == sorted(scores, reverse=True)


def test_rrf_empty():
    assert rrf_fuse([], []) == []
