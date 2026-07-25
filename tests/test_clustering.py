from dedupe.clustering import adjusted_rand_index, connected_components, evaluate_clusters


def gt(q, matches, cluster_id="p1"):
    return {
        "query_id": {"code": q},
        "expected_matches": [{"code": m} for m in matches],
        "cluster_id": cluster_id,
    }


def test_components_chain_transitively():
    comps = connected_components([("a", "b"), ("b", "c"), ("x", "y")])
    assert comps["a"] == {"a", "b", "c"}
    assert comps["c"] == {"a", "b", "c"}
    assert comps["x"] == {"x", "y"}


def test_perfect_reproduction():
    truth = {"q1": "p1", "m1": "p1", "m2": "p1", "q2": "p2", "m3": "p2"}
    out = evaluate_clusters(
        [("q1", "m1"), ("q1", "m2"), ("q2", "m3")],
        [gt("q1", ["m1", "m2"], "p1"), gt("q2", ["m3"], "p2")],
        truth,
    )
    assert out["b3_f1"] == 1.0
    assert out["exact_cluster_precision"] == 1.0
    assert out["exact_cluster_recall"] == 1.0
    assert out["ari"] == 1.0


def test_overmerge_bridges_two_truth_clusters():
    truth = {"q1": "p1", "m1": "p1", "q2": "p2", "m2": "p2"}
    out = evaluate_clusters(
        [("q1", "m1"), ("q2", "m2"), ("q1", "m2")],  # last pair is the bad bridge
        [gt("q1", ["m1"], "p1"), gt("q2", ["m2"], "p2")],
        truth,
    )
    assert out["b3_recall"] == 1.0
    assert out["b3_precision"] == 0.5  # every record sees 2 of 4 component members correct
    assert out["exact_cluster_recall"] == 0.0
    assert out["ari"] < 1.0


def test_missed_merge_costs_recall_not_precision():
    truth = {"q1": "p1", "m1": "p1", "m2": "p1"}
    out = evaluate_clusters(
        [("q1", "m1")],
        [gt("q1", ["m1", "m2"], "p1")],
        truth,
    )
    assert out["b3_precision"] == 1.0
    assert out["b3_recall"] < 1.0
    assert out["exact_cluster_recall"] == 0.0


def test_distractor_pulled_in_costs_precision():
    truth = {"q1": "p1", "m1": "p1", "d1": "s_d1"}
    out = evaluate_clusters(
        [("q1", "m1"), ("q1", "d1")],
        [gt("q1", ["m1"], "p1")],
        truth,
    )
    assert out["b3_recall"] == 1.0
    assert round(out["b3_precision"], 4) == round(2 / 3, 4)
    assert out["exact_cluster_precision"] == 0.0


def test_no_merges_scores_zero_ari():
    truth = {"q1": "p1", "m1": "p1"}
    out = evaluate_clusters([], [gt("q1", ["m1"], "p1")], truth)
    assert out["ari"] == 0.0
    assert out["b3_precision"] == 1.0  # singletons are pure, just tiny
    assert out["b3_recall"] == 0.5


def test_ari_identical_partitions():
    a = {"x": "1", "y": "1", "z": "2"}
    assert adjusted_rand_index(a, {"x": "a", "y": "a", "z": "b"}) == 1.0


def test_bootstrap_ci_brackets_point_estimate():
    truth = {f"q{i}": f"p{i}" for i in range(20)} | {f"m{i}": f"p{i}" for i in range(20)}
    pairs = [(f"q{i}", f"m{i}") for i in range(10)]  # half the clusters resolved
    out = evaluate_clusters(pairs, [gt(f"q{i}", [f"m{i}"], f"p{i}") for i in range(20)], truth)
    lo, hi = out["b3_f1_ci95"]
    assert lo <= out["b3_f1"] <= hi
    assert lo < hi
