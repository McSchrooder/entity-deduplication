import random
from collections import defaultdict

from dedupe.config import settings


def connected_components(pairs: list[tuple[str, str]]) -> dict[str, set[str]]:
    """Union-find over merge decisions -> record: its component (shared set objects)."""
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:
            parent[x], x = root, parent[x]
        return root

    for a, b in pairs:
        parent.setdefault(a, a)
        parent.setdefault(b, b)
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    members: dict[str, set[str]] = defaultdict(set)
    for node in parent:
        members[find(node)].add(node)
    return {node: members[find(node)] for node in parent}


def adjusted_rand_index(labels_a: dict[str, str], labels_b: dict[str, str]) -> float:
    """ARI over the keys of labels_a (both maps must cover them)."""

    def comb2(n: int) -> int:
        return n * (n - 1) // 2

    cont: dict[tuple[str, str], int] = defaultdict(int)
    a_sizes: dict[str, int] = defaultdict(int)
    b_sizes: dict[str, int] = defaultdict(int)
    for k, la in labels_a.items():
        lb = labels_b[k]
        cont[(la, lb)] += 1
        a_sizes[la] += 1
        b_sizes[lb] += 1

    n = len(labels_a)
    sum_cells = sum(comb2(c) for c in cont.values())
    sum_a = sum(comb2(c) for c in a_sizes.values())
    sum_b = sum(comb2(c) for c in b_sizes.values())
    expected = sum_a * sum_b / comb2(n) if n > 1 else 0.0
    max_index = (sum_a + sum_b) / 2
    return (sum_cells - expected) / (max_index - expected) if max_index != expected else 1.0


def evaluate_clusters(
    merged_pairs: list[tuple[str, str]],
    ground_truth: list[dict],
    truth_label_of: dict[str, str],
    bootstrap: int = 1000,
    seed: int = 42,
) -> dict:
    """B3 / ARI / exact-cluster metrics after transitive closure of the merge decisions.

    truth_label_of must cover every record a merge can touch (queries + index).
    Records outside any accepted merge stay singletons.
    """
    pred_of = connected_components(merged_pairs)
    id_field = settings.schema.id_field

    # per truth cluster: (sum of record B3 precisions, sum of recalls, size)
    per_cluster: list[tuple[float, float, int]] = []
    truth_records: set[str] = set()
    truth_keys: set[frozenset] = set()
    exact_found = 0
    for item in ground_truth:
        q = str(item["query_id"][id_field])
        truth = {q} | {str(m[id_field]) for m in item["expected_matches"]}
        truth_records |= truth
        truth_keys.add(frozenset(truth))
        p_sum = r_sum = 0.0
        for r in truth:
            pred = pred_of.get(r, {r})
            inter = len(pred & truth)
            p_sum += inter / len(pred)
            r_sum += inter / len(truth)
        per_cluster.append((p_sum, r_sum, len(truth)))
        if pred_of.get(q, {q}) == truth:
            exact_found += 1

    def b3(cells: list[tuple[float, float, int]]) -> tuple[float, float, float]:
        n = sum(c[2] for c in cells)
        p = sum(c[0] for c in cells) / n
        r = sum(c[1] for c in cells) / n
        f = 2 * p * r / (p + r) if p + r else 0.0
        return p, r, f

    b3_p, b3_r, b3_f = b3(per_cluster)

    rng = random.Random(seed)
    f1s = sorted(
        b3([per_cluster[rng.randrange(len(per_cluster))] for _ in per_cluster])[2] for _ in range(bootstrap)
    )
    ci = (f1s[int(bootstrap * 0.025)], f1s[int(bootstrap * 0.975)])

    # every merge pair involves a query, so every predicted component touches a truth cluster
    pred_components = {frozenset(c) for c in pred_of.values()}
    exact_precision = (
        sum(1 for c in pred_components if c in truth_keys) / len(pred_components) if pred_components else 1.0
    )

    universe = truth_records | set(pred_of)
    pred_labels = {r: min(pred_of[r]) if r in pred_of else r for r in universe}
    ari = adjusted_rand_index({r: truth_label_of[r] for r in universe}, pred_labels)

    return {
        "records": len(universe),
        "clusters": len(per_cluster),
        "b3_precision": round(b3_p, 4),
        "b3_recall": round(b3_r, 4),
        "b3_f1": round(b3_f, 4),
        "b3_f1_ci95": [round(ci[0], 4), round(ci[1], 4)],
        "exact_cluster_precision": round(exact_precision, 4),
        "exact_cluster_recall": round(exact_found / len(per_cluster), 4),
        "ari": round(ari, 4),
    }
