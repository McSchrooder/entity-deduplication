"""Blocking Pair Completeness / Pair Quality / Reduction Ratio over k.
The tuning instrument for SearchParams.
"""

import argparse
import dataclasses
import json
import time
from pathlib import Path

from tqdm import tqdm

from dedupe.benchmark import BenchmarkResult, build_gt_map
from dedupe.config import settings
from dedupe.db import get_db, search
from dedupe.embedding import Embedder
from dedupe.serialization import doc_id

KS = [10, 25, 50, 100, 150]
WARMUP = 20


def main():
    ap = argparse.ArgumentParser(description="Blocking PC/PQ/RR curve per retrieval mode")
    ap.add_argument("--db", choices=["mongo", "qdrant", "vespa"], default=None, help="defaults to config")
    ap.add_argument("--data", type=Path, default=Path(settings.data_dir))
    ap.add_argument("--modes", nargs="+", default=["lexical", "vector", "hybrid"])
    ap.add_argument("--k-max", type=int, default=150)
    ap.add_argument("--out", type=Path, default=Path(settings.data_dir) / "results")
    args = ap.parse_args()

    with open(args.data / "benchmark_queries.json", encoding="utf-8") as f:
        queries = json.load(f)
    with open(args.data / "benchmark_ground_truth.json", encoding="utf-8") as f:
        gt_map = build_gt_map(json.load(f))
    with open(args.data / "benchmark_db_index.json", encoding="utf-8") as f:
        index_size = len(json.load(f))
    total_expected = sum(len(v) for v in gt_map.values())

    db = get_db(args.db)
    embedder = Embedder()
    params = dataclasses.replace(settings.search, retrieval_k=args.k_max)

    report = {
        "db": db.name,
        "index_size": index_size,
        "queries": len(queries),
        "total_expected": total_expected,
        "modes": {},
    }
    for mode in args.modes:
        ranked: list[tuple[str, list[str]]] = []
        latency: list[float] = []
        for i, q in enumerate(tqdm(queries, desc=mode)):
            qid = doc_id(q)
            t0 = time.perf_counter()
            hits = search(db, embedder, q, mode=mode, params=params)
            if i >= WARMUP:
                latency.append((time.perf_counter() - t0) * 1000)
            ranked.append((qid, [h.id for h in hits if h.id != qid]))

        curve = []
        for k in [k for k in KS if k <= args.k_max]:
            got = cand = 0
            for qid, ids in ranked:
                top = ids[:k]
                cand += len(top)
                got += len(set(top) & gt_map.get(qid, set()))
            curve.append(
                {
                    "k": k,
                    "pair_completeness": round(got / total_expected, 4),
                    "pair_quality": round(got / cand, 4) if cand else 0.0,
                    "reduction_ratio": round(1 - cand / (len(queries) * index_size), 5),
                }
            )
        report["modes"][mode] = {
            "curve": curve,
            "retrieval_latency_ms": BenchmarkResult._percentiles(latency),
        }
        print(json.dumps({mode: report["modes"][mode]}, indent=2))

    args.out.mkdir(parents=True, exist_ok=True)
    with open(args.out / f"blocking_{db.name}.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
