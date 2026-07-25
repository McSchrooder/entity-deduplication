import argparse
import asyncio
import dataclasses
import json
from pathlib import Path

from dedupe.benchmark import run_benchmark
from dedupe.clustering import evaluate_clusters
from dedupe.config import settings
from dedupe.db import get_db
from dedupe.embedding import Embedder
from dedupe.resolver import get_resolver


def main():
    ap = argparse.ArgumentParser(description="Run the dedupe benchmark")
    ap.add_argument("--db", choices=["mongo", "qdrant", "vespa"], default=None, help="defaults to config")
    ap.add_argument(
        "--resolver",
        choices=["none", "heuristic", "cross-encoder", "llm", "nn"],
        default=None,
        help="defaults to config",
    )
    ap.add_argument("--data", type=Path, default=Path(settings.data_dir))
    ap.add_argument("--limit", type=int, default=None, help="cap query count for a quick run")
    ap.add_argument("--retrieval", choices=["hybrid", "vector", "lexical"], default=None)
    ap.add_argument("--retrieval-k", type=int, default=None)
    ap.add_argument("--rerank-k", type=int, default=None)
    ap.add_argument("--out", type=Path, default=Path(settings.data_dir) / "results")
    args = ap.parse_args()

    with open(args.data / "benchmark_queries.json", encoding="utf-8") as f:
        queries = json.load(f)
    with open(args.data / "benchmark_ground_truth.json", encoding="utf-8") as f:
        ground_truth = json.load(f)
    if args.limit:
        queries = queries[: args.limit]

    db = get_db(args.db)
    resolver = get_resolver(args.resolver)
    embedder = Embedder()
    overrides = {
        k: v
        for k, v in {
            "mode": args.retrieval,
            "retrieval_k": args.retrieval_k,
            "rerank_k": args.rerank_k,
        }.items()
        if v is not None
    }
    params = dataclasses.replace(settings.search, **overrides)

    # limit runs get their own tag so a quick probe can never clobber a full result
    tag = f"{db.name}_{resolver.name}_{params.mode}" + (f"_limit{args.limit}" if args.limit else "")
    result = asyncio.run(
        run_benchmark(
            db,
            resolver,
            embedder,
            queries,
            ground_truth,
            params=params,
            failures_path=args.out / f"failures_{tag}.jsonl",
        )
    )

    summary = result.summary()
    # cluster metrics need the full query set and the index's truth labels
    if resolver.name != "none" and not args.limit:
        with open(args.data / "benchmark_db_index.json", encoding="utf-8") as f:
            index = json.load(f)
        id_field, label_field = settings.schema.id_field, settings.schema.label_field
        truth_label_of = {str(d[id_field]): str(d[label_field]) for d in index}
        truth_label_of |= {str(i["query_id"][id_field]): str(i["cluster_id"]) for i in ground_truth}
        summary["cluster"] = evaluate_clusters(result.merged_pairs, ground_truth, truth_label_of)
    print(json.dumps(summary, indent=2))
    args.out.mkdir(parents=True, exist_ok=True)
    with open(args.out / f"summary_{tag}.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
