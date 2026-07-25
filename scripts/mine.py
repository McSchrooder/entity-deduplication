import argparse
import json
from pathlib import Path

from dedupe.config import settings
from dedupe.db import get_db
from dedupe.embedding import Embedder
from dedupe.mining import mine_pairs


def main():
    ap = argparse.ArgumentParser(description="Mine hard-negative doc pairs for training")
    ap.add_argument("--db", choices=["mongo", "qdrant", "vespa"], default=None, help="defaults to config")
    ap.add_argument("--data", type=Path, default=Path(settings.data_dir))
    ap.add_argument("--out", type=Path, default=Path(settings.data_dir) / "pairs.jsonl")
    ap.add_argument("--negatives", type=int, default=7)
    args = ap.parse_args()

    with open(args.data / "benchmark_queries.json", encoding="utf-8") as f:
        queries = json.load(f)
    with open(args.data / "benchmark_ground_truth.json", encoding="utf-8") as f:
        ground_truth = json.load(f)
    with open(args.data / "benchmark_db_index.json", encoding="utf-8") as f:
        db_docs = json.load(f)

    n = mine_pairs(
        get_db(args.db),
        Embedder(),
        queries,
        ground_truth,
        db_docs,
        args.out,
        negatives_per_query=args.negatives,
    )
    print(f"{n} records -> {args.out}")


if __name__ == "__main__":
    main()
