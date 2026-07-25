import argparse
import json
from collections import Counter
from pathlib import Path

from dedupe.config import settings
from dedupe.dataset import build_benchmark


def main():
    ap = argparse.ArgumentParser(description="Build the benchmark from the guest export + linking table")
    ap.add_argument("--guests", type=Path, default=Path(settings.guests_parquet))
    ap.add_argument("--person-map", type=Path, default=Path(settings.person_map_parquet))
    ap.add_argument("--index-size", type=int, default=200_000)
    ap.add_argument("--queries", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=Path(settings.data_dir))
    args = ap.parse_args()

    index, queries, ground_truth = build_benchmark(
        args.guests,
        args.person_map,
        index_size=args.index_size,
        num_queries=args.queries,
        seed=args.seed,
    )

    args.out.mkdir(parents=True, exist_ok=True)
    for name, data in [("db_index", index), ("queries", queries), ("ground_truth", ground_truth)]:
        path = args.out / f"benchmark_{name}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)
        print(f"{path}: {len(data)} records")

    expected = sum(len(g["expected_matches"]) for g in ground_truth)
    tags = Counter(t for g in ground_truth for t in g["duplicate_type"].split(" + "))
    print(f"expected matches: {expected} | cluster tags: {dict(tags.most_common())}")


if __name__ == "__main__":
    main()
