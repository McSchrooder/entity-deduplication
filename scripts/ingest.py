import argparse
import json
from pathlib import Path

from tqdm import tqdm

from dedupe.config import settings
from dedupe.db import get_db
from dedupe.embedding import Embedder
from dedupe.serialization import stringify


def main():
    ap = argparse.ArgumentParser(description="Embed and ingest the benchmark index into a db")
    ap.add_argument("--db", choices=["mongo", "qdrant", "vespa"], default=None, help="defaults to config")
    ap.add_argument("--data", type=Path, default=Path(settings.data_dir) / "benchmark_db_index.json")
    ap.add_argument("--batch", type=int, default=256)
    args = ap.parse_args()

    with open(args.data, encoding="utf-8") as f:
        docs = json.load(f)

    db = get_db(args.db)
    db.init()
    embedder = Embedder()

    total = 0
    for i in tqdm(range(0, len(docs), args.batch), desc="ingest"):
        chunk = docs[i : i + args.batch]
        embeddings = embedder.encode_texts([stringify(d) for d in chunk])
        total += db.ingest(chunk, embeddings)

    print(f"ingested {total} docs -> {db.name} (count={db.count()})")


if __name__ == "__main__":
    main()
