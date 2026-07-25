"""Hard-negative mining."""

import json
import random
from pathlib import Path

from tqdm import tqdm

from dedupe.config import settings
from dedupe.db.base import DB
from dedupe.embedding import Embedder
from dedupe.serialization import doc_id


def _bare(doc: dict) -> dict:
    """Strip ids, labels, and runtime keys, keep only evidence-bearing schema fields."""
    keep = {f.name for f in settings.schema.fields if f.role not in ("id", "label")}
    return {k: v for k, v in doc.items() if k in keep}


def mine_pairs(
    db: DB,
    embedder: Embedder,
    queries: list[dict],
    ground_truth: list[dict],
    db_docs: list[dict],
    out_path: str | Path,
    negatives_per_query: int = 7,
    retrieval_k: int = 150,
    seed: int = 42,
) -> int:
    id_field = settings.schema.id_field
    label_field = settings.schema.label_field
    db_map = {doc_id(d): d for d in db_docs}
    query_map = {doc_id(q): q for q in queries}

    records = []
    for item in tqdm(ground_truth, desc="mining"):
        query = query_map.get(str(item["query_id"][id_field]))
        if not query:
            continue
        cluster_id = item.get("cluster_id")

        positives = [
            _bare(db_map[code]) for m in item["expected_matches"] if (code := str(m[id_field])) in db_map
        ]
        if not positives:
            continue

        vector = embedder.encode_doc(query)
        negatives = []
        for hit in db.vector_search(vector, retrieval_k):
            if cluster_id is not None and hit.doc.get(label_field) == cluster_id:
                continue  # same real person — not a negative
            if hit.id == doc_id(query):
                continue
            negatives.append(_bare(hit.doc))
            if len(negatives) >= negatives_per_query:
                break
        if not negatives:
            continue

        records.append({"query": _bare(query), "pos": positives, "neg": negatives})

    random.Random(seed).shuffle(records)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
    return len(records)
