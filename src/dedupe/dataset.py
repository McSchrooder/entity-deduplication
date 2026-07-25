"""Benchmark built from a production guest export plus its linking table."""

import random
import re
import unicodedata
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq

from dedupe.config import settings

COLUMNS = settings.schema.source_map

# the labeler's retrieval key
_BLOB = "_identityBlob"


def _norm(s) -> str:
    if not s:
        return ""
    s = unicodedata.normalize("NFKD", str(s).lower())
    return re.sub(r"[^a-z0-9]", "", "".join(c for c in s if not unicodedata.combining(c)))


def load_person_map(path: str | Path) -> dict[str, int]:
    id_col = settings.schema[settings.schema.id_field].source
    t = pq.read_table(path, columns=[id_col, "person_id"])
    return dict(zip(t.column(id_col).to_pylist(), t.column("person_id").to_pylist(), strict=True))


def load_guests(path: str | Path, codes: set[str]) -> tuple[dict[str, dict], dict[str, str]]:
    """Stream the export, materialising only the codes the benchmark needs."""
    docs, blobs = {}, {}
    pf = pq.ParquetFile(path)
    id_field = settings.schema.id_field
    id_col = settings.schema[id_field].source
    for batch in pf.iter_batches(batch_size=50_000, columns=[*COLUMNS, _BLOB]):
        cols = batch.to_pydict()
        for i, code in enumerate(cols[id_col]):
            if code not in codes:
                continue
            doc = {"_id": {id_field: code}}
            for src, dst in COLUMNS.items():
                if (v := cols[src][i]) is not None:
                    doc[dst] = v
            docs[code] = doc
            blobs[code] = cols[_BLOB][i]
    return docs, blobs


def duplicate_tags(members: list[str], docs: dict[str, dict], blobs: dict[str, str]) -> str:
    """Why this cluster is hard, as ' + '-joined tags the benchmark breaks recall down by."""
    tags = ["Exact" if len({blobs.get(c) for c in members}) == 1 else "Fuzzy"]
    if len({docs[c].get("source") for c in members}) > 1:
        tags.append("CrossSystem")
    name_fields = [f.name for f in settings.schema.by_role("name")]
    if len({tuple(_norm(docs[c].get(f)) for f in name_fields) for c in members}) > 1:
        tags.append("NameDiff")
    return " + ".join(tags)


def build_benchmark(
    guests_path: str | Path,
    person_map_path: str | Path,
    index_size: int = 200_000,
    num_queries: int = 3000,
    seed: int = 42,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Returns (index, queries, ground_truth).

    One member of each sampled duplicate cluster is held out as the query, the rest go
    into the index. Singleton persons fill the index as distractors.
    """
    person_of = load_person_map(person_map_path)
    clusters = defaultdict(list)
    for code, pid in person_of.items():
        clusters[pid].append(code)

    multi = sorted(sorted(v) for v in clusters.values() if len(v) > 1)
    singles = sorted(v[0] for v in clusters.values() if len(v) == 1)
    rng = random.Random(seed)
    rng.shuffle(multi)
    rng.shuffle(singles)

    chosen = multi[:num_queries]
    members_indexed = sum(len(m) - 1 for m in chosen)
    fill = max(0, index_size - members_indexed)
    distractors = singles[:fill]

    docs, blobs = load_guests(guests_path, {c for m in chosen for c in m} | set(distractors))

    id_field = settings.schema.id_field
    label_field = settings.schema.label_field
    index, queries, ground_truth = [], [], []
    for members in chosen:
        members = [c for c in members if c in docs]
        if len(members) < 2:
            continue
        q_code, rest = members[0], members[1:]
        queries.append(docs[q_code])
        index.extend({**docs[c], label_field: person_of[c]} for c in rest)
        ground_truth.append(
            {
                "query_id": {id_field: q_code},
                "expected_matches": [{id_field: c} for c in rest],
                "cluster_id": person_of[q_code],
                "duplicate_type": duplicate_tags(members, docs, blobs),
            }
        )
    index.extend({**docs[c], label_field: person_of[c]} for c in distractors if c in docs)

    return index, queries, ground_truth
