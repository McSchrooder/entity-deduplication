"""Turn a human-verified linkage column into ground-truth labels.

The rule: same link value + same birth date + names that don't clearly disagree
=> one person. Exact birth date rejects a child registered on a parent's
document."""

import argparse
import itertools
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from rapidfuzz import fuzz

from dedupe.config import settings

MIN_LINK_LEN = 4  # shorter link values are junk ("BB", "000")
MAX_LINK_FREQ = 20  # a link value on more profiles than this is a placeholder
NAME_AGREEMENT = 60  # rejects different humans, keeps typos ("Katarina"/"Katrina" passes)


def norm(s) -> str:
    """lowercase, fold diacritics, keep letters/digits/spaces"""
    if not s:
        return ""
    s = unicodedata.normalize("NFKD", str(s).lower())
    return re.sub(r"[^a-z0-9 ]", "", "".join(c for c in s if not unicodedata.combining(c))).strip()


def _parts_agree(a: tuple[str, ...], b: tuple[str, ...]) -> bool:
    """Per-part comparison: a joined-string ratio lets a shared surname carry two
    different first names past the guard (twins on a parent's document)."""
    compared = [(x, y) for x, y in zip(a, b, strict=True) if x and y]
    if compared and all(fuzz.token_set_ratio(x, y) >= NAME_AGREEMENT for x, y in compared):
        return True
    if len(a) == 2 and all(a) and all(b):
        # crossed check rescues first/last swapped at the desk
        return (
            fuzz.token_set_ratio(a[0], b[1]) >= NAME_AGREEMENT
            and fuzz.token_set_ratio(a[1], b[0]) >= NAME_AGREEMENT
        )
    if not compared:  # disjoint filled parts — all we can do is compare what's there
        return fuzz.token_set_ratio(" ".join(a), " ".join(b)) >= NAME_AGREEMENT
    return False


def names_agree(names: list[tuple[str, ...]]) -> bool:
    return all(_parts_agree(a, b) for a, b in itertools.combinations(names, 2))


def main():
    ap = argparse.ArgumentParser(description="Derive person_map labels from a linkage column")
    ap.add_argument("--guests", type=Path, default=Path(settings.guests_parquet))
    ap.add_argument("--link-column", default="DOCUMENT_NUMBER", help="raw column carrying the human signal")
    ap.add_argument("--out", type=Path, default=Path(settings.person_map_parquet))
    args = ap.parse_args()

    schema = settings.schema
    id_col = schema[schema.id_field].source
    name_cols = [f.source for f in schema.by_role("name")]
    dob_col = schema.by_role("hard_anchor")[0].source

    table = pq.read_table(args.guests, columns=[id_col, args.link_column, dob_col, *name_cols])

    # bucket profiles by (link value, birth date)
    buckets: dict[tuple, list[tuple[str, tuple[str, ...]]]] = defaultdict(list)  # -> [(id, name parts)]
    link_freq: Counter = Counter()
    for row in table.to_pylist():
        link = norm(row[args.link_column]).replace(" ", "")
        dob = str(row[dob_col])[:10] if row[dob_col] else None
        parts = tuple(norm(row[c]) for c in name_cols)
        if len(link) < MIN_LINK_LEN or not dob or not any(parts):
            continue
        link_freq[link] += 1
        buckets[(link, dob)].append((row[id_col], parts))

    placeholders = {link for link, n in link_freq.items() if n > MAX_LINK_FREQ}
    person_of: dict[str, int] = {}
    pid = 0
    dropped = 0
    for (link, _dob), members in buckets.items():
        if len(members) < 2 or link in placeholders:
            continue
        if not names_agree([name for _id, name in members]):
            dropped += 1
            continue
        for profile_id, _name in members:
            person_of[profile_id] = pid
        pid += 1

    # everyone else is a singleton person
    for profile_id in table.column(id_col).to_pylist():
        if profile_id not in person_of:
            person_of[profile_id] = pid
            pid += 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table({id_col: list(person_of), "person_id": list(person_of.values())}), args.out)

    sizes = Counter(Counter(person_of.values()).values())
    clusters = sum(n for size, n in sizes.items() if size > 1)
    records = sum(size * n for size, n in sizes.items() if size > 1)
    print(f"{args.out}: {len(person_of)} profiles")
    print(f"duplicate clusters: {clusters} covering {records} profiles ({records / len(person_of):.1%})")
    print(f"placeholder link values skipped: {len(placeholders)} | groups dropped on name clash: {dropped}")


if __name__ == "__main__":
    main()
