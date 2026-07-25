"""Synthetic smoke corpus in the real export schema."""

import argparse
import random
import unicodedata
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from dedupe.config import settings

FIRST = [
    "Ivan",
    "Ana",
    "Marko",
    "Petra",
    "Luka",
    "Marija",
    "Josip",
    "Elena",
    "Matej",
    "Katarina",
    "Jürgen",
    "Søren",
    "François",
    "Zsófia",
    "Björn",
    "Chiara",
    "Miloš",
    "Agnieszka",
    "Tomáš",
    "Amélie",
    "Nikola",
    "Dražen",
    "Květa",
    "Đurđa",
    "Håkon",
]
LAST = [
    "Horvat",
    "Kovačić",
    "Novak",
    "Babić",
    "Jurić",
    "Müller",
    "Schmidt",
    "Rossi",
    "Bianchi",
    "Kowalski",
    "Nagy",
    "Novotný",
    "Jensen",
    "Larsen",
    "Dubois",
    "García",
    "Öztürk",
    "Papadopoulos",
    "Šarić",
    "Živković",
    "Đorđević",
    "Wagner",
    "Keller",
]
CITIES = ["Zagreb", "Vienna", "Munich", "Milan", "Prague", "Warsaw", "Budapest", "Oslo"]
COUNTRIES = ["HR", "AT", "DE", "IT", "CZ", "PL", "HU", "NO"]


def fold(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))


def typo(rng: random.Random, s: str) -> str:
    if len(s) < 3:
        return s
    i = rng.randrange(1, len(s) - 1)
    return s[:i] + s[i + 1] + s[i] + s[i + 2 :]  # transpose two inner chars


def blob(g: dict) -> str:
    # mimic the pipeline's identity hash
    keys = ("FIRST_NAME", "LAST_NAME", "EMAIL", "PHONE_NUMBER", "CITY", "ZIP_CODE", "CITIZENSHIP")
    return "|".join(fold(str(g.get(k) or "")).lower() for k in keys)


def perturb_dob(rng: random.Random, dob: str) -> str:
    """The anchor-class taxonomy, generated: swap / year_off1 / year_typo."""
    y, m, d = dob.split("-")
    kind = rng.random()
    if kind < 0.4 and int(d) <= 12 and d != m:
        return f"{y}-{d}-{m}"  # day/month swap
    if kind < 0.7:
        return f"{int(y) + rng.choice([-1, 1])}-{m}-{d}"  # year off by one
    return f"{int(y) + rng.choice([-10, 10])}-{m}-{d}"  # tens-digit typo


def base_guest(rng: random.Random, code: int) -> dict:
    nat = rng.choice(COUNTRIES)
    return {
        "CODE": str(code),
        "FIRST_NAME": rng.choice(FIRST),
        "LAST_NAME": rng.choice(LAST),
        "MIDDLE_NAME": None,
        "GENDER": rng.choice(["M", "F", None]),
        "BIRTH_DATE": f"{rng.randint(1945, 2005)}-{rng.randint(1, 12):02}-{rng.randint(1, 28):02}",
        "BIRTH_PLACE": None,
        "CITIZENSHIP": nat,
        "CITY": rng.choice(CITIES) if rng.random() < 0.61 else None,
        "ZIP_CODE": f"{rng.randint(10000, 99999)}" if rng.random() < 0.05 else None,
        "COUNTRY": nat if rng.random() < 0.5 else None,
        "EMAIL": None,
        "PHONE_NUMBER": f"+385{rng.randint(10**8, 10**9 - 1)}" if rng.random() < 0.04 else None,
        "SOURCE": "1",
        "DOCUMENT_NUMBER": None,
    }


def duplicate_of(rng: random.Random, g: dict, code: int) -> dict:
    d = {**g, "CODE": str(code)}
    kind = rng.random()
    if kind < 0.2:
        pass  # exact copy
    elif kind < 0.45:
        d["FIRST_NAME"] = typo(rng, d["FIRST_NAME"])
    elif kind < 0.65:
        d["FIRST_NAME"], d["LAST_NAME"] = fold(d["FIRST_NAME"]), fold(d["LAST_NAME"])
    elif kind < 0.75:
        d["LAST_NAME"] = d["LAST_NAME"].upper()
    else:
        d["BIRTH_DATE"] = perturb_dob(rng, d["BIRTH_DATE"])
    if rng.random() < 0.3:
        d["CITY"] = rng.choice(CITIES)  # contact drift
    if rng.random() < 0.2:
        # cross-system copy: the other system stores no document, email, or phone
        d.update(SOURCE="2", DOCUMENT_NUMBER=None, EMAIL=None, PHONE_NUMBER=None)
    return d


def main():
    ap = argparse.ArgumentParser(description="Generate the synthetic smoke corpus")
    ap.add_argument("--persons", type=int, default=1200)
    ap.add_argument("--duplicate-rate", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=Path(settings.data_dir) / "demo")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    guests, person_map, code = [], [], 1000
    for pid in range(args.persons):
        g = base_guest(rng, code)
        if rng.random() < 0.6:
            g["DOCUMENT_NUMBER"] = f"D{rng.randint(10**6, 10**7 - 1)}"
        members = [g]
        code += 1
        if rng.random() < args.duplicate_rate:
            for _ in range(rng.choice([1, 1, 2, 3])):
                members.append(duplicate_of(rng, g, code))
                code += 1
        for m in members:
            m["_identityBlob"] = blob(m)
            guests.append(m)
            person_map.append({"CODE": m["CODE"], "person_id": pid})

    for i in range(10):
        parent = base_guest(rng, code)
        parent["DOCUMENT_NUMBER"] = f"D{rng.randint(10**6, 10**7 - 1)}"
        code += 1
        child = {
            **base_guest(rng, code),
            "DOCUMENT_NUMBER": parent["DOCUMENT_NUMBER"],
            "LAST_NAME": parent["LAST_NAME"],
        }
        if i % 2:
            child["BIRTH_DATE"] = parent["BIRTH_DATE"]
        code += 1
        for m in (parent, child):
            m["_identityBlob"] = blob(m)
            guests.append(m)
            person_map.append({"CODE": m["CODE"], "person_id": len(person_map) + 10**6})

    args.out.mkdir(parents=True, exist_ok=True)
    cols = [*guests[0].keys()]
    pq.write_table(pa.table({c: [g[c] for g in guests] for c in cols}), args.out / "guests.parquet")
    pq.write_table(
        pa.table(
            {
                "CODE": [p["CODE"] for p in person_map],
                "person_id": [p["person_id"] for p in person_map],
            }
        ),
        args.out / "person_map.parquet",
    )
    dupes = sum(1 for p in person_map) - args.persons - 20
    print(f"{len(guests)} profiles, {args.persons + 20} persons, {dupes} duplicates -> {args.out}")


if __name__ == "__main__":
    main()
