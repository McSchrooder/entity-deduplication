# Agent instructions

## Commands

```
uv sync --extra dev --extra mongo --extra ml
uv run pytest -q
uv run ruff check .
uv run ruff format --check .
docker compose -f docker/mongo/compose.yml up -d --wait
```

Pipeline order: build_labels (optional, needs a linkage column) -> build_benchmark ->
ingest -> blocking / benchmark. Training branch: mine -> train (or tune) -> export_onnx.
Scripts choose db/resolver from config (`DEDUPE_DB`, `DEDUPE_RESOLVER`), overridable via
`--db` / `--resolver`. The nn resolver consumes schema-driven comparison features
(resolver/features.py).

## Hard constraints

- **Never commit anything under `data/` or `models/`.** The corpus is a production
  guest export with real PII (names, birth dates, emails). `data/results/summary_*.json`
  are aggregates and are the only exception.
- No concrete guest records in README, docs, commit messages, or test fixtures,
  aggregates only. Tests use synthetic in-test fixtures.
- The loader (`src/dedupe/dataset.py`) drops DOCUMENT_NUMBER on ingest and reads
  `_identityBlob` only for cluster tagging, never index it.
- Corpus paths live in `.env` (`DEDUPE_GUESTS_PARQUET`, `DEDUPE_PERSON_MAP_PARQUET`);
  the parquets sit outside the repo. CI runs without them.

## Context that shapes the numbers

Ground-truth labels are the output of the incumbent production deduplicator, not human
adjudication. It filters candidates on exact birth-date equality, so no labeled cluster
disagrees on DOB. Frame all metrics as agreement with production plus what production
misses, never as absolute correctness. Don't "fix" results that look wrong under this
lens before rereading the Corpus and labels section of the README.

## Style

- Match existing idiom: pydantic settings, protocol-based ports
  (`SearchBackend`, `Scorer`), pure functions over classes where state isn't needed.
- `uv run pytest -q` and `uv run ruff check .` must pass before any commit.
