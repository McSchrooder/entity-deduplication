# dedupe

Tiered entity resolution for guest profiles. Hybrid retrieval narrows the corpus to a
candidate set, then a resolver decides between auto-merge, reject and human review.

Four resolvers compete on one benchmark.

| resolver | what | cost |
|---|---|---|
| `heuristic` | rapidfuzz + schema-role-weighted field evidence, fuzzy DOB logic | CPU, ~1 ms/query |
| `nn` | calibrated MLP over ~19 schema-driven comparison features, trained on mined hard negatives | CPU train (seconds) and serve, ~3 ms/query |
| `cross-encoder` | bge-reranker-v2-m3, query and candidate jointly attended | GPU, ~115 ms/query |
| `llm` | any OpenAI-compatible judge, structured 3-tier verdict (merge / reject / review) | API, ~$ at scale |

Dbs are pluggable behind a small protocol, currently MongoDB Atlas Search (CE), Qdrant
and Vespa. Lexical search is an optional capability, dbs without it degrade to vector-only
retrieval. One db and one resolver are chosen in config, every script can override per
run.

The field schema lives in config as role-tagged specs (`id`, `label`, `name`,
`hard_anchor`, `soft_anchor`, `context`, `detail`).
Serialization order, lexical index
fields, the export column mapping, the heuristic's evidence weights, the nn's feature
vector and the generated Vespa schema all derive from it. A new corpus requires a new schema.

The LLM resolver uses the OpenAI wire protocol.

## Corpus and labels

A production guest export of ~1.3M profiles from two hospitality systems, the scale
this architecture was built for and ran against. The benchmark is the 300,000
most recently updated profiles (Sept 2023 to Oct 2025). Within it, 17% of profiles
belong to a person with more than one profile.

Both source systems support explicit profile linking, but the ETL consumes it, linked
records collapse into one row at the sink's upsert key. What survives into the export
is the human signal as a document number, the same physical ID card typed at the front desk onto
several profiles. `scripts/build_labels.py` turns that into ground truth.

Clusters are tagged so recall breaks down by how hard they are.

| tag | share | meaning |
|---|---|---|
| `Exact` | 989 | every profile shares the identity hash, carbon copies |
| `Fuzzy` | 2,011 | identity fields differ, contact drift, spacing, diacritics |
| `NameDiff` | 585 | normalized names disagree, typos, transliteration |

The benchmark holds out 3,000 labeled clusters as queries (4,722 expected matches).

The index is 200,000 profiles.

## Metrics

Strict P/R/F1, plus operational recall.

Accepted merges are chained into connected components and scored again at the cluster level with B³
precision and recall, exact-cluster match rate and chance-adjusted ARI, with a
bootstrap CI that resamples whole clusters.

## Results

200,000 indexed profiles, 3,000 held-out clusters, 4,722 expected matches, document-GT,
`mongodb-atlas-local` engine.

### Retrieval ablation (mongo, k=150, no resolving)

| mode | recall | Exact | Fuzzy | NameDiff |
|---|---|---|---|---|
| lexical | 0.9941 | 1.0000 | 0.9907 | 0.9650 |
| vector | 0.9945 | 1.0000 | 0.9914 | 0.9831 |
| hybrid | 1.0000 | 1.0000 | 1.0000 | 1.0000 |


### Blocking quality over k

The search index is the blocking
strategy, an inverted index with fuzzy term expansion on the lexical side and an HNSW
graph on the vector side, fused by rank.

Pair Completeness per mode:

| k | lexical | vector | hybrid |
|---|---|---|---|
| 10 | 0.9526 | 0.9553 | 0.9651 |
| 25 | 0.9869 | 0.9845 | 0.9936 |
| 50 | 0.9894 | 0.9905 | 0.9987 |
| 100 | 0.9924 | 0.9932 | 0.9992 |
| 150 | 0.9941 | 0.9945 | 0.9998 |

`scripts/blocking.py` is the
tuning script for the retrieval knobs.

### Resolvers (hybrid retrieval, k=150, rerank top-50, 3,000 queries)

| resolver | precision | recall | F1 | operational recall |
|---|---|---|---|---|
| **nn** (comparison features into a calibrated MLP) | **0.949** | **0.999** | **0.973** | 0.999 |
| heuristic (schema-role rules) | 0.944 | 0.999 | 0.971 | 0.999 |
| heuristic, lexical-only retrieval, no GPU (1,000-query run) | 0.941 | 0.989 | 0.964 | 0.989 |
| llm, gemini-3.1-flash-lite (top-10) | 1.000 | 0.958 | 0.979 | 0.958 |
| cross-encoder (1,000-query run) | 0.670 | 0.992 | 0.800 | 0.992 |


### Cluster level

| resolver | B³ P | B³ R | B³ F1 (95% CI) | exact P | exact R | ARI |
|---|---|---|---|---|---|---|
| nn | 0.978 | 0.999 | 0.988 [0.987, 0.990] | 0.936 | 0.931 | **0.964** |
| heuristic | 0.975 | 0.999 | 0.987 [0.985, 0.989] | 0.930 | 0.924 | 0.959 |

### Training receipts

Mined pairs come from 3,000 query records, positives from labeled clusters, negatives
from what vector search retrieves outside them. Negatives are presumed, not confirmed,
and with sparse labels some are wrong.

Validation is split group-wise, all of a query's pairs stay on one side.

| run | val AUC |
|---|---|
| MLP on 19 comparison features | 0.9952 |
| GBDT on the same features (control) | 0.9942 |
| MLP, Optuna-tuned (20 trials) | 0.9958 |


### Latency

RTX 4090 Laptop (16 GB) + i9-13900HX, single request at a time, first 20 queries
excluded as warmup.

| stage | p50 | p95 | p99 |
|---|---|---|---|
| retrieval, lexical | 13.2 ms | 18.8 ms | 25.7 ms |
| retrieval, vector | 77.0 ms | 106.1 ms | 144.7 ms |
| retrieval, hybrid | 96.9 ms | 151.1 ms | 209.4 ms |
| resolving, heuristic | 2.6 ms | 6.6 ms | 8.5 ms |
| resolving, nn | 4.3 ms | 5.8 ms | 6.7 ms |
| resolving, cross-encoder | ~115 ms | ~146 ms | ~165 ms |


## Run

Setup + start Db.

```
uv sync --extra mongo --extra ml
docker compose -f docker/mongo/compose.yml up -d --wait
```

Measure.

```
uv run python scripts/build_labels.py               # linkage column -> person_map (ours: DOCUMENT_NUMBER)
uv run python scripts/build_benchmark.py --index-size 200000 --queries 3000
uv run python scripts/ingest.py
uv run python scripts/blocking.py                   # tune the retrieval knobs
uv run python scripts/benchmark.py --resolver heuristic
```

Db and resolver default from config (`DEDUPE_DB`, `DEDUPE_RESOLVER`), and `--db`,
`--resolver`, `--retrieval hybrid|vector|lexical`, `--retrieval-k`, `--rerank-k`
override per run. `--limit N` is a quick probe, it writes its summary under a separate
tag and skips cluster metrics. The LLM resolver needs any OpenAI-compatible endpoint in
`.env` (`LLM_BASE_URL`, `LLM_MODEL`, `LLM_API_KEY`), then `--resolver llm`.

### No corpus? Run the smoke demo

```
uv run python scripts/demo_data.py
uv run python scripts/build_benchmark.py --guests data/demo/guests.parquet \
    --person-map data/demo/person_map.parquet --index-size 2000 --queries 150 --out data/demo_bench
uv run python scripts/ingest.py --db qdrant --data data/demo_bench/benchmark_db_index.json
uv run python scripts/benchmark.py --db qdrant --resolver heuristic --data data/demo_bench
```


## Train the nn resolver

```
uv run python scripts/mine.py                       # hard negatives via vector search -> doc pairs
uv run python scripts/train.py
uv run python scripts/benchmark.py --resolver nn
```


```
uv run python scripts/train.py --model gbdt
uv run python scripts/train.py --exclude-features birth_date_exact
uv run python scripts/tune.py --trials 100
uv run python scripts/train.py --params models/tuned_params.json
uv run python scripts/export_onnx.py --int8
```

## Data handling

The corpus stays out of the repo. `data/` and `models/` are ignored, the loader drops
identity-document numbers on the way in (they exist only inside `build_labels.py`, as
label provenance), and the pipeline's identity hash is read for tagging but never
indexed. Only aggregates are published, the metric summaries under `data/results/` are
committed and the per-record failure logs are not. [DATASHEET.md](DATASHEET.md)
documents the corpus.

## Layout

```
src/dedupe/
  config.py       the choices, db, resolver, SearchParams, and the field schema (roles)
  db/             DB protocol + mongo/qdrant/vespa, RRF fusion and the search() entry point
  resolver/       Resolver protocol + heuristic, nn, cross-encoder, llm, features.py featurizer
  training/       PairClassifier (comparison features -> calibrated MLP) + train/GBDT control
  dataset.py      export + person map -> index, queries, ground truth
  mining.py       hard-negative doc-pair miner
  clustering.py   B3 / ARI / exact-cluster metrics after transitive closure
  stream.py       one record at a time
  benchmark.py    the scoreboard
scripts/          thin argparse shells over the library, in pipeline order
```

## Tests

```
uv run pytest
```
