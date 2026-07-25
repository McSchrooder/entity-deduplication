"""One harness, any db, any resolver. P/R/F1 plus operational recall."""

import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from tqdm import tqdm

from dedupe.config import SearchParams, settings
from dedupe.db.base import DB, search
from dedupe.embedding import Embedder
from dedupe.resolver.base import Resolver
from dedupe.serialization import doc_id


@dataclass
class BenchmarkResult:
    db: str
    resolver: str
    retrieval: str = "hybrid"
    queries: int = 0
    tp: int = 0
    fp: int = 0
    fn: int = 0
    fn_caught_in_review: int = 0
    retrieved_expected: int = 0
    total_expected: int = 0
    # duplicate-type tag -> [retrieved, expected]; shows which kinds each retrieval mode survives
    by_type: dict = field(default_factory=dict)
    failures: list[dict] = field(default_factory=list)
    # every accepted merge, for transitive-closure cluster metrics downstream
    merged_pairs: list[tuple[str, str]] = field(default_factory=list)
    # wall-clock ms per query, warmup excluded
    retrieval_ms: list[float] = field(default_factory=list)
    scoring_ms: list[float] = field(default_factory=list)

    @staticmethod
    def _percentiles(samples: list[float]) -> dict | None:
        if not samples:
            return None
        s = sorted(samples)
        pick = lambda q: round(s[min(len(s) - 1, int(len(s) * q))], 2)  # noqa: E731
        return {"p50": pick(0.50), "p95": pick(0.95), "p99": pick(0.99)}

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if self.tp + self.fp else 0.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if self.tp + self.fn else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if p + r else 0.0

    @property
    def operational_recall(self) -> float:
        denom = self.tp + self.fn
        return (self.tp + self.fn_caught_in_review) / denom if denom else 0.0

    @property
    def retrieval_recall(self) -> float:
        """Ceiling for every resolver: a match never retrieved can never be merged."""
        return self.retrieved_expected / self.total_expected if self.total_expected else 0.0

    def summary(self) -> dict:
        return {
            "db": self.db,
            "resolver": self.resolver,
            "retrieval": self.retrieval,
            "queries": self.queries,
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            "fn_caught_in_review": self.fn_caught_in_review,
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "operational_recall": round(self.operational_recall, 4),
            "retrieval_recall": round(self.retrieval_recall, 4),
            "retrieval_recall_by_type": {
                tag: round(got / total, 4) for tag, (got, total) in sorted(self.by_type.items()) if total
            },
            "retrieval_latency_ms": self._percentiles(self.retrieval_ms),
            "scoring_latency_ms": self._percentiles(self.scoring_ms),
        }


def build_gt_map(ground_truth: list[dict]) -> dict[str, set[str]]:
    id_field = settings.schema.id_field
    gt_map = {}
    for item in ground_truth:
        q_code = str(item["query_id"][id_field])
        matches = {str(m[id_field]) for m in item["expected_matches"]}
        matches.discard(q_code)
        gt_map[q_code] = matches
    return gt_map


def type_tags(duplicate_type: str | None) -> list[str]:
    """'Fuzzy + CrossSystem' -> ['Fuzzy', 'CrossSystem']"""
    if not duplicate_type:
        return ["Unknown"]
    return [part.split("(")[0].strip() for part in duplicate_type.split(" + ")]


async def run_benchmark(
    db: DB,
    resolver: Resolver,
    embedder: Embedder,
    queries: list[dict],
    ground_truth: list[dict],
    params: SearchParams | None = None,
    failures_path: str | Path | None = None,
    max_logged_failures: int = 2000,
) -> BenchmarkResult:
    params = params or settings.search
    gt_map = build_gt_map(ground_truth)
    id_field = settings.schema.id_field
    type_map = {str(i["query_id"][id_field]): i.get("duplicate_type") for i in ground_truth}
    # report what actually ran, not what was asked for
    effective_mode = params.mode if db.supports_lexical else "vector"
    result = BenchmarkResult(db=db.name, resolver=resolver.name, retrieval=effective_mode)

    warmup = 20  # first queries pay for model load, JIT, cold caches — keep them out of the stats
    for i, query in enumerate(tqdm(queries, desc=f"{db.name}/{resolver.name}")):
        query_id = doc_id(query)
        expected = gt_map.get(query_id, set())
        result.queries += 1

        t0 = time.perf_counter()
        hits = search(db, embedder, query, params=params)
        if i >= warmup:
            result.retrieval_ms.append((time.perf_counter() - t0) * 1000)
        candidates = [h for h in hits if h.id != query_id]

        # retrieval-stage accounting, before rerank_k truncation and scoring
        retrieved = {h.id for h in candidates} & expected
        result.retrieved_expected += len(retrieved)
        result.total_expected += len(expected)
        for tag in type_tags(type_map.get(query_id)):
            got, total = result.by_type.get(tag, (0, 0))
            result.by_type[tag] = (got + len(retrieved), total + len(expected))

        candidates = candidates[: params.rerank_k]
        if not candidates:
            result.fn += len(expected)
            continue

        t0 = time.perf_counter()
        verdicts = await resolver.score(query, candidates)
        if i >= warmup:
            result.scoring_ms.append((time.perf_counter() - t0) * 1000)

        merged, review_queue = set(), set()
        for v in verdicts:
            if v.review:
                review_queue.add(v.candidate_id)
            elif v.score > resolver.threshold:
                merged.add(v.candidate_id)

        result.merged_pairs.extend((query_id, m) for m in merged)

        tp_set = merged & expected
        fp_set = merged - expected
        fn_set = expected - merged

        result.tp += len(tp_set)
        result.fp += len(fp_set)
        result.fn += len(fn_set)
        result.fn_caught_in_review += len(fn_set & review_queue)

        if len(result.failures) < max_logged_failures:
            verdict_map = {v.candidate_id: v for v in verdicts}
            retrieved_ids = {h.id for h in candidates}
            for bad_id in fp_set:
                result.failures.append(
                    {
                        "type": "FP",
                        "query_id": query_id,
                        "match_id": bad_id,
                        "score": verdict_map[bad_id].score,
                        "reason": verdict_map[bad_id].reason,
                    }
                )
            for miss_id in fn_set:
                v = verdict_map.get(miss_id)
                result.failures.append(
                    {
                        "type": "FN",
                        "query_id": query_id,
                        "match_id": miss_id,
                        "saved_by_review": miss_id in review_queue,
                        "score": v.score if v else None,
                        "reason": v.reason
                        if v
                        else ("not retrieved" if miss_id not in retrieved_ids else "no verdict"),
                    }
                )

    if failures_path:
        failures_path = Path(failures_path)
        failures_path.parent.mkdir(parents=True, exist_ok=True)
        with open(failures_path, "w", encoding="utf-8") as f:
            for rec in result.failures:
                f.write(json.dumps({**rec, "ts": datetime.now().isoformat()}, default=str) + "\n")

    return result
