"""Deterministic resolver: fuzzy name baseline + role-weighted field evidence.
Evidence semantics come from the schema,
hard anchors can veto, soft anchors only reward a match, context nudges."""

from datetime import datetime

from rapidfuzz import fuzz
from unidecode import unidecode

from dedupe.config import FieldSchema, settings
from dedupe.db.base import Hit
from dedupe.resolver.base import Verdict


def _clean(text) -> str:
    if not text:
        return ""
    return unidecode(str(text).lower()).strip()


def _parse_date(d):
    if not d:
        return None
    if isinstance(d, datetime):
        return d
    try:
        return datetime.strptime(str(d)[:10], "%Y-%m-%d")
    except ValueError:
        return None


def score_dob(a, b) -> tuple[int, str | None]:
    """Fuzzy DOB comparison tolerant of the mutations real front desks produce."""
    d1, d2 = _parse_date(a), _parse_date(b)
    if not d1 or not d2:
        return 0, None
    if d1 == d2:
        return 25, "DOB exact"
    if d1.year == d2.year and d1.month == d2.day and d1.day == d2.month:
        return 15, "DOB day/month swap"
    if d1.month == d2.month and d1.day == d2.day:
        diff = abs(d1.year - d2.year)
        if diff <= 1:
            return 10, "DOB year off-by-one"
        if diff in (9, 10):
            return 5, "DOB year typo"
    return -50, f"DOB mismatch {d1.date()} vs {d2.date()}"


def score_hard_anchor(name: str, a, b) -> tuple[int, str | None]:
    if _parse_date(a) or _parse_date(b):
        return score_dob(a, b)
    ca, cb = _clean(a), _clean(b)
    if not ca or not cb:
        return 0, None
    if ca == cb:
        return 25, f"{name} exact"
    return -50, f"{name} mismatch"


class HeuristicResolver:
    name = "heuristic"

    def __init__(self, threshold: float = 0.90, review_band: float = 0.85, schema: FieldSchema | None = None):
        self.threshold = threshold
        self.review_band = review_band
        self.schema = schema or settings.schema

    def _score_pair(self, q: dict, c: dict) -> tuple[float, bool, str]:
        reasons = []

        name_fields = [f.name for f in self.schema.by_role("name")]
        q_name = _clean(" ".join(str(q.get(f, "")) for f in name_fields))
        c_name = _clean(" ".join(str(c.get(f, "")) for f in name_fields))
        # token_set_ratio: order-insensitive, duplicate-insensitive, noise-tolerant
        score = float(fuzz.token_set_ratio(q_name, c_name))
        reasons.append(f"name {score:.0f}")

        for spec in self.schema.by_role("hard_anchor"):
            delta, reason = score_hard_anchor(spec.name, q.get(spec.name), c.get(spec.name))
            score += delta
            if reason:
                reasons.append(reason)

        for spec in self.schema.by_role("soft_anchor"):
            if (qv := _clean(q.get(spec.name))) and qv == _clean(c.get(spec.name)):
                score += spec.weight
                reasons.append(spec.name)

        for spec in self.schema.by_role("context"):
            qv, cv = _clean(q.get(spec.name)), _clean(c.get(spec.name))
            if not qv or not cv:
                continue
            if fuzz.ratio(qv, cv) > 85 if spec.fuzzy else qv == cv:
                score += spec.weight
                reasons.append(spec.name)

        final = min(100, max(0, score)) / 100.0
        review = self.review_band <= final < self.threshold
        return final, review, ", ".join(reasons)

    async def score(self, query_doc: dict, candidates: list[Hit]) -> list[Verdict]:
        return [
            Verdict(candidate_id=h.id, score=s, review=r, reason=reason)
            for h in candidates
            for s, r, reason in [self._score_pair(query_doc, h.doc)]
        ]
