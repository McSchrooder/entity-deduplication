"""Schema-driven pair comparison features."""

from rapidfuzz import fuzz

from dedupe.config import FieldSchema, settings
from dedupe.resolver.heuristic import _clean, _parse_date

ANCHOR_CLASSES = ("missing", "exact", "swap", "year_off1", "year_typo", "conflict")


def anchor_class(a, b) -> str:
    """Mirror of the heuristic's hard-anchor logic, as a class instead of points."""
    d1, d2 = _parse_date(a), _parse_date(b)
    if d1 and d2:
        if d1 == d2:
            return "exact"
        if d1.year == d2.year and d1.month == d2.day and d1.day == d2.month:
            return "swap"
        if d1.month == d2.month and d1.day == d2.day:
            diff = abs(d1.year - d2.year)
            if diff <= 1:
                return "year_off1"
            if diff in (9, 10):
                return "year_typo"
        return "conflict"
    ca, cb = _clean(a), _clean(b)
    if not ca or not cb:
        return "missing"
    return "exact" if ca == cb else "conflict"


def _pairs(q: dict, d: dict, schema: FieldSchema) -> list[tuple[str, float]]:
    out: list[tuple[str, float]] = []

    name_fields = [f.name for f in schema.by_role("name")]
    qn = _clean(" ".join(str(q.get(f) or "") for f in name_fields))
    dn = _clean(" ".join(str(d.get(f) or "") for f in name_fields))
    both = bool(qn and dn)
    out.append(("name_token_set", fuzz.token_set_ratio(qn, dn) / 100 if both else 0.0))
    out.append(("name_ratio", fuzz.ratio(qn, dn) / 100 if both else 0.0))
    out.append(("name_present", float(both)))

    for spec in schema.by_role("hard_anchor"):
        cls = anchor_class(q.get(spec.name), d.get(spec.name))
        out.extend((f"{spec.name}_{c}", float(cls == c)) for c in ANCHOR_CLASSES)

    for spec in schema.by_role("soft_anchor"):
        qv, dv = _clean(q.get(spec.name)), _clean(d.get(spec.name))
        both = bool(qv and dv)
        out.append((f"{spec.name}_eq", float(both and qv == dv)))
        out.append((f"{spec.name}_present", float(both)))

    for spec in schema.by_role("context"):
        qv, dv = _clean(q.get(spec.name)), _clean(d.get(spec.name))
        both = bool(qv and dv)
        sim = (fuzz.ratio(qv, dv) / 100 if spec.fuzzy else float(qv == dv)) if both else 0.0
        out.append((f"{spec.name}_sim", sim))
        out.append((f"{spec.name}_present", float(both)))

    return out


def feature_names(schema: FieldSchema | None = None, exclude: frozenset[str] = frozenset()) -> list[str]:
    schema = schema or settings.schema
    return [n for n, _v in _pairs({}, {}, schema) if n not in exclude]


def featurize(
    q: dict, d: dict, schema: FieldSchema | None = None, exclude: frozenset[str] = frozenset()
) -> list[float]:
    schema = schema or settings.schema
    return [v for n, v in _pairs(q, d, schema) if n not in exclude]
