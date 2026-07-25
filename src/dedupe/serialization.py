"""Canonical doc -> text rendering. Every model (embedder, cross-encoder, LLM) sees
profiles through stringify, so the field schema is the single source of what renders,
in what order, under what label. Changing the schema invalidates embedding caches deliberately."""

from dedupe.config import settings

# runtime/bookkeeping keys that never belong in model-visible text
_RUNTIME_FIELDS = {
    "_id",
    "unified_guest_id",
    "mutation_type",
    "update_date",
    "score",
    "rrf_score",
    "rerank_score",
    "embedding",
    "llm_review_needed",
    "llm_reason",
}

_NULLISH = {"none", "null", "", "n/a", "unknown"}


def _ignored() -> set[str]:
    ignored = set(_RUNTIME_FIELDS)
    ignored.add(settings.schema.id_field)
    if settings.schema.label_field:
        ignored.add(settings.schema.label_field)
    return ignored


def clean_value(key: str, value) -> str | None:
    if value is None:
        return None
    s = str(value).strip()
    if s.lower() in _NULLISH:
        return None
    if "phone" in key.lower():
        digits = "".join(c for c in s if c.isdigit() or c == "+")
        return digits if len(digits) > 3 else None
    if "email" in key.lower():
        return s.lower()
    return s


def stringify(doc: dict, max_fields: int = 10) -> str:
    """Flatten a profile into 'Label: value | ...' text, priority fields first,
    the rest alphabetically. Unknown fields render under their raw key."""
    ignored = _ignored()
    cleaned = {}
    for k, v in doc.items():
        if k in ignored:
            continue
        c = clean_value(k, v)
        if c:
            cleaned[k] = c

    parts = []
    for key in settings.schema.priority_fields:
        if len(parts) >= max_fields:
            break
        if key in cleaned:
            parts.append(f"{settings.schema.display_label(key)}: {cleaned.pop(key)}")
    for key in sorted(cleaned):
        if len(parts) >= max_fields:
            break
        parts.append(f"{settings.schema.display_label(key)}: {cleaned[key]}")

    return " | ".join(parts)


def doc_id(doc: dict) -> str:
    """Stable string id from either {'_id': {code: ...}} or a flat id field."""
    id_field = settings.schema.id_field
    _id = doc.get("_id", doc.get(id_field))
    if isinstance(_id, dict):
        return str(_id.get(id_field))
    return str(_id)
