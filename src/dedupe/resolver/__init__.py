from dedupe.config import settings
from dedupe.resolver.base import Resolver, Verdict


class NullResolver:
    """Merges nothing, isolates retrieval_recall from resolving effects."""

    name = "none"
    threshold = 2.0

    async def score(self, query_doc, candidates):
        return [Verdict(candidate_id=h.id, score=0.0) for h in candidates]


def get_resolver(name: str | None = None, **kwargs) -> Resolver:
    name = name or settings.resolver
    if name == "none":
        return NullResolver()
    if name == "heuristic":
        from dedupe.resolver.heuristic import HeuristicResolver

        return HeuristicResolver(**kwargs)
    if name == "cross-encoder":
        from dedupe.resolver.cross_encoder import CrossEncoderResolver

        return CrossEncoderResolver(**kwargs)
    if name == "llm":
        from dedupe.resolver.llm import LLMResolver

        return LLMResolver(**kwargs)
    if name == "nn":
        from dedupe.resolver.nn import NNResolver

        return NNResolver(**kwargs)
    raise ValueError(f"unknown resolver: {name}")


__all__ = ["Resolver", "Verdict", "get_resolver"]
