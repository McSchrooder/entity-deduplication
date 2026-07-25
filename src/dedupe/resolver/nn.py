"""Learned resolver: schema-driven comparison features -> calibrated MLP"""

from pathlib import Path

import torch

from dedupe.config import settings
from dedupe.db.base import Hit
from dedupe.resolver.base import Verdict
from dedupe.resolver.features import feature_names, featurize
from dedupe.training.model import PairClassifier


class NNResolver:
    name = "nn"

    def __init__(self, threshold: float = 0.5, checkpoint: str | None = None):
        self.threshold = threshold
        self.checkpoint = Path(checkpoint or Path(settings.models_dir) / "pair_classifier.pt")
        self._model = None
        self._exclude: frozenset[str] = frozenset()

    @property
    def model(self) -> PairClassifier:
        if self._model is None:
            m = PairClassifier.load(self.checkpoint).eval()
            current = feature_names()
            if m.features and m.features != current:
                # trained with an exclude list is fine; a schema drift is not
                missing = set(m.features) - set(current)
                if missing:
                    raise RuntimeError(
                        f"checkpoint expects features absent from the schema: {sorted(missing)} — retrain"
                    )
                self._exclude = frozenset(set(current) - set(m.features))
            self._model = m
        return self._model

    async def score(self, query_doc: dict, candidates: list[Hit]) -> list[Verdict]:
        if not candidates:
            return []
        model = self.model
        x = torch.tensor(
            [featurize(query_doc, h.doc, exclude=self._exclude) for h in candidates],
            dtype=torch.float32,
        )
        probs = model.predict(x)
        return [Verdict(candidate_id=h.id, score=float(p)) for h, p in zip(candidates, probs, strict=True)]
