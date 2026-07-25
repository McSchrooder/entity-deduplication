import torch

from dedupe.config import settings
from dedupe.db.base import Hit
from dedupe.resolver.base import Verdict
from dedupe.serialization import stringify


class CrossEncoderResolver:
    name = "cross-encoder"

    # raw transformers instead of FlagEmbedding's FlagReranker — its slow-tokenizer
    # path broke against current transformers
    def __init__(self, threshold: float = 0.70, model_name: str | None = None, batch_size: int = 64):
        self.threshold = threshold
        self.model_name = model_name or settings.reranker_model
        self.batch_size = batch_size
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = None
        self._tokenizer = None

    def _load(self):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = (
            AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            )
            .to(self.device)
            .eval()
        )

    async def score(self, query_doc: dict, candidates: list[Hit]) -> list[Verdict]:
        if not candidates:
            return []
        if self._model is None:
            self._load()

        query_text = stringify(query_doc)
        pairs = [(query_text, stringify(h.doc)) for h in candidates]

        scores: list[float] = []
        with torch.no_grad():
            for i in range(0, len(pairs), self.batch_size):
                batch = pairs[i : i + self.batch_size]
                inputs = self._tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt",
                ).to(self.device)
                logits = self._model(**inputs).logits.view(-1).float()
                scores.extend(torch.sigmoid(logits).cpu().tolist())

        return [Verdict(candidate_id=h.id, score=s) for h, s in zip(candidates, scores, strict=True)]
