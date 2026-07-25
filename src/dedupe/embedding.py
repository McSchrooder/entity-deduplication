from dedupe.config import settings
from dedupe.serialization import stringify


class Embedder:
    """BGE-M3 dense embeddings. Lazy-loaded."""

    def __init__(self, model_name: str | None = None, use_fp16: bool = True):
        self.model_name = model_name or settings.embedding_model
        self.use_fp16 = use_fp16
        self._model = None

    @property
    def model(self):
        if self._model is None:
            import FlagEmbedding.inference.embedder.encoder_only.m3 as m3
            from FlagEmbedding import BGEM3FlagModel
            from tqdm import tqdm, trange

            # m3 passes disable=... explicitly to its bars, so a partial default
            # loses — force the kwarg after the caller's
            def _quiet(bar):
                def wrapped(*args, **kwargs):
                    kwargs["disable"] = True
                    return bar(*args, **kwargs)

                return wrapped

            m3.tqdm = _quiet(tqdm)
            m3.trange = _quiet(trange)
            self._model = BGEM3FlagModel(self.model_name, use_fp16=self.use_fp16)
        return self._model

    def encode_texts(self, texts: list[str], batch_size: int = 32) -> list[list[float]]:
        out = self.model.encode(
            texts, batch_size=batch_size, return_dense=True, return_sparse=False, return_colbert_vecs=False
        )
        return [v.tolist() for v in out["dense_vecs"]]

    def encode_doc(self, doc: dict) -> list[float]:
        return self.encode_texts([stringify(doc)])[0]
