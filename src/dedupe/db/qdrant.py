import hashlib

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from dedupe.config import settings
from dedupe.db.base import Hit
from dedupe.serialization import doc_id

COLLECTION = "dedupe_base"
EMBEDDING_DIM = 1024


def _point_id(code: str) -> str:
    return hashlib.md5(code.encode()).hexdigest()


class QdrantDB:
    name = "qdrant"
    supports_lexical = False  # no BM25; retrieval runs vector-only here

    def __init__(self, url: str | None = None):
        self.client = QdrantClient(url=url or settings.qdrant_url)

    def init(self) -> None:
        if not self.client.collection_exists(COLLECTION):
            self.client.create_collection(
                collection_name=COLLECTION,
                vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
            )

    def ingest(self, docs: list[dict], embeddings: list[list[float]]) -> int:
        points = [
            PointStruct(id=_point_id(doc_id(doc)), vector=emb, payload=doc)
            for doc, emb in zip(docs, embeddings, strict=False)
        ]
        for i in range(0, len(points), 512):
            self.client.upsert(collection_name=COLLECTION, points=points[i : i + 512])
        return len(points)

    def vector_search(self, vector: list[float], k: int) -> list[Hit]:
        res = self.client.query_points(collection_name=COLLECTION, query=vector, limit=k, with_payload=True)
        return [Hit(id=doc_id(p.payload), doc=p.payload, score=p.score) for p in res.points]

    def lexical_search(self, query_doc: dict, k: int) -> list[Hit]:
        return []

    def count(self) -> int:
        return self.client.count(collection_name=COLLECTION).count
