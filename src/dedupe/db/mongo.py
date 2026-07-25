import time

from pymongo import MongoClient, ReplaceOne

from dedupe.config import settings
from dedupe.db.base import Hit
from dedupe.serialization import doc_id

EMBEDDING_DIM = 1024


class MongoDB:
    name = "mongo"
    supports_lexical = True

    def __init__(self, uri: str | None = None):
        self.client = MongoClient(uri or settings.mongo_uri)
        self.coll = self.client[settings.mongo_db][settings.mongo_collection]
        self.vector_index = settings.mongo_vector_index
        self.lexical_index = settings.mongo_lexical_index
        self.params = settings.search
        self.lexical_fields = settings.schema.lexical_fields

    def init(self) -> None:
        db = self.coll.database
        if settings.mongo_collection not in db.list_collection_names():
            db.create_collection(settings.mongo_collection)
        existing = {ix["name"] for ix in self.coll.list_search_indexes()}
        if self.vector_index not in existing:
            self.coll.database.command(
                {
                    "createSearchIndexes": settings.mongo_collection,
                    "indexes": [
                        {
                            "name": self.vector_index,
                            "type": "vectorSearch",
                            "definition": {
                                "fields": [
                                    {
                                        "type": "vector",
                                        "path": "embedding",
                                        "numDimensions": EMBEDDING_DIM,
                                        "similarity": "cosine",
                                    }
                                ]
                            },
                        }
                    ],
                }
            )
        if self.lexical_index not in existing:
            self.coll.database.command(
                {
                    "createSearchIndexes": settings.mongo_collection,
                    "indexes": [
                        {
                            "name": self.lexical_index,
                            "definition": {"mappings": {"dynamic": True}},
                        }
                    ],
                }
            )
        # mongot builds indexes async, poll until queryable
        for _ in range(60):
            states = {ix["name"]: ix.get("status") for ix in self.coll.list_search_indexes()}
            if all(states.get(n) == "READY" for n in (self.vector_index, self.lexical_index)):
                return
            time.sleep(2)

    def ingest(self, docs: list[dict], embeddings: list[list[float]]) -> int:
        ops = []
        for doc, emb in zip(docs, embeddings, strict=False):
            record = {**doc, "embedding": emb}
            ops.append(ReplaceOne({"_id": doc["_id"]}, record, upsert=True))
        if not ops:
            return 0
        result = self.coll.bulk_write(ops, ordered=False)
        return result.upserted_count + result.modified_count

    def vector_search(self, vector: list[float], k: int) -> list[Hit]:
        pipeline = [
            {
                "$vectorSearch": {
                    "index": self.vector_index,
                    "path": "embedding",
                    "queryVector": vector,
                    "numCandidates": k * self.params.num_candidates_factor,
                    "limit": k,
                }
            },
            {"$project": {"embedding": 0}},
            {"$addFields": {"_score": {"$meta": "vectorSearchScore"}}},
        ]
        return [self._hit(d) for d in self.coll.aggregate(pipeline)]

    def lexical_search(self, query_doc: dict, k: int) -> list[Hit]:
        clauses = [
            {"text": {"query": v, "path": f, "fuzzy": {"maxEdits": self.params.fuzzy_max_edits}}}
            for f in self.lexical_fields
            if (v := query_doc.get(f))
        ]
        if not clauses:
            return []
        pipeline = [
            {
                "$search": {
                    "index": self.lexical_index,
                    "compound": {"should": clauses, "minimumShouldMatch": 1},
                }
            },
            {"$limit": k},
            {"$project": {"embedding": 0}},
            {"$addFields": {"_score": {"$meta": "searchScore"}}},
        ]
        return [self._hit(d) for d in self.coll.aggregate(pipeline)]

    def count(self) -> int:
        return self.coll.count_documents({})

    @staticmethod
    def _hit(d: dict) -> Hit:
        score = d.pop("_score", 0.0)
        return Hit(id=doc_id(d), doc=d, score=score)
