from dedupe.config import settings
from dedupe.db.base import DB, Hit, rrf_fuse, search


def get_db(name: str | None = None) -> DB:
    name = name or settings.db
    if name == "mongo":
        from dedupe.db.mongo import MongoDB

        return MongoDB()
    if name == "qdrant":
        from dedupe.db.qdrant import QdrantDB

        return QdrantDB()
    if name == "vespa":
        from dedupe.db.vespa import VespaDB

        return VespaDB()
    raise ValueError(f"unknown db: {name}")


__all__ = ["DB", "Hit", "get_db", "rrf_fuse", "search"]
