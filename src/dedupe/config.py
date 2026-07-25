"""Central config.

which db, which resolver, every retrieval knob, and the field schema.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv

load_dotenv()

# repo root, so default paths work from any working directory
_ROOT = Path(__file__).resolve().parents[2]


def _path(env: str, default: str) -> str:
    return os.getenv(env) or str(_ROOT / default)


Role = Literal["id", "label", "name", "hard_anchor", "soft_anchor", "context", "detail"]


@dataclass(frozen=True)
class FieldSpec:
    """One field's contract with the pipeline.

    Roles carry evidence semantics for the heuristic resolver, hard anchors can veto
    (mismatch is counter-evidence), soft anchors only reward a match, context nudges,
    detail carries no evidence at all. id/label are meta-signal and excluded from
    embedding text and never scored.
    """

    name: str
    role: Role = "detail"
    source: str | None = None  # raw export column this maps from
    label: str | None = None  # human-readable label for embedding text and LLM prompts
    lexical: bool = False  # part of the text index
    priority: bool = False  # rendered before the alphabetical tail in stringify
    weight: float = 0.0  # evidence points (soft_anchor / context)
    fuzzy: bool = False  # context matches by ratio instead of equality


@dataclass(frozen=True)
class FieldSchema:
    fields: tuple[FieldSpec, ...]

    def __post_init__(self):
        roles = [f.role for f in self.fields]
        if roles.count("id") != 1:
            raise ValueError("schema needs exactly one id field")
        if roles.count("name") < 1:
            raise ValueError("schema needs at least one name field")
        if roles.count("label") > 1:
            raise ValueError("schema allows at most one label field")

    def __getitem__(self, name: str) -> FieldSpec:
        for f in self.fields:
            if f.name == name:
                return f
        raise KeyError(name)

    def by_role(self, *roles: Role) -> tuple[FieldSpec, ...]:
        return tuple(f for f in self.fields if f.role in roles)

    @property
    def id_field(self) -> str:
        return self.by_role("id")[0].name

    @property
    def label_field(self) -> str | None:
        specs = self.by_role("label")
        return specs[0].name if specs else None

    @property
    def lexical_fields(self) -> tuple[str, ...]:
        return tuple(f.name for f in self.fields if f.lexical)

    @property
    def priority_fields(self) -> tuple[str, ...]:
        return tuple(f.name for f in self.fields if f.priority)

    @property
    def source_map(self) -> dict[str, str]:
        """raw export column -> canonical field name, for the dataset loader"""
        return {f.source: f.name for f in self.fields if f.source}

    def display_label(self, name: str) -> str:
        for f in self.fields:
            if f.name == name:
                return f.label or name
        return name


GUEST_SCHEMA = FieldSchema(
    fields=(
        FieldSpec("first_name", "name", source="FIRST_NAME", label="First Name", lexical=True, priority=True),
        FieldSpec("last_name", "name", source="LAST_NAME", label="Last Name", lexical=True, priority=True),
        FieldSpec("middle_name", source="MIDDLE_NAME", label="Middle Name", priority=True),
        FieldSpec("birth_date", "hard_anchor", source="BIRTH_DATE", label="Date of Birth", priority=True),
        FieldSpec("gender", source="GENDER", label="Gender", priority=True),
        FieldSpec("birth_place", source="BIRTH_PLACE", label="Place of Birth", priority=True),
        FieldSpec("nationality", source="CITIZENSHIP", label="Nationality Code", lexical=True, priority=True),
        FieldSpec(
            "email",
            "soft_anchor",
            source="EMAIL",
            label="Email Address",
            lexical=True,
            priority=True,
            weight=20,
        ),
        FieldSpec(
            "phone", "soft_anchor", source="PHONE_NUMBER", label="Phone Number", priority=True, weight=20
        ),
        FieldSpec("city", "context", source="CITY", label="City", lexical=True, weight=10, fuzzy=True),
        FieldSpec("zip_code", "soft_anchor", source="ZIP_CODE", label="Zip Code", lexical=True, weight=15),
        FieldSpec("country", "context", source="COUNTRY", label="Country Code", weight=5),
        FieldSpec("source", source="SOURCE", label="Source System"),
        FieldSpec("code", "id", source="CODE", label="Unique Guest Code"),
        FieldSpec("meta_unified_id", "label"),
    )
)


@dataclass(frozen=True)
class SearchParams:
    """Every retrieval knob."""

    mode: str = field(default_factory=lambda: os.getenv("DEDUPE_RETRIEVAL_MODE", "hybrid"))
    retrieval_k: int = field(default_factory=lambda: int(os.getenv("DEDUPE_RETRIEVAL_K", "150")))
    rerank_k: int = field(default_factory=lambda: int(os.getenv("DEDUPE_RERANK_K", "50")))
    rrf_k: int = field(default_factory=lambda: int(os.getenv("DEDUPE_RRF_K", "60")))
    num_candidates_factor: int = field(
        default_factory=lambda: int(os.getenv("DEDUPE_NUM_CANDIDATES_FACTOR", "5"))
    )
    fuzzy_max_edits: int = field(default_factory=lambda: int(os.getenv("DEDUPE_FUZZY_MAX_EDITS", "1")))


@dataclass(frozen=True)
class Settings:
    db: str = field(default_factory=lambda: os.getenv("DEDUPE_DB", "mongo"))
    resolver: str = field(default_factory=lambda: os.getenv("DEDUPE_RESOLVER", "heuristic"))

    search: SearchParams = field(default_factory=SearchParams)
    schema: FieldSchema = GUEST_SCHEMA

    mongo_uri: str = field(
        default_factory=lambda: os.getenv("MONGO_URI", "mongodb://localhost:27018/?directConnection=true")
    )
    mongo_db: str = field(default_factory=lambda: os.getenv("MONGO_DB", "dedupe_base"))
    mongo_collection: str = field(default_factory=lambda: os.getenv("MONGO_COLLECTION", "documents"))
    mongo_vector_index: str = field(
        default_factory=lambda: os.getenv("MONGO_VECTOR_INDEX", "bgem3_vector_index")
    )
    mongo_lexical_index: str = field(
        default_factory=lambda: os.getenv("MONGO_LEXICAL_INDEX", "dynamic_text_index")
    )
    qdrant_url: str = field(default_factory=lambda: os.getenv("QDRANT_URL", "http://localhost:6333"))
    vespa_url: str = field(default_factory=lambda: os.getenv("VESPA_URL", "http://localhost:8080"))
    vespa_config_url: str = field(
        default_factory=lambda: os.getenv("VESPA_CONFIG_URL", "http://localhost:19071")
    )

    llm_base_url: str = field(default_factory=lambda: os.getenv("LLM_BASE_URL", "http://localhost:8000/v1"))
    llm_model: str = field(default_factory=lambda: os.getenv("LLM_MODEL", "Qwen/Qwen3-8B-FP8"))
    llm_api_key: str = field(default_factory=lambda: os.getenv("LLM_API_KEY", "empty"))

    embedding_model: str = "BAAI/bge-m3"
    reranker_model: str = "BAAI/bge-reranker-v2-m3"

    data_dir: str = field(default_factory=lambda: _path("DEDUPE_DATA_DIR", "data"))
    models_dir: str = field(default_factory=lambda: _path("DEDUPE_MODELS_DIR", "models"))

    guests_parquet: str = field(
        default_factory=lambda: _path("DEDUPE_GUESTS_PARQUET", "data/real/guests.parquet")
    )
    person_map_parquet: str = field(
        default_factory=lambda: _path("DEDUPE_PERSON_MAP_PARQUET", "data/real/person_map.parquet")
    )


settings = Settings()
