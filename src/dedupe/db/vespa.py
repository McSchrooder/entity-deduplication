"""The application package (services.xml + .sd) is generated from the
field schema and deployed over the config-server HTTP API in init()"""

import io
import time
import zipfile

import requests
from vespa.application import Vespa

from dedupe.config import settings
from dedupe.db.base import Hit
from dedupe.serialization import doc_id

SCHEMA = "deduplication_primary"
EMBEDDING_DIM = 1024

SERVICES_XML = """<?xml version="1.0" encoding="UTF-8"?>
<services version="1.0">
    <container id="dedupe_container" version="1.0">
        <search/>
        <document-api/>
        <nodes>
            <node hostalias="node1"/>
        </nodes>
    </container>
    <content id="dedupe_content" version="1.0">
        <redundancy>1</redundancy>
        <documents>
            <document type="{schema}" mode="index"/>
        </documents>
        <nodes>
            <node distribution-key="0" hostalias="node1"/>
        </nodes>
    </content>
</services>
"""


def render_sd() -> str:
    lines = [f"schema {SCHEMA} {{", f"    document {SCHEMA} {{"]
    for spec in settings.schema.fields:
        lines.append(f"        field {spec.name} type string {{")
        if spec.role in ("id", "label"):
            lines.append("            indexing: summary | attribute")
        elif spec.lexical:
            lines.append("            indexing: summary | index")
            lines.append("            index: enable-bm25")
        else:
            lines.append("            indexing: summary")
        lines.append("        }")
    lines += [
        f"        field embedding type tensor<float>(x[{EMBEDDING_DIM}]) {{",
        "            indexing: summary | attribute",
        "            attribute {",
        "                distance-metric: angular",
        "            }",
        "        }",
        "    }",
        "",
        "    fieldset default {",
        f"        fields: {', '.join(settings.schema.lexical_fields)}",
        "    }",
        "",
        "    rank-profile vector {",
        "        inputs {",
        f"            query(q) tensor<float>(x[{EMBEDDING_DIM}])",
        "        }",
        "        first-phase {",
        "            expression: closeness(field, embedding)",
        "        }",
        "    }",
        "",
        "    rank-profile lexical {",
        "        first-phase {",
        "            expression: nativeRank",
        "        }",
        "    }",
        "}",
    ]
    return "\n".join(lines) + "\n"


def app_package() -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("services.xml", SERVICES_XML.format(schema=SCHEMA))
        z.writestr(f"schemas/{SCHEMA}.sd", render_sd())
    return buf.getvalue()


class VespaDB:
    name = "vespa"
    supports_lexical = True

    def __init__(self, url: str | None = None):
        base = url or settings.vespa_url
        host, _, port = base.rpartition(":")
        self.app = Vespa(url=host or base, port=int(port) if port.isdigit() else 8080)
        self.lexical_fields = settings.schema.lexical_fields
        self.schema_fields = tuple(f.name for f in settings.schema.fields)

    def init(self) -> None:
        resp = requests.post(
            f"{settings.vespa_config_url}/application/v2/tenant/default/prepareandactivate",
            data=app_package(),
            headers={"Content-Type": "application/zip"},
            timeout=120,
        )
        resp.raise_for_status()
        for _ in range(60):
            try:
                if self.app.get_application_status() is not None:
                    return
            except Exception:  # container port not up yet
                pass
            time.sleep(2)
        raise ConnectionError("vespa deployed but container port never became ready")

    def ingest(self, docs: list[dict], embeddings: list[list[float]]) -> int:
        n = 0
        for doc, emb in zip(docs, embeddings, strict=False):
            fields = {k: doc[k] for k in self.schema_fields if doc.get(k) is not None}
            fields["embedding"] = emb
            self.app.feed_data_point(schema=SCHEMA, data_id=doc_id(doc), fields=fields)
            n += 1
        return n

    def vector_search(self, vector: list[float], k: int) -> list[Hit]:
        resp = self.app.query(
            body={
                "yql": f"select * from sources {SCHEMA} where "
                f"{{targetHits:{k}}}nearestNeighbor(embedding, q)",
                "input.query(q)": vector,
                "ranking.profile": "vector",
                "hits": k,
            }
        )
        return [self._hit(h) for h in resp.hits]

    def lexical_search(self, query_doc: dict, k: int) -> list[Hit]:
        terms = [str(v) for f in self.lexical_fields if (v := query_doc.get(f))]
        if not terms:
            return []
        resp = self.app.query(
            body={
                "yql": f"select * from sources {SCHEMA} where userQuery()",
                "query": " ".join(terms),
                "type": "weakAnd",
                "ranking.profile": "lexical",
                "hits": k,
            }
        )
        return [self._hit(h) for h in resp.hits]

    def count(self) -> int:
        resp = self.app.query(
            body={
                "yql": f"select * from sources {SCHEMA} where true",
                "hits": 0,
            }
        )
        return resp.json.get("root", {}).get("fields", {}).get("totalCount", 0)

    @staticmethod
    def _hit(h: dict) -> Hit:
        fields = h.get("fields", {})
        return Hit(id=doc_id(fields), doc=fields, score=h.get("relevance", 0.0))
