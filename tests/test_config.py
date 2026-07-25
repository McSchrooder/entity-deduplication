import pytest

from dedupe.config import FieldSchema, FieldSpec, settings


def test_schema_requires_exactly_one_id():
    with pytest.raises(ValueError, match="exactly one id"):
        FieldSchema(fields=(FieldSpec("first_name", "name"),))
    with pytest.raises(ValueError, match="exactly one id"):
        FieldSchema(fields=(FieldSpec("a", "id"), FieldSpec("b", "id"), FieldSpec("n", "name")))


def test_schema_requires_a_name_field():
    with pytest.raises(ValueError, match="name field"):
        FieldSchema(fields=(FieldSpec("code", "id"),))


def test_minimal_schema_is_valid():
    s = FieldSchema(fields=(FieldSpec("code", "id"), FieldSpec("full_name", "name")))
    assert s.id_field == "code"
    assert s.label_field is None
    assert s.lexical_fields == ()


def test_guest_schema_derivations():
    s = settings.schema
    assert s.id_field == "code"
    assert s.label_field == "meta_unified_id"
    assert "first_name" in s.lexical_fields
    assert s.priority_fields[0] == "first_name"
    assert s.source_map["CITIZENSHIP"] == "nationality"
    assert s.display_label("birth_date") == "Date of Birth"
    assert s.display_label("unknown_key") == "unknown_key"


def test_roles_carry_evidence_weights():
    soft = {f.name: f.weight for f in settings.schema.by_role("soft_anchor")}
    assert soft == {"email": 20, "phone": 20, "zip_code": 15}
    assert [f.name for f in settings.schema.by_role("hard_anchor")] == ["birth_date"]
