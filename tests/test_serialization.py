from dedupe.serialization import clean_value, doc_id, stringify


def test_stringify_priority_and_labels():
    doc = {
        "first_name": "Ana",
        "last_name": "Horvat",
        "city": "Zagreb",
        "birth_date": "1990-01-01",
        "embedding": [0.1],
        "_id": {"code": "x"},
    }
    s = stringify(doc)
    assert s.startswith("First Name: Ana | Last Name: Horvat")
    assert "Date of Birth: 1990-01-01" in s
    assert "embedding" not in s


def test_stringify_respects_max_fields():
    doc = {f"field_{i}": f"v{i}" for i in range(20)}
    assert stringify(doc, max_fields=5).count("|") == 4


def test_clean_value_nullish():
    for v in (None, "", "None", "null", "N/A", "unknown"):
        assert clean_value("city", v) is None


def test_clean_value_phone_email():
    assert clean_value("phone", "+385 (91) 123-456") == "+38591123456"
    assert clean_value("phone", "12") is None
    assert clean_value("email", "  Ana.H@Example.COM ") == "ana.h@example.com"


def test_doc_id_variants():
    assert doc_id({"_id": {"code": "abc", "source": "SYSTEM_A"}}) == "abc"
    assert doc_id({"_id": 42}) == "42"
    assert doc_id({"code": "z"}) == "z"
