from dedupe.resolver.features import anchor_class, feature_names, featurize


def test_names_and_values_align():
    names = feature_names()
    q = {"first_name": "Ivan", "last_name": "Horvat", "birth_date": "1980-05-05"}
    vec = featurize(q, dict(q))
    assert len(vec) == len(names)


def test_identical_pair_maxes_signals():
    q = {
        "first_name": "Ivan",
        "last_name": "Horvat",
        "birth_date": "1980-05-05",
        "email": "a@b.com",
        "city": "Split",
    }
    f = dict(zip(feature_names(), featurize(q, dict(q)), strict=True))
    assert f["name_token_set"] == 1.0
    assert f["birth_date_exact"] == 1.0
    assert f["birth_date_conflict"] == 0.0
    assert f["email_eq"] == 1.0
    assert f["city_sim"] == 1.0


def test_missing_fields_are_neutral_not_negative():
    q = {"first_name": "Ivan", "last_name": "Horvat"}
    f = dict(zip(feature_names(), featurize(q, dict(q)), strict=True))
    assert f["birth_date_missing"] == 1.0
    assert f["email_eq"] == 0.0
    assert f["email_present"] == 0.0


def test_anchor_classes():
    assert anchor_class("1980-05-05", "1980-05-05") == "exact"
    assert anchor_class("1980-05-12", "1980-12-05") == "swap"
    assert anchor_class("1980-05-05", "1981-05-05") == "year_off1"
    assert anchor_class("1980-05-05", "1990-05-05") == "year_typo"
    assert anchor_class("1980-05-05", "1993-06-06") == "conflict"
    assert anchor_class(None, "1980-05-05") == "missing"


def test_exclude_drops_named_features():
    full = feature_names()
    cut = feature_names(exclude=frozenset({"birth_date_exact"}))
    assert "birth_date_exact" in full
    assert "birth_date_exact" not in cut
    assert len(cut) == len(full) - 1
    vec = featurize({}, {}, exclude=frozenset({"birth_date_exact"}))
    assert len(vec) == len(cut)


def test_diacritics_folded_in_name_ratio():
    q = {"first_name": "Šimun", "last_name": "Babić"}
    c = {"first_name": "Simun", "last_name": "Babic"}
    f = dict(zip(feature_names(), featurize(q, c), strict=True))
    assert f["name_token_set"] == 1.0
