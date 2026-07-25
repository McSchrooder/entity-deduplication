import pytest

torch = pytest.importorskip("torch")

from dedupe.training.model import PairClassifier, PairClassifierConfig  # noqa: E402


def test_forward_shapes():
    m = PairClassifier(dim=19, hidden=32)
    out = m(torch.rand(4, 19))
    assert out.shape == (4,)


def test_predict_in_unit_interval():
    m = PairClassifier(dim=19, hidden=32)
    p = m.predict(torch.rand(10, 19))
    assert ((p >= 0) & (p <= 1)).all()


@pytest.mark.parametrize("depth", [1, 3])
def test_depth_variants_forward(depth):
    m = PairClassifier(dim=8, hidden=16, depth=depth)
    assert m(torch.rand(5, 8)).shape == (5,)


def test_checkpoint_roundtrip_carries_config_features_temperature(tmp_path):
    m = PairClassifier(dim=8, hidden=16, depth=2, features=["a", "b"]).eval()
    m.temperature.fill_(1.7)
    x = torch.rand(6, 8)
    m.save(tmp_path / "m.pt")
    loaded = PairClassifier.load(tmp_path / "m.pt").eval()
    assert loaded.config == PairClassifierConfig(dim=8, hidden=16, depth=2)
    assert loaded.features == ["a", "b"]
    assert float(loaded.temperature) == pytest.approx(1.7)
    assert torch.allclose(loaded.predict(x), m.predict(x))


def test_temperature_scales_probabilities_not_ranking():
    m = PairClassifier(dim=8, hidden=16).eval()
    x = torch.rand(20, 8)
    cold = m.predict(x)
    m.temperature.fill_(5.0)
    hot = m.predict(x)
    assert torch.equal(torch.argsort(cold), torch.argsort(hot))
    assert (hot - 0.5).abs().max() < (cold - 0.5).abs().max()  # squashed toward 0.5
