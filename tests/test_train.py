import pytest

torch = pytest.importorskip("torch")

from dedupe.training.train import _split  # noqa: E402


def test_split_never_divides_a_group():
    n_groups, pairs_per = 50, 8
    groups = torch.arange(n_groups).repeat_interleave(pairs_per)
    x = groups.float().unsqueeze(1)  # feature = group id, so sides are inspectable
    y = torch.zeros(len(groups))

    x_train, _y1, x_val, _y2, _g = _split(x, y, groups, val_frac=0.1, seed=42)
    train_gids = set(x_train.squeeze(1).tolist())
    val_gids = set(x_val.squeeze(1).tolist())
    assert train_gids.isdisjoint(val_gids)
    assert len(x_train) + len(x_val) == len(groups)
    assert len(x_val) >= int(len(groups) * 0.1)


def test_split_is_seeded():
    groups = torch.arange(20).repeat_interleave(4)
    x = groups.float().unsqueeze(1)
    y = torch.zeros(len(groups))
    a = _split(x, y, groups, 0.2, seed=7)[2]
    b = _split(x, y, groups, 0.2, seed=7)[2]
    assert torch.equal(a, b)
