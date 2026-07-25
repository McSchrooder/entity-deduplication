"""Train PairClassifier on mined doc pairs."""

import json
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

from dedupe.resolver.features import feature_names, featurize
from dedupe.training.model import PairClassifier, PairClassifierConfig


def load_pairs(pairs_path: str | Path) -> tuple[list[tuple[dict, dict]], list[int], list[int]]:
    """Explode query/pos/neg records into labeled (query_doc, cand_doc) pairs.
    Also returns a group id per pair (the record it came from) so splits can keep
    all of a query's pairs on one side."""
    pairs, labels, groups = [], [], []
    with open(pairs_path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            r = json.loads(line)
            for pos in r["pos"]:
                pairs.append((r["query"], pos))
                labels.append(1)
                groups.append(i)
            for neg in r["neg"]:
                pairs.append((r["query"], neg))
                labels.append(0)
                groups.append(i)
    return pairs, labels, groups


def load_features(
    pairs_path: str | Path, exclude: frozenset[str] = frozenset()
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    pairs, labels, groups = load_pairs(pairs_path)
    x = np.array([featurize(q, d, exclude=exclude) for q, d in pairs], dtype=np.float32)
    return torch.from_numpy(x), torch.tensor(labels, dtype=torch.float32), torch.tensor(groups)


def focal_loss(logits: torch.Tensor, y: torch.Tensor, gamma: float) -> torch.Tensor:
    bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, y, reduction="none")
    p_t = torch.exp(-bce)  # prob assigned to the true class
    return ((1 - p_t) ** gamma * bce).mean()


def fit_temperature(logits: torch.Tensor, y: torch.Tensor) -> float:
    """Single-parameter temperature scaling on held-out logits."""
    log_t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=50)

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits / log_t.exp(), y)
        loss.backward()
        return loss

    opt.step(closure)
    return float(log_t.exp())


def _split(
    x: torch.Tensor, y: torch.Tensor, groups: torch.Tensor, val_frac: float, seed: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Generator]:
    """Group-wise split: all pairs sharing a query land on the same side."""
    g = torch.Generator().manual_seed(seed)
    uniq = torch.unique(groups)
    order = uniq[torch.randperm(len(uniq), generator=g)]
    cum = torch.bincount(groups)[order].cumsum(0)
    n_val_groups = int(torch.searchsorted(cum, int(len(y) * val_frac))) + 1
    val_mask = torch.isin(groups, order[:n_val_groups])
    return x[~val_mask], y[~val_mask], x[val_mask], y[val_mask], g


def train(
    pairs_path: str | Path,
    out_dir: str | Path = "models",
    config: PairClassifierConfig | None = None,
    exclude: frozenset[str] = frozenset(),
    epochs: int = 30,
    batch_size: int = 256,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    focal_gamma: float = 0.0,
    val_frac: float = 0.1,
    seed: int = 42,
    save: bool = True,
    epoch_callback: Callable[[int, float], bool] | None = None,
    data: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
) -> dict:
    """Returns {"best_auc", "temperature", "pairs"}. epoch_callback(epoch, val_auc) -> True
    stops early (Optuna pruning). Pass `data` to reuse tensors across tuning trials."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    names = feature_names(exclude=exclude)
    config = config or PairClassifierConfig(dim=len(names))

    x_t, y_t, groups = data if data is not None else load_features(pairs_path, exclude)
    x_train, y_train, x_val, y_val, g = _split(x_t, y_t, groups, val_frac, seed)
    train_dl = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True, generator=g)

    model = PairClassifier(config, features=names)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * len(train_dl))

    def loss_fn(logits, yb):
        if focal_gamma > 0:
            return focal_loss(logits, yb, focal_gamma)
        return torch.nn.functional.binary_cross_entropy_with_logits(logits, yb)

    best_auc, best_state = 0.0, None
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        for xb, yb in train_dl:
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            sched.step()
            total += loss.item() * len(yb)

        model.eval()
        with torch.no_grad():
            val_logits = model(x_val)
        auc = roc_auc_score(y_val.numpy(), val_logits.numpy())
        print(f"epoch {epoch}: loss={total / len(y_train):.4f} val_auc={auc:.4f}")

        if auc > best_auc:
            best_auc = auc
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if epoch_callback and epoch_callback(epoch, auc):
            break

    temperature = 1.0
    if best_state is not None:
        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            val_logits = model(x_val)
        temperature = fit_temperature(val_logits, y_val)
        model.temperature.fill_(temperature)
        if save:
            model.save(out_dir / "pair_classifier.pt")
            print(f"best val_auc={best_auc:.4f} T={temperature:.3f} -> {out_dir / 'pair_classifier.pt'}")

    return {"best_auc": best_auc, "temperature": temperature, "pairs": len(y_t)}


def train_gbdt(
    pairs_path: str | Path,
    exclude: frozenset[str] = frozenset(),
    val_frac: float = 0.1,
    seed: int = 42,
    data: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
) -> dict:
    """GBDT on the same features."""
    from sklearn.ensemble import HistGradientBoostingClassifier

    x_t, y_t, groups = data if data is not None else load_features(pairs_path, exclude)
    x_train, y_train, x_val, y_val, _g = _split(x_t, y_t, groups, val_frac, seed)
    clf = HistGradientBoostingClassifier(random_state=seed)
    clf.fit(x_train.numpy(), y_train.numpy())
    auc = roc_auc_score(y_val.numpy(), clf.predict_proba(x_val.numpy())[:, 1])
    print(f"gbdt val_auc={auc:.4f}")
    return {"best_auc": float(auc), "pairs": len(y_t)}
