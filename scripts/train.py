import argparse
import json
from pathlib import Path

from dedupe.config import settings
from dedupe.training.model import PairClassifierConfig
from dedupe.training.train import train, train_gbdt


def main():
    ap = argparse.ArgumentParser(description="Train the pair classifier on mined doc pairs")
    ap.add_argument("--pairs", type=Path, default=Path(settings.data_dir) / "pairs.jsonl")
    ap.add_argument("--out", type=Path, default=Path(settings.models_dir))
    ap.add_argument("--model", choices=["mlp", "gbdt"], default="mlp", help="gbdt = AUC control run only")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--focal-gamma", type=float, default=0.0, help="0 = plain BCE")
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument(
        "--exclude-features",
        nargs="*",
        default=[],
        help="feature names to drop, e.g. birth_date_exact for the label-leakage ablation",
    )
    ap.add_argument("--params", type=Path, help="JSON from scripts/tune.py; overrides the flags above")
    args = ap.parse_args()

    exclude = frozenset(args.exclude_features)
    if args.model == "gbdt":
        train_gbdt(args.pairs, exclude=exclude)
        return

    hp = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "focal_gamma": args.focal_gamma,
    }
    arch = {"hidden": args.hidden, "depth": args.depth, "dropout": args.dropout}
    if args.params:
        best = json.loads(args.params.read_text(encoding="utf-8"))
        hp.update({k: v for k, v in best.items() if k in hp})
        arch.update({k: v for k, v in best.items() if k in arch})

    from dedupe.resolver.features import feature_names

    config = PairClassifierConfig(dim=len(feature_names(exclude=exclude)), **arch)
    train(args.pairs, out_dir=args.out, config=config, exclude=exclude, **hp)


if __name__ == "__main__":
    main()
