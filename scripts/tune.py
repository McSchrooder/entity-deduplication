"""Optuna search over the pair-classifier space. Features are computed once and
shared across trials."""

import argparse
import json
from pathlib import Path

import optuna

from dedupe.config import settings
from dedupe.resolver.features import feature_names
from dedupe.training.model import PairClassifierConfig
from dedupe.training.train import load_features, train


def main():
    ap = argparse.ArgumentParser(description="Tune the pair classifier with Optuna")
    ap.add_argument("--pairs", type=Path, default=Path(settings.data_dir) / "pairs.jsonl")
    ap.add_argument("--out", type=Path, default=Path(settings.models_dir))
    ap.add_argument("--trials", type=int, default=100)
    ap.add_argument("--max-epochs", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    dim = len(feature_names())
    data = load_features(args.pairs)

    def objective(trial: optuna.Trial) -> float:
        config = PairClassifierConfig(
            dim=dim,
            hidden=trial.suggest_categorical("hidden", [32, 64, 128, 256]),
            depth=trial.suggest_int("depth", 1, 3),
            dropout=trial.suggest_float("dropout", 0.0, 0.4),
        )

        def report(epoch: int, auc: float) -> bool:
            trial.report(auc, epoch)
            return trial.should_prune()

        result = train(
            args.pairs,
            out_dir=args.out,
            config=config,
            epochs=args.max_epochs,
            batch_size=trial.suggest_categorical("batch_size", [128, 256, 512]),
            lr=trial.suggest_float("lr", 1e-4, 1e-2, log=True),
            weight_decay=trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
            focal_gamma=trial.suggest_categorical("focal_gamma", [0.0, 1.0, 2.0]),
            seed=args.seed,
            save=False,
            epoch_callback=report,
            data=data,
        )
        return result["best_auc"]

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=args.seed, multivariate=True),
        pruner=optuna.pruners.HyperbandPruner(
            min_resource=3, max_resource=args.max_epochs, reduction_factor=3
        ),
    )
    study.optimize(objective, n_trials=args.trials, show_progress_bar=True)

    out = args.out / "tuned_params.json"
    out.write_text(json.dumps(study.best_params, indent=2), encoding="utf-8")
    print(f"best val_auc={study.best_value:.4f} -> {out}")


if __name__ == "__main__":
    main()
