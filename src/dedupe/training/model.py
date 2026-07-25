"""Pairwise classifier over schema-driven comparison features."""

from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn as nn


@dataclass
class PairClassifierConfig:
    dim: int = 19  # len(feature_names()) for the active schema
    hidden: int = 64
    depth: int = 2
    dropout: float = 0.1


class _Block(nn.Module):
    def __init__(self, hidden: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(hidden)
        self.up = nn.Linear(hidden, hidden)
        self.down = nn.Linear(hidden, hidden)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.drop(self.down(self.act(self.up(self.norm(x)))))


class PairClassifier(nn.Module):
    def __init__(
        self, config: PairClassifierConfig | None = None, features: list[str] | None = None, **kwargs
    ):
        super().__init__()
        self.config = config or PairClassifierConfig(**kwargs)
        self.features = features or []
        c = self.config
        self.proj = nn.Sequential(nn.LayerNorm(c.dim), nn.Linear(c.dim, c.hidden))
        self.blocks = nn.Sequential(*[_Block(c.hidden, c.dropout) for _ in range(c.depth)])
        self.head = nn.Sequential(nn.LayerNorm(c.hidden), nn.Linear(c.hidden, 1))
        # calibration temperature, fitted post-train
        self.register_buffer("temperature", torch.ones(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.blocks(self.proj(x))).squeeze(-1)

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward(x) / self.temperature)

    def save(self, path: str | Path) -> None:
        torch.save(
            {"config": asdict(self.config), "features": self.features, "state_dict": self.state_dict()},
            path,
        )

    @classmethod
    def load(cls, path: str | Path, map_location: str = "cpu") -> "PairClassifier":
        ckpt = torch.load(path, map_location=map_location, weights_only=True)
        model = cls(PairClassifierConfig(**ckpt["config"]), features=list(ckpt.get("features", [])))
        model.load_state_dict(ckpt["state_dict"])
        return model
