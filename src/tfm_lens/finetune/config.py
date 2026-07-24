"""Typed, validated finetune config (pydantic v2).

Defaults are the paper's verified recipe; per-model YAML files override only the
differences; every run saves a config.json snapshot so it is self-describing.
"""

from pathlib import Path
from typing import Literal

import torch
import yaml
from pydantic import BaseModel, Field, field_validator


class PriorConfig(BaseModel):
    prior_type: Literal["mlp_scm", "tree_scm", "mix_scm"] = "mix_scm"
    mix_probs: tuple[float, float] = (0.7, 0.3)  # (mlp, tree) selection probabilities
    min_features: int = 2
    max_features: int = 30
    max_classes: int = 10
    max_seq_len: int = 1024
    min_train_size: float = 0.1
    max_train_size: float = 0.9
    batch_size_per_gp: int = 4
    log_seq_len: bool = False
    n_jobs: int = -1

    @field_validator("mix_probs")
    @classmethod
    def _sum_to_one(cls, v: tuple[float, float]) -> tuple[float, float]:
        if abs(sum(v) - 1.0) > 1e-6:
            raise ValueError(f"mix_probs must sum to 1, got {v}")
        return v


def _default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


class TrainConfig(BaseModel):
    seed: int = 0

    # optimizer
    lr: float = 3e-5
    weight_decay: float = 1e-4
    grad_clip: float = 1.0

    # finetune loop
    max_steps: int = 200
    prior_batch_size: int = 512
    micro_batch_size: int = 64
    training_batch_size: int = 8

    prior: PriorConfig = Field(default_factory=PriorConfig)

    # logging / saving
    out_dir: Path
    log_every: int = 1
    save_every: int = 100
    wandb: bool = False

    device: str = Field(default_factory=_default_device)

    @classmethod
    def from_yaml(cls, path) -> "TrainConfig":
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        return cls.model_validate(data)

    def save_snapshot(self, out_dir) -> None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "config.json").write_text(self.model_dump_json(indent=2))
