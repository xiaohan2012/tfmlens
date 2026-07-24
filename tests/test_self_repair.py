"""Ablation sweep: baseline + skip-each-layer per-depth AUC trajectories."""

import numpy as np
import torch

from tfm_lens.evaluation.self_repair import ablation_sweep
from toys import ToyAdapter3D


def test_ablation_sweep_returns_baseline_and_per_layer_trajectories(toy_adapter, toy_decoders):
    f = ToyAdapter3D.HIDDEN
    X_train, y_train = torch.randn(6, f), torch.tensor([0.0, 1.0, 0.0, 1.0, 0.0, 1.0])
    X_test, y_test = torch.randn(4, f), np.array([0, 1, 0, 1])

    res = ablation_sweep(toy_adapter, toy_decoders, X_train, y_train, X_test, y_test, n_classes=2)

    n_depths = toy_adapter.n_layers + 1
    assert len(res["baseline"]) == n_depths  # one AUC per capture depth
    assert set(res["skip"]) == set(range(toy_adapter.n_layers))  # skip each layer once
    for traj in res["skip"].values():
        assert len(traj) == n_depths
    assert all(0.0 <= a <= 1.0 for a in res["baseline"])
