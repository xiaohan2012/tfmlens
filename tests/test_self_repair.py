"""Ablation sweep: baseline + skip-each-layer per-depth AUC trajectories."""

import numpy as np
import pytest
import torch

from tfm_lens.evaluation.self_repair import ablation_diffs, ablation_sweep, native_final_auc
from toys import ToyAdapter3D


@pytest.fixture
def toy_table():
    f = ToyAdapter3D.HIDDEN
    X_train, y_train = torch.randn(6, f), torch.tensor([0.0, 1.0, 0.0, 1.0, 0.0, 1.0])
    X_test, y_test = torch.randn(4, f), np.array([0, 1, 0, 1])
    return X_train, y_train, X_test, y_test


def test_ablation_sweep_returns_baseline_and_per_layer_trajectories(
    toy_adapter, toy_decoders, toy_table
):
    X_train, y_train, X_test, y_test = toy_table

    res = ablation_sweep(toy_adapter, toy_decoders, X_train, y_train, X_test, y_test, n_classes=2)

    n_depths = toy_adapter.n_layers + 1
    assert len(res["baseline"]) == n_depths  # one AUC per capture depth
    assert set(res["skip"]) == set(range(toy_adapter.n_layers))  # skip each layer once
    for traj in res["skip"].values():
        assert len(traj) == n_depths
    assert all(0.0 <= a <= 1.0 for a in res["baseline"])


def test_native_final_auc_is_a_score(toy_adapter, toy_table):
    X_train, y_train, X_test, y_test = toy_table
    auc = native_final_auc(toy_adapter, X_train, y_train, X_test, y_test, n_classes=2)
    assert isinstance(auc, float)
    assert 0.0 <= auc <= 1.0


def test_ablation_diffs_one_per_ablated_layer(toy_adapter, toy_decoders, toy_table):
    X_train, y_train, X_test, y_test = toy_table
    diffs = ablation_diffs(toy_adapter, toy_decoders, X_train, y_train, X_test, y_test, n_classes=2)
    assert len(diffs) == toy_adapter.n_layers  # one row per ablated layer
    for m, immediate, final in diffs:
        assert isinstance(m, int)
        assert isinstance(immediate, float)
        assert isinstance(final, float)
