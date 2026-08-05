"""Ablation sweep: baseline + skip-each-layer per-depth AUC trajectories."""

import numpy as np
import pytest
import torch

from tfm_lens.evaluation.balef_exp6 import ablation_diffs, ablation_sweep
from tfm_lens.evaluation.native_readout import native_final_auc
from toys import ToyAdapter3D

H = ToyAdapter3D.HIDDEN


def _donor_tables():
    """Two leave-one-out donor tables of differing row counts (binary)."""
    return [
        (torch.randn(5, H), torch.tensor([0.0, 1.0, 0.0, 1.0, 0.0]), torch.randn(3, H)),
        (torch.randn(7, H), torch.tensor([1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0]), torch.randn(2, H)),
    ]


@pytest.fixture
def toy_table():
    f = ToyAdapter3D.HIDDEN
    X_train, y_train = torch.randn(6, f), torch.tensor([0.0, 1.0, 0.0, 1.0, 0.0, 1.0])
    X_test, y_test = torch.randn(4, f), np.array([0, 1, 0, 1])
    return X_train, y_train, X_test, y_test


def test_ablation_sweep_returns_all_metrics_and_zscore(toy_adapter, toy_decoders, toy_table):
    X_train, y_train, X_test, y_test = toy_table

    res = ablation_sweep(toy_adapter, toy_decoders, X_train, y_train, X_test, y_test, n_classes=2)

    n_depths = toy_adapter.n_layers + 1
    metrics = {"auc", "gt_logit", "margin"}
    assert set(res["baseline"]) == metrics  # one trajectory per metric, off one forward
    assert all(len(res["baseline"][name]) == n_depths for name in metrics)
    assert set(res["skip"]) == set(range(toy_adapter.n_layers))  # skip each layer once
    for traj in res["skip"].values():
        assert set(traj) == metrics
        assert all(len(traj[name]) == n_depths for name in metrics)
    assert all(0.0 <= a <= 1.0 for a in res["baseline"]["auc"])
    assert set(res["zscore"]) == {"mu", "sigma"}  # clean-baseline normalization stats
    assert res["zscore"]["sigma"] > 0


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


class TestResampleAblation:
    def test_resample_sweep_shape_and_iqr(self, toy_adapter, toy_decoders, toy_table):
        X_train, y_train, X_test, y_test = toy_table
        res = ablation_sweep(
            toy_adapter,
            toy_decoders,
            X_train,
            y_train,
            X_test,
            y_test,
            n_classes=2,
            ablation="resample",
            donor_tables=_donor_tables(),
            n_donors=2,
            seed=0,
        )
        n_depths = toy_adapter.n_layers + 1
        assert set(res["skip"]) == set(range(toy_adapter.n_layers))
        for traj in res["skip"].values():
            for name in ("auc", "gt_logit", "margin"):
                assert len(traj[name]) == n_depths  # donor-averaged trajectory
                assert len(traj[f"{name}_p25"]) == n_depths  # IQR band kept
                assert len(traj[f"{name}_p75"]) == n_depths
        assert set(res["zscore"]) == {"mu", "sigma"}

    def test_resample_is_reproducible(self, toy_adapter, toy_decoders, toy_table):
        X_train, y_train, X_test, y_test = toy_table
        donors = _donor_tables()
        kw = dict(n_classes=2, ablation="resample", donor_tables=donors, n_donors=2, seed=3)
        a = ablation_sweep(toy_adapter, toy_decoders, X_train, y_train, X_test, y_test, **kw)
        b = ablation_sweep(toy_adapter, toy_decoders, X_train, y_train, X_test, y_test, **kw)
        assert a["skip"][0]["auc"] == b["skip"][0]["auc"]  # fixed seed → identical

    def test_resample_requires_donor_tables(self, toy_adapter, toy_decoders, toy_table):
        X_train, y_train, X_test, y_test = toy_table
        with pytest.raises(ValueError, match="donor_tables"):
            ablation_sweep(
                toy_adapter,
                toy_decoders,
                X_train,
                y_train,
                X_test,
                y_test,
                n_classes=2,
                ablation="resample",
            )

    def test_unknown_ablation_raises(self, toy_adapter, toy_decoders, toy_table):
        X_train, y_train, X_test, y_test = toy_table
        with pytest.raises(ValueError, match="unknown ablation"):
            ablation_sweep(
                toy_adapter,
                toy_decoders,
                X_train,
                y_train,
                X_test,
                y_test,
                n_classes=2,
                ablation="mean",
            )
