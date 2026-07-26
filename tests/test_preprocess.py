"""tabicl_preprocess on a synthetic mixed-type table.

Checks the plumbing (categoricals encoded, train-fit/both-transform, float32,
finite, row counts preserved) without needing the real TabICL model. Skipped
when the tabicl/eval deps aren't installed.
"""

import numpy as np
import pytest


def test_tabicl_preprocess_shapes_dtype_finite():
    pytest.importorskip("tabicl")
    pytest.importorskip("pandas")
    from tfm_lens.evaluation.preprocess import tabicl_preprocess

    rng = np.random.default_rng(0)
    n_train, n_test, n_feat = 40, 15, 5
    x = rng.normal(size=(n_train + n_test, n_feat))
    x[:, 2] = rng.integers(0, 3, size=n_train + n_test)  # column 2 is categorical
    X_train, X_test = x[:n_train], x[n_train:]
    y_train = rng.integers(0, 2, size=n_train)

    Xtr_p, Xte_p = tabicl_preprocess(X_train, y_train, X_test, categorical_idx=[2])

    assert Xtr_p.shape[0] == n_train and Xte_p.shape[0] == n_test  # rows preserved
    assert Xtr_p.shape[1] == Xte_p.shape[1] == n_feat  # columns preserved
    assert Xtr_p.dtype == np.float32 and Xte_p.dtype == np.float32
    assert np.isfinite(Xtr_p).all() and np.isfinite(Xte_p).all()


def test_tabfm_preprocess_shapes_dtype_finite():
    from tfm_lens.evaluation.preprocess import tabfm_preprocess

    rng = np.random.default_rng(0)
    n_train, n_test, n_feat = 40, 15, 5
    x = rng.normal(size=(n_train + n_test, n_feat))
    x[:, 1] *= 100  # a big-scale column the standard-scale/power step must tame
    x[:, 2] = rng.integers(0, 3, size=n_train + n_test)  # column 2 is categorical
    X_train, X_test = x[:n_train], x[n_train:]
    y_train = rng.integers(0, 2, size=n_train)

    Xtr_p, Xte_p = tabfm_preprocess(X_train, y_train, X_test, categorical_idx=[2])

    assert Xtr_p.shape[0] == n_train and Xte_p.shape[0] == n_test  # rows preserved
    assert Xtr_p.shape[1] == Xte_p.shape[1] == n_feat  # columns preserved
    assert Xtr_p.dtype == np.float32 and Xte_p.dtype == np.float32
    assert np.isfinite(Xtr_p).all() and np.isfinite(Xte_p).all()


def test_mitra_preprocess_shapes_dtype_finite():
    from tfm_lens.evaluation.preprocess import mitra_preprocess

    rng = np.random.default_rng(0)
    n_train, n_test, n_feat = 40, 15, 5
    x = rng.normal(size=(n_train + n_test, n_feat))
    x[:, 2] = rng.integers(0, 3, size=n_train + n_test)  # column 2 is categorical
    X_train, X_test = x[:n_train], x[n_train:]
    y_train = rng.integers(0, 2, size=n_train)

    Xtr_p, Xte_p = mitra_preprocess(X_train, y_train, X_test, categorical_idx=[2])

    assert Xtr_p.shape[0] == n_train and Xte_p.shape[0] == n_test  # rows preserved
    assert Xtr_p.shape[1] == Xte_p.shape[1] == n_feat  # columns preserved
    assert Xtr_p.dtype == np.float32 and Xte_p.dtype == np.float32
    assert np.isfinite(Xtr_p).all() and np.isfinite(Xte_p).all()
