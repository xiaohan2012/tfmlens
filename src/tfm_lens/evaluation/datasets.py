"""Load a TabArena binary-classification task from OpenML into a train/test table.

Returns numeric arrays plus the categorical column indices that the LimiX
preprocessing (``evaluation.preprocess``) needs. Categoricals are factorized to
integer codes; the target is label-encoded to ``0..C-1``.
"""

import numpy as np
import openml
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder

# TabArena binary tasks (from the reference experiment's BENCHMARK_TASK_IDS).
TABARENA_BINARY_TASK_IDS = [
    363621,
    363671,
    363696,
    363629,
    363626,
    363682,
    363684,
    363700,
    363674,
    363694,
    363619,
    363623,
    363689,
    363706,
    363624,
]


def load_tabarena_task(task_id: int, fold: int = 0, repeat: int = 0):
    """Fetch one OpenML task's fold as ``(X_train, y_train, X_test, y_test, categorical_idx)``."""
    task = openml.tasks.get_task(task_id)
    X, y, categorical_mask, _ = task.get_dataset().get_data(
        target=task.target_name, dataset_format="dataframe"
    )
    categorical_idx = [i for i, is_cat in enumerate(categorical_mask) if is_cat]
    X_num = _to_numeric(X, categorical_idx)
    y_enc = LabelEncoder().fit_transform(y)

    train_idx, test_idx = task.get_train_test_split_indices(fold=fold, repeat=repeat)
    return X_num[train_idx], y_enc[train_idx], X_num[test_idx], y_enc[test_idx], categorical_idx


def subsample(X: np.ndarray, y: np.ndarray, n: int, seed: int = 0):
    """Take at most ``n`` rows (``n <= 0`` keeps all), seeded for reproducibility."""
    if n <= 0 or n >= len(X):
        return X, y
    idx = np.random.RandomState(seed).choice(len(X), n, replace=False)
    return X[idx], y[idx]


def load_task_record(task_id, preprocess, subsample_train, subsample_test, seed=0) -> dict:
    """Load -> subsample -> 0-index labels -> preprocess -> tensors for one task.

    Shared by the sweep scripts. ``preprocess`` is the model-specific
    ``evaluation.preprocess`` fn. Returns tensors + row/class counts.
    """
    X_train, y_train, X_test, y_test, cat_idx = load_tabarena_task(task_id)
    X_train, y_train = subsample(X_train, y_train, subsample_train, seed)
    X_test, y_test = subsample(X_test, y_test, subsample_test, seed)
    # 0-index the labels: TabICL's one-hot y_encoder needs contiguous ints; harmless
    # for LimiX since binary labels are already 0/1.
    classes = np.unique(y_train)
    y_train = np.searchsorted(classes, y_train)
    y_test = np.searchsorted(classes, y_test)
    X_train_p, X_test_p = preprocess(X_train, y_train, X_test, cat_idx)
    return {
        "Xtr": torch.tensor(X_train_p),
        "ytr": torch.tensor(y_train).float(),
        "Xte": torch.tensor(X_test_p),
        "y_test": y_test,
        "n_classes": len(classes),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
    }


def _to_numeric(X: pd.DataFrame, categorical_idx: list[int]) -> np.ndarray:
    """DataFrame -> float32 array: categoricals factorized to codes, numerics coerced."""
    out = np.empty(X.shape, dtype=np.float32)
    for j, col in enumerate(X.columns):
        if j in categorical_idx:
            out[:, j] = pd.factorize(X[col])[0]  # integer codes, -1 for NaN
        else:
            out[:, j] = pd.to_numeric(X[col], errors="coerce")
    return out
