"""Load a TabArena binary-classification task from OpenML into a train/test table.

Returns numeric arrays plus the categorical column indices that the LimiX
preprocessing (``evaluation.preprocess``) needs. Categoricals are factorized to
integer codes; the target is label-encoded to ``0..C-1``.
"""

import numpy as np
import openml
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# TabArena binary tasks (from the reference experiment's BENCHMARK_TASK_IDS).
TABARENA_BINARY = [
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


def _to_numeric(X: pd.DataFrame, categorical_idx: list[int]) -> np.ndarray:
    """DataFrame -> float32 array: categoricals factorized to codes, numerics coerced."""
    out = np.empty(X.shape, dtype=np.float32)
    for j, col in enumerate(X.columns):
        if j in categorical_idx:
            out[:, j] = pd.factorize(X[col])[0]  # integer codes, -1 for NaN
        else:
            out[:, j] = pd.to_numeric(X[col], errors="coerce")
    return out
