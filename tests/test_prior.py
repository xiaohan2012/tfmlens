"""Block 6 — smoke test for the vendored TabICL prior.

It's copied upstream code, so this only checks it *runs* and yields a sane
synthetic table. Uses ``mlp_scm`` (deterministic, no xgboost) to stay fast and
crash-free — ``tree_scm`` segfaults on macOS arm64 (xgboost/OpenMP vs torch);
the mix used for real training runs on Linux.
"""

import torch

from tfm_lens.vendor.tabicl_prior import PriorDataset


def test_prior_generates_a_batch():
    ds = PriorDataset(
        batch_size=2,
        batch_size_per_gp=2,
        min_features=2,
        max_features=5,
        max_classes=3,
        max_seq_len=32,
        prior_type="mlp_scm",
        n_jobs=1,
        device="cpu",
    )
    X, y, d, seq_lens, train_sizes = next(iter(ds))
    assert X.dim() == 3 and X.shape[0] == 2  # (batch, rows, features)
    assert y.shape[:2] == X.shape[:2]  # a label per row
    assert torch.isfinite(X).all()
