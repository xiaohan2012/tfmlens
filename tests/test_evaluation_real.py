"""Real-data smoke: OpenML TabArena task -> LimiX preprocessing -> per-layer AUC.

Needs the eval group (openml), the LimiX ckpt, the trained decoders, and network
access, so it is skipped in CI. Run locally with ``uv run --group eval pytest``.
"""

import copy
from pathlib import Path

import numpy as np
import pytest
import torch

pytest.importorskip("openml")  # eval-group dep; skip cleanly without it

from tfm_lens.adapters.limix import LimixAdapter  # noqa: E402
from tfm_lens.evaluation.datasets import TABARENA_BINARY, load_tabarena_task  # noqa: E402
from tfm_lens.evaluation.layerwise import layerwise_auc, predict_layers  # noqa: E402
from tfm_lens.evaluation.preprocess import limix_preprocess  # noqa: E402

TRAINED_DECODERS = Path("weights/limix_2m")


@pytest.mark.skipif(not TRAINED_DECODERS.exists(), reason="no trained decoders at weights/limix_2m")
def test_real_tabarena_task_builds_up_with_depth(limix_model):
    # a real OpenML binary task through the full eval pipeline: per-layer AUC
    # should climb with depth, proving preprocessing + decode work on real data.
    adapter = LimixAdapter(limix_model)
    decoders = []
    for layer in range(adapter.n_layers + 1):
        d = copy.deepcopy(adapter.decoder_template())
        state = torch.load(TRAINED_DECODERS / f"decoder_layer_{layer}.pth", map_location="cpu")
        d.load_state_dict(state)
        decoders.append(d)

    X_train, y_train, X_test, y_test, categorical_idx = load_tabarena_task(TABARENA_BINARY[0])
    rng = np.random.RandomState(0)  # subsample for a fast CPU smoke
    tr = rng.choice(len(X_train), min(300, len(X_train)), replace=False)
    te = rng.choice(len(X_test), min(100, len(X_test)), replace=False)
    X_train, y_train, X_test, y_test = X_train[tr], y_train[tr], X_test[te], y_test[te]

    X_train_p, X_test_p = limix_preprocess(X_train, y_train, X_test, categorical_idx)
    probs = predict_layers(
        adapter,
        decoders,
        torch.tensor(X_train_p),
        torch.tensor(y_train).float(),
        torch.tensor(X_test_p),
        n_classes=len(np.unique(y_train)),
    )
    aucs = layerwise_auc(probs, y_test)

    assert len(aucs) == adapter.n_layers + 1
    assert all(0.0 <= a <= 1.0 for a in aucs)
    assert aucs[-1] > 0.6  # the final layer decodes a real task above chance
    assert aucs[-1] > aucs[0]  # accuracy builds up with depth
