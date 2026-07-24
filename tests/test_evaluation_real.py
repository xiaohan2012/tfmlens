"""Real-data smoke: OpenML TabArena task -> LimiX preprocessing -> per-layer AUC.

Needs the eval group (openml), the LimiX ckpt, and network access, so it is
skipped in CI. Run locally with ``uv run --group eval pytest``.

Uses the model's own native decoder (``cls_y_decoder``) at every depth — no
fine-tuned probes, so nothing has to be checked into version control. The claim
is narrow but real: the native final-layer decode scores a real task above
chance, proving load -> preprocess -> forward -> decode -> AUC all work on real
data. ("AUC builds up with depth" needs the fine-tuned probes and is covered by
the layerwise unit test.)
"""

import numpy as np
import pytest
import torch

pytest.importorskip("openml")  # eval-group dep; skip cleanly without it

from tfm_lens.adapters.limix import LimixAdapter  # noqa: E402
from tfm_lens.evaluation.datasets import (  # noqa: E402
    TABARENA_BINARY_TASK_IDS,
    load_tabarena_task,
)
from tfm_lens.evaluation.layerwise import layerwise_auc, predict_layers  # noqa: E402
from tfm_lens.evaluation.preprocess import limix_preprocess  # noqa: E402


def test_real_tabarena_task_native_decode_beats_chance(limix_model):
    # a real OpenML binary task through the full eval pipeline; the model's own
    # decoder at the final layer must score above chance, proving preprocessing
    # + forward + decode work on real data.
    adapter = LimixAdapter(limix_model)
    native = adapter.decoder_template()
    decoders = [native] * (adapter.n_layers + 1)

    X_train, y_train, X_test, y_test, categorical_idx = load_tabarena_task(
        TABARENA_BINARY_TASK_IDS[0]
    )
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
    assert aucs[-1] > 0.6  # the native final layer decodes a real task above chance
