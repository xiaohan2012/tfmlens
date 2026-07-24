"""Per-layer decode + ROC-AUC on one table (evaluation.layerwise)."""

import copy
from pathlib import Path

import numpy as np
import pytest
import torch

from tfm_lens.adapters.limix import LimixAdapter
from tfm_lens.evaluation.layerwise import layerwise_auc, predict_layers
from toys import ToyAdapter3D

TRAINED_DECODERS = Path("weights/limix_2m")


def test_predict_layers_returns_valid_probs_per_depth(toy_adapter, toy_decoders):
    f, c = ToyAdapter3D.HIDDEN, ToyAdapter3D.N_CLASSES
    X_train, y_train = torch.randn(5, f), torch.randint(0, c, (5,))
    X_test = torch.randn(3, f)
    probs = predict_layers(toy_adapter, toy_decoders, X_train, y_train, X_test, n_classes=c)
    assert len(probs) == toy_adapter.n_layers + 1  # one array per capture depth
    for p in probs:
        assert p.shape == (3, c)  # [n_test, n_classes]
        np.testing.assert_allclose(p.sum(axis=1), 1.0, atol=1e-5)  # valid distributions


def test_layerwise_auc_scores_each_depth():
    y_test = np.array([0, 0, 1, 1])
    perfect = np.array([[0.9, 0.1], [0.8, 0.2], [0.2, 0.8], [0.1, 0.9]])
    chance = np.full((4, 2), 0.5)
    aucs = layerwise_auc([perfect, chance], y_test)
    assert len(aucs) == 2  # one score per depth
    assert aucs[0] == 1.0  # perfect separation
    assert 0.0 <= aucs[1] <= 1.0


@pytest.mark.skipif(not TRAINED_DECODERS.exists(), reason="no trained decoders at weights/limix_2m")
def test_predict_layers_real_limix_builds_up_with_depth(limix_model):
    # real LimiX + the fine-tuned decoders on a separable binary table: per-layer
    # AUC should climb with depth (the paper's logit-lens story), proving the
    # trained decoders plug into the eval pipeline unchanged.
    adapter = LimixAdapter(limix_model)
    n = adapter.n_layers + 1
    decoders = []
    for layer in range(n):
        d = copy.deepcopy(adapter.decoder_template())
        state = torch.load(TRAINED_DECODERS / f"decoder_layer_{layer}.pth", map_location="cpu")
        d.load_state_dict(state)
        decoders.append(d)

    torch.manual_seed(0)
    f = 5
    X_train, X_test = torch.randn(60, f), torch.randn(40, f)
    w = torch.tensor([1.5, -1.0, 0.5, 0.0, 0.0])
    y_train = (X_train @ w > 0).float()
    y_test = (X_test @ w > 0).long().numpy()

    aucs = layerwise_auc(
        predict_layers(adapter, decoders, X_train, y_train, X_test, n_classes=2), y_test
    )
    assert len(aucs) == n
    assert all(0.0 <= a <= 1.0 for a in aucs)
    assert aucs[-1] > 0.8  # the final layer decodes the separable task
    assert aucs[-1] > aucs[0] + 0.2  # accuracy builds up with depth
