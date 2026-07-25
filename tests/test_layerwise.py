"""Per-layer decode + ROC-AUC on one table (evaluation.layerwise)."""

from pathlib import Path

import numpy as np
import pytest
import torch

from tfm_lens.adapters.limix import LimixAdapter
from tfm_lens.evaluation.layerwise import layerwise_auc, load_decoders, predict_layers
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


def test_load_decoders_returns_one_per_depth(toy_adapter, tmp_path):
    # load_decoders is pure plumbing (read N state dicts -> N decoders), so a toy
    # template round-tripped through disk exercises it — no trained weights needed.
    n = toy_adapter.n_layers + 1
    for i in range(n):
        torch.save(toy_adapter.decoder_template().state_dict(), tmp_path / f"decoder_layer_{i}.pth")
    decoders = load_decoders(tmp_path, toy_adapter)
    assert len(decoders) == n  # one decoder per capture depth


@pytest.mark.skipif(not TRAINED_DECODERS.exists(), reason="no trained decoders at weights/limix_2m")
def test_finetuned_decoders_decode_real_limix_smoke(limix_model):
    # Smoke test: the fine-tuned decoders load and plug into the eval pipeline on
    # real LimiX, yielding valid per-depth AUCs with signal at the final layer.
    # It deliberately does NOT assert the "builds up with depth" logit-lens story
    # — that's a scientific claim, shown by the Figure-8 reproduction, not a unit
    # test, and a hard-coded threshold on it would just be a brittle post-hoc guess.
    adapter = LimixAdapter(limix_model)
    n = adapter.n_layers + 1
    decoders = load_decoders(TRAINED_DECODERS, adapter)

    torch.manual_seed(0)
    f = 5
    X_train, X_test = torch.randn(60, f), torch.randn(40, f)
    w = torch.tensor([1.5, -1.0, 0.5, 0.0, 0.0])
    y_train = (X_train @ w > 0).float()
    y_test = (X_test @ w > 0).long().numpy()

    aucs = layerwise_auc(
        predict_layers(adapter, decoders, X_train, y_train, X_test, n_classes=2), y_test
    )
    assert len(aucs) == n  # one AUC per capture depth
    assert all(np.isfinite(a) and 0.0 <= a <= 1.0 for a in aucs)  # all valid scores
    assert aucs[-1] > 0.5  # the final layer decodes the separable task above chance
