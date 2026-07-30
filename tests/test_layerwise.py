"""Per-layer decode + metrics on one table (evaluation.layerwise)."""

import numpy as np
import pytest
import torch
from scipy.special import softmax

from tfm_lens.evaluation.layerwise import (
    gt_logit_zscore_stats,
    layerwise_auc,
    layerwise_gt_logit,
    layerwise_margin,
    load_decoders,
    predict_layers,
    predict_layers_logits,
)
from toys import ToyAdapter3D


def test_predict_layers_returns_valid_probs_per_depth(toy_adapter, toy_decoders):
    f, c = ToyAdapter3D.HIDDEN, ToyAdapter3D.N_CLASSES
    X_train, y_train = torch.randn(5, f), torch.randint(0, c, (5,))
    X_test = torch.randn(3, f)
    probs = predict_layers(toy_adapter, toy_decoders, X_train, y_train, X_test, n_classes=c)
    assert len(probs) == toy_adapter.n_layers + 1  # one array per capture depth
    for p in probs:
        assert p.shape == (3, c)  # [n_test, n_classes]
        np.testing.assert_allclose(p.sum(axis=1), 1.0, atol=1e-5)  # valid distributions


def test_predict_layers_is_softmax_of_logits(toy_adapter, toy_decoders):
    # predict_layers must be exactly softmax(predict_layers_logits) — same forward,
    # metrics differ only in the reduction. Independent scipy softmax as the oracle.
    f, c = ToyAdapter3D.HIDDEN, ToyAdapter3D.N_CLASSES
    torch.manual_seed(0)
    X_train, y_train = torch.randn(5, f), torch.randint(0, c, (5,))
    X_test = torch.randn(3, f)
    logits = predict_layers_logits(toy_adapter, toy_decoders, X_train, y_train, X_test, n_classes=c)
    probs = predict_layers(toy_adapter, toy_decoders, X_train, y_train, X_test, n_classes=c)
    for z, p in zip(logits, probs, strict=True):
        assert z.shape == (3, c)  # raw logits, [n_test, n_classes]
        np.testing.assert_allclose(p, softmax(z, axis=1), rtol=1e-6, atol=1e-6)


def test_layerwise_auc_scores_each_depth():
    y_test = np.array([0, 0, 1, 1])
    perfect = np.array([[0.9, 0.1], [0.8, 0.2], [0.2, 0.8], [0.1, 0.9]])
    chance = np.full((4, 2), 0.5)
    aucs = layerwise_auc([perfect, chance], y_test)
    assert len(aucs) == 2  # one score per depth
    assert aucs[0] == 1.0  # perfect separation
    assert 0.0 <= aucs[1] <= 1.0


def test_gt_logit_means_true_class_per_depth():
    # true-class logits: depth0 -> [2, 1] (mean 1.5); depth1 -> [4, 0] (mean 2.0)
    y_test = np.array([0, 1])
    depth0 = np.array([[2.0, 0.0], [0.0, 1.0]])
    depth1 = np.array([[4.0, 0.0], [0.0, 0.0]])
    assert layerwise_gt_logit([depth0, depth1], y_test) == pytest.approx([1.5, 2.0])


class TestLayerwiseMargin:
    """Test layerwise_margin (binary z_true - z_other)."""

    def test_means_logit_difference_per_depth(self):
        # margins: depth -> [3-1, 2-0] = [2, 2] (mean 2.0)
        y_test = np.array([0, 1])
        depth = np.array([[3.0, 1.0], [0.0, 2.0]])
        assert layerwise_margin([depth], y_test) == pytest.approx([2.0])

    def test_raises_on_non_binary(self):
        y_test = np.array([0, 1])
        three_class = np.zeros((2, 3))
        with pytest.raises(ValueError, match="binary-only"):
            layerwise_margin([three_class], y_test)


class TestGtLogitZscoreStats:
    """Test gt_logit_zscore_stats (mu, sigma from clean final-layer, cross-row)."""

    def test_mean_and_std_of_final_layer_true_class(self):
        y_test = np.array([0, 0, 0])
        early = np.zeros((3, 2))
        final = np.array([[3.0, 9.0], [1.0, 9.0], [2.0, 9.0]])  # true-class col -> [3, 1, 2]
        mu, sigma = gt_logit_zscore_stats([early, final], y_test)
        assert mu == pytest.approx(2.0)
        assert sigma == pytest.approx(np.std([3.0, 1.0, 2.0]))

    def test_mu_equals_final_layer_gt_logit_so_final_zscores_to_zero(self):
        # (s_final - mu) / sigma == 0, since mu is exactly the final-layer GT-logit mean
        y_test = np.array([0, 0, 0])
        final = np.array([[3.0, 9.0], [1.0, 9.0], [2.0, 9.0]])
        mu, _ = gt_logit_zscore_stats([final], y_test)
        s_final = layerwise_gt_logit([final], y_test)[0]
        assert s_final == pytest.approx(mu)

    def test_raises_on_degenerate_zero_spread(self):
        y_test = np.array([0, 0, 0])
        final = np.array([[5.0, 0.0], [5.0, 0.0], [5.0, 0.0]])  # true-class col all 5 -> sigma 0
        with pytest.raises(ValueError, match="degenerate"):
            gt_logit_zscore_stats([final], y_test)


def test_load_decoders_returns_one_per_depth(toy_adapter, tmp_path):
    # load_decoders is pure plumbing (read N state dicts -> N decoders), so a toy
    # template round-tripped through disk exercises it — no trained weights needed.
    n = toy_adapter.n_layers + 1
    for i in range(n):
        torch.save(toy_adapter.decoder_template().state_dict(), tmp_path / f"decoder_layer_{i}.pth")
    decoders = load_decoders(tmp_path, toy_adapter)
    assert len(decoders) == n  # one decoder per capture depth
