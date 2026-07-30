"""Ablation sweep for the self-repair analysis.

One frozen forward per condition (baseline, then skip each layer), decode every
depth with the fine-tuned decoders, reduce to all metrics off the same logits.

- per-depth metrics: AUC / GT-logit / margin (one forward feeds all).
- ``zscore``: clean-baseline (mu, sigma) for cross-task GT-logit normalization.
- skip is orthogonal — ``skip_layer`` wraps the same forward.
"""

import numpy as np
import torch
from scipy.special import softmax

from tfm_lens.adapters.base import ModelAdapter
from tfm_lens.core.interventions import skip_layer
from tfm_lens.evaluation.layerwise import (
    gt_logit_zscore_stats,
    layerwise_auc,
    layerwise_gt_logit,
    layerwise_margin,
    predict_layers,
    predict_layers_logits,
)


def native_final_auc(
    adapter: ModelAdapter,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_test: torch.Tensor,
    y_test: np.ndarray,
    n_classes: int,
) -> float:
    """AUC of the model's own decoder on the final layer — the paper's 'main' score
    (used for normalization and final_diff; distinct from the fine-tuned probes)."""
    native = [adapter.decoder_template()] * (adapter.n_layers + 1)
    probs = predict_layers(adapter, native, X_train, y_train, X_test, n_classes)
    return layerwise_auc([probs[-1]], y_test)[0]


def ablation_diffs(
    adapter: ModelAdapter,
    decoders: list[torch.nn.Module],
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_test: torch.Tensor,
    y_test: np.ndarray,
    n_classes: int,
) -> list[tuple[int, float, float]]:
    """Per ablated layer m: ``(m, immediate_diff, final_diff)`` — the two
    normalized AUC drops that reveal self-repair.

    Both diffs are normalized by the baseline native-final AUC (floored at 0.5):
    ``immediate_diff`` is the fine-tuned decode right after the neutered layer
    (depth m+1) vs baseline at the same depth; ``final_diff`` is the model's
    native final prediction under ablation vs baseline. Self-repair = large
    immediate drop, small final drop.
    """
    sweep = ablation_sweep(adapter, decoders, X_train, y_train, X_test, y_test, n_classes)
    baseline_ft = sweep["baseline"]["auc"]
    baseline_main = native_final_auc(adapter, X_train, y_train, X_test, y_test, n_classes)
    m_norm = max(baseline_main, 0.5)

    diffs = []
    for m in range(adapter.n_layers):
        with skip_layer(adapter, m):
            ablated_main = native_final_auc(adapter, X_train, y_train, X_test, y_test, n_classes)
        immediate = (sweep["skip"][m]["auc"][m + 1] - baseline_ft[m + 1]) / m_norm
        final = (ablated_main - baseline_main) / m_norm
        diffs.append((m, float(immediate), float(final)))
    return diffs


def _all_metrics(logits: list[np.ndarray], y_test: np.ndarray) -> dict[str, list[float]]:
    """Every per-depth metric off one forward's logits (binary tasks)."""
    return {
        "auc": layerwise_auc([softmax(z, axis=1) for z in logits], y_test),
        "gt_logit": layerwise_gt_logit(logits, y_test),
        "margin": layerwise_margin(logits, y_test),
    }


def ablation_sweep(
    adapter: ModelAdapter,
    decoders: list[torch.nn.Module],
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_test: torch.Tensor,
    y_test: np.ndarray,
    n_classes: int,
) -> dict:
    """Baseline + skip-each-layer per-depth metric trajectories for one table.

    One forward per condition feeds every metric. Returns::

        {"baseline": {metric: [per depth]},
         "skip": {layer: {metric: [per depth]}},
         "zscore": {"mu": float, "sigma": float}}

    ``metric`` in {auc, gt_logit, margin}; ``zscore`` = clean-baseline stats for
    cross-task GT-logit normalization.
    """

    def _logits():
        return predict_layers_logits(adapter, decoders, X_train, y_train, X_test, n_classes)

    base_logits = _logits()
    baseline = _all_metrics(base_logits, y_test)
    mu, sigma = gt_logit_zscore_stats(base_logits, y_test)
    skip = {}
    for m in range(adapter.n_layers):
        with skip_layer(adapter, m):
            skip[m] = _all_metrics(_logits(), y_test)
    return {"baseline": baseline, "skip": skip, "zscore": {"mu": mu, "sigma": sigma}}
