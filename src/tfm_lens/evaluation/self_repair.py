"""Ablation sweep for the self-repair analysis.

Runs the frozen forward once per condition — baseline, then skipping each layer
in turn — decoding every depth with the fine-tuned decoders and scoring test-row
AUC. Reuses predict_layers / layerwise_auc; the skip is orthogonal (skip_layer
just wraps the same call).
"""

import numpy as np
import torch

from tfm_lens.adapters.base import ModelAdapter
from tfm_lens.core.interventions import skip_layer
from tfm_lens.evaluation.layerwise import layerwise_auc, predict_layers


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
    baseline_ft = sweep["baseline"]
    baseline_main = native_final_auc(adapter, X_train, y_train, X_test, y_test, n_classes)
    m_norm = max(baseline_main, 0.5)

    diffs = []
    for m in range(adapter.n_layers):
        with skip_layer(adapter, m):
            ablated_main = native_final_auc(adapter, X_train, y_train, X_test, y_test, n_classes)
        immediate = (sweep["skip"][m][m + 1] - baseline_ft[m + 1]) / m_norm
        final = (ablated_main - baseline_main) / m_norm
        diffs.append((m, float(immediate), float(final)))
    return diffs


def ablation_sweep(
    adapter: ModelAdapter,
    decoders: list[torch.nn.Module],
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_test: torch.Tensor,
    y_test: np.ndarray,
    n_classes: int,
) -> dict:
    """Baseline + skip-each-layer per-depth AUC trajectories for one table.

    Returns ``{"baseline": [auc per depth], "skip": {layer: [auc per depth]}}``.
    """

    def _aucs():
        probs = predict_layers(adapter, decoders, X_train, y_train, X_test, n_classes)
        return layerwise_auc(probs, y_test)

    baseline = _aucs()
    skip = {}
    for m in range(adapter.n_layers):
        with skip_layer(adapter, m):
            skip[m] = _aucs()
    return {"baseline": baseline, "skip": skip}
