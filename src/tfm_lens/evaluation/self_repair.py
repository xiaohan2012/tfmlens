"""Ablation sweep for the self-repair analysis.

Runs the frozen forward once per condition — baseline, then skipping each layer
in turn — decoding every depth with the fine-tuned decoders and scoring test-row
AUC. Reuses predict_layers / layerwise_auc; the skip is orthogonal (skip_layer
just wraps the same call).
"""

from tfm_lens.core.interventions import skip_layer
from tfm_lens.evaluation.layerwise import layerwise_auc, predict_layers


def ablation_sweep(adapter, decoders, X_train, y_train, X_test, y_test, n_classes):
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
