"""Per-layer decode + metrics on one real table.

One frozen forward, score each captured depth vs the true test labels:

- ``predict_layers_logits`` — model-heavy: one forward -> raw logits per depth.
- ``layerwise_*`` — cheap reducers off those logits: AUC / GT-logit / margin.
- ablation: wrap the forward in ``skip_layer`` for the intervened trajectory.
"""

from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from scipy.special import softmax
from sklearn.metrics import roc_auc_score

from tfm_lens.adapters.base import ModelAdapter
from tfm_lens.core.capture import capture_layers
from tfm_lens.core.logit_lens import logit_lens


def load_decoders(path: str | Path, adapter: ModelAdapter) -> list[torch.nn.Module]:
    """One fine-tuned decoder per capture depth, from ``path/decoder_layer_{i}.pth``.

    Deep-copies the adapter's decoder template (``decoder_template`` returns the
    single shared module) so each depth gets an independent instance, then loads
    its trained weights. Returns ``n_layers + 1`` decoders, ordered by depth.
    """
    path = Path(path)
    decoders = []
    for i in range(adapter.n_layers + 1):
        d = deepcopy(adapter.decoder_template())
        d.load_state_dict(torch.load(path / f"decoder_layer_{i}.pth", map_location="cpu"))
        decoders.append(d)
    return decoders


def predict_layers_logits(
    adapter: ModelAdapter,
    decoders: list[torch.nn.Module],
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_test: torch.Tensor,
    n_classes: int,
) -> list[np.ndarray]:
    """One frozen forward; decode each depth to raw logits.

    - returns ``n_layers + 1`` arrays, each ``[n_test, n_classes]``.
    - one forward feeds every metric — don't re-run per metric; softmax is in ``predict_layers``.
    """
    device = next(decoders[0].parameters()).device
    eval_pos = X_train.shape[0]
    X = torch.cat([X_train, X_test], dim=0).unsqueeze(0).to(device)  # [1, seq, n_features]
    y = y_train.unsqueeze(0).to(device)  # [1, n_train]

    with torch.no_grad(), capture_layers(adapter) as cache:
        adapter.forward_frozen(X, y, eval_pos)
        preds = logit_lens(cache, decoders, adapter, eval_pos)  # each [1, n_test, >=n_classes]
        return [p[0, :, :n_classes].float().cpu().numpy() for p in preds]


def predict_layers(
    adapter: ModelAdapter,
    decoders: list[torch.nn.Module],
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_test: torch.Tensor,
    n_classes: int,
) -> list[np.ndarray]:
    """Softmax of :func:`predict_layers_logits`: probabilities ``[n_test, n_classes]`` per depth."""
    logits = predict_layers_logits(adapter, decoders, X_train, y_train, X_test, n_classes)
    return [softmax(z, axis=1) for z in logits]


def layerwise_auc(probs: list[np.ndarray], y_test: np.ndarray) -> list[float]:
    """ROC-AUC per depth: binary uses the positive-class probability, multiclass
    uses one-vs-rest macro averaging."""
    y_test = np.asarray(y_test)
    scores = []
    for p in probs:
        if p.shape[1] == 2:
            scores.append(roc_auc_score(y_test, p[:, 1]))
        else:
            scores.append(roc_auc_score(y_test, p, multi_class="ovr", average="macro"))
    return scores


def _gt_logit(z: np.ndarray, y_test: np.ndarray) -> np.ndarray:
    """True-class logit per row: ``z[i, y[i]]`` (``y_test`` must be an array)."""
    return z[np.arange(y_test.shape[0]), y_test]


def layerwise_gt_logit(logits: list[np.ndarray], y_test: np.ndarray) -> list[float]:
    """Per depth: mean over test rows of the true-class raw logit.

    Fixed coordinate (one-hot GT projection) -> additive -> feeds DE/IE; any class count.
    """
    y_test = np.asarray(y_test)
    return [float(_gt_logit(z, y_test).mean()) for z in logits]


def layerwise_margin(logits: list[np.ndarray], y_test: np.ndarray) -> list[float]:
    """Per depth: **median** over rows of ``z_true - z_other`` (log-odds). Binary only.

    Median, not mean: skip ablation is off-distribution and blows up a few rows'
    logits (LayerNorm rescale), so a mean gets hijacked by the tail while AUC
    (rank-based) stays flat. The median is a robust location — median margin > 0
    <=> most rows correct, tracking AUC instead of fighting it.

    Raises on multiclass: the 'max other' competitor drifts across runs -> not a fixed coordinate.
    """
    y_test = np.asarray(y_test)
    scores = []
    for z in logits:
        if z.shape[1] != 2:
            raise ValueError(f"layerwise_margin is binary-only; got {z.shape[1]} classes")
        scores.append(float(np.median(_gt_logit(z, y_test) - _gt_logit(z, 1 - y_test))))
    return scores


def gt_logit_zscore_stats(
    clean_logits: list[np.ndarray], y_test: np.ndarray
) -> tuple[float, float]:
    """``(mu, sigma)`` to z-score a task's GT-logit trajectory onto a common cross-task unit.

    - both from the clean final-layer true-class logit over test rows (the task's output scale).
    - cross-row spread only -> doesn't fold in the depth trend we plot.
    - raises if sigma == 0 (degenerate task).
    """
    y_test = np.asarray(y_test)
    gt = _gt_logit(clean_logits[-1], y_test)  # [-1] = final layer (the task's output scale)
    mu, sigma = float(gt.mean()), float(gt.std())
    if sigma == 0.0:
        raise ValueError("degenerate task: final-layer GT-logit has zero spread across rows")
    return mu, sigma
