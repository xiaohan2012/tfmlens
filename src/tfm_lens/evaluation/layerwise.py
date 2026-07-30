"""Per-layer decode + metrics on one real table.

The evaluation counterpart of ``finetune``: given a single classification table
(train rows as in-context support, test rows to predict), run the frozen backbone
once, decode every captured depth with its (fine-tuned) decoder, and score each
depth against the true test labels. ``predict_layers_logits`` is the model-heavy
half (one forward -> raw logits per depth); the ``layerwise_*`` reducers are the
cheap scoring half (AUC / GT-logit / margin, all off the same logits). Layer
ablations stay orthogonal — wrap the forward in ``skip_layer`` for the intervened
trajectory.
"""

from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
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
    """One frozen forward over the (train+test) table; decode each depth to RAW logits.

    Returns ``n_layers + 1`` logit arrays, each ``[n_test, n_classes]``. The metric
    layer (AUC / GT-logit / margin) reduces these; softmax lives in ``predict_layers``.
    One forward feeds every metric — don't re-run per metric.
    """
    device = next(decoders[0].parameters()).device
    eval_pos = X_train.shape[0]
    X = torch.cat([X_train, X_test], dim=0).unsqueeze(0).to(device)  # [1, seq, n_features]
    y = y_train.unsqueeze(0).to(device)  # [1, n_train]

    with torch.no_grad(), capture_layers(adapter) as cache:
        adapter.forward_frozen(X, y, eval_pos)
        preds = logit_lens(cache, decoders, adapter, eval_pos)  # each [1, n_test, >=n_classes]
        return [p[0, :, :n_classes].float().cpu().numpy() for p in preds]


def _softmax(logits: np.ndarray) -> np.ndarray:
    """Row-wise softmax, numerically stable."""
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def predict_layers(
    adapter: ModelAdapter,
    decoders: list[torch.nn.Module],
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_test: torch.Tensor,
    n_classes: int,
) -> list[np.ndarray]:
    """One frozen forward over the (train+test) table; decode each captured depth.

    Returns ``n_layers + 1`` probability arrays, each ``[n_test, n_classes]`` —
    softmax of :func:`predict_layers_logits`.
    """
    logits = predict_layers_logits(adapter, decoders, X_train, y_train, X_test, n_classes)
    return [_softmax(z) for z in logits]


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


def layerwise_gt_logit(logits: list[np.ndarray], y_test: np.ndarray) -> list[float]:
    """Per depth: mean over test rows of the true-class raw logit.

    The fixed-coordinate metric (projection onto the one-hot GT direction) — additive,
    so it feeds the DE/IE decomposition; works for any class count.
    """
    y_test = np.asarray(y_test)
    rows = np.arange(y_test.shape[0])
    return [float(z[rows, y_test].mean()) for z in logits]


def layerwise_margin(logits: list[np.ndarray], y_test: np.ndarray) -> list[float]:
    """Binary only. Per depth: mean over rows of ``z_true - z_other`` (the logit
    difference / log-odds). Raises on non-binary logits — the multiclass 'max other'
    competitor drifts across runs and breaks the fixed-coordinate property.
    """
    y_test = np.asarray(y_test)
    rows = np.arange(y_test.shape[0])
    scores = []
    for z in logits:
        if z.shape[1] != 2:
            raise ValueError(f"layerwise_margin is binary-only; got {z.shape[1]} classes")
        scores.append(float((z[rows, y_test] - z[rows, 1 - y_test]).mean()))
    return scores


def gt_logit_zscore_stats(
    clean_logits: list[np.ndarray], y_test: np.ndarray
) -> tuple[float, float]:
    """``(mu, sigma)`` for z-scoring a task's per-layer GT-logit trajectory so tasks
    share a common unit before cross-task aggregation.

    Both come from the **clean final-layer** true-class logit over test rows — the
    task's output-logit unit (cross-row spread), which does not fold in the depth
    trend we plot. Raises if sigma is 0 (degenerate task).
    """
    y_test = np.asarray(y_test)
    rows = np.arange(y_test.shape[0])
    gt = clean_logits[-1][rows, y_test]
    mu, sigma = float(gt.mean()), float(gt.std())
    if sigma == 0.0:
        raise ValueError("degenerate task: final-layer GT-logit has zero spread across rows")
    return mu, sigma
