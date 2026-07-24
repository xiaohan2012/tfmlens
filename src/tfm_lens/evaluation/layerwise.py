"""Per-layer decode + ROC-AUC on one real table.

The evaluation counterpart of ``finetune``: given a single classification table
(train rows as in-context support, test rows to predict), run the frozen backbone
once, decode every captured depth with its (fine-tuned) decoder, and score each
depth against the true test labels. ``predict_layers`` is the model-heavy half;
``layerwise_auc`` is the cheap scoring half. Layer ablations stay orthogonal —
wrap ``predict_layers`` in ``skip_layer`` to get the intervened trajectory.
"""

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from tfm_lens.adapters.base import ModelAdapter
from tfm_lens.core.capture import capture_layers
from tfm_lens.core.logit_lens import logit_lens


def predict_layers(
    adapter: ModelAdapter,
    decoders: list[torch.nn.Module],
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_test: torch.Tensor,
    n_classes: int,
) -> list[np.ndarray]:
    """One frozen forward over the (train+test) table; decode each captured depth.

    Returns ``n_layers + 1`` probability arrays, each ``[n_test, n_classes]``.
    """
    device = next(decoders[0].parameters()).device
    eval_pos = X_train.shape[0]
    X = torch.cat([X_train, X_test], dim=0).unsqueeze(0).to(device)  # [1, seq, n_features]
    y = y_train.unsqueeze(0).to(device)  # [1, n_train]

    with torch.no_grad(), capture_layers(adapter) as cache:
        adapter.forward_frozen(X, y, eval_pos)
        preds = logit_lens(cache, decoders, adapter, eval_pos)  # each [1, n_test, >=n_classes]
        return [F.softmax(p[0, :, :n_classes], dim=-1).float().cpu().numpy() for p in preds]


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
