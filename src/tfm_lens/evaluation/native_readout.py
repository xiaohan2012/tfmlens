"""Read the final output through the model's **own** decoder.

Distinct from the fine-tuned per-depth probes (``layerwise.py``): this is the ruler
the model actually decides with, and the one both DE and TE are read on.

Shared on purpose — used by ``path_patching`` (DE/TE) and by ``balef_exp6``
(per-dataset normalizer), so it belongs to neither.
"""

import numpy as np
import torch
from scipy.special import softmax

from tfm_lens.adapters.base import ModelAdapter
from tfm_lens.evaluation.layerwise import layerwise_auc, predict_layers_logits


def native_final_logits(
    adapter: ModelAdapter,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_test: torch.Tensor,
    n_classes: int,
) -> np.ndarray:
    """The model's own decoder on the **final** layer → ``[n_test, n_classes]`` logits.

    Wrap the call in a ``skip_layer`` / ``inject_delta`` context to read it under ablation.
    """
    # predict_layers_logits wants one decoder per depth; here every depth is the one
    # shared native decoder (the same instance repeated). Fine because we read only the
    # final layer ([-1]) and the decoder is stateless — depths never diverge. Do NOT
    # reuse this list to read *intermediate* depths expecting independent decoders.
    native = [adapter.decoder_template()] * (adapter.n_layers + 1)
    return predict_layers_logits(adapter, native, X_train, y_train, X_test, n_classes)[-1]


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
    logits = native_final_logits(adapter, X_train, y_train, X_test, n_classes)
    return layerwise_auc([softmax(logits, axis=1)], y_test)[0]
