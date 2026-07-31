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
from tfm_lens.core.capture import capture_layers
from tfm_lens.core.donor import build_donor_delta, donor_deltas
from tfm_lens.core.interventions import resample_layer, skip_layer
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
    ablation: str = "zero",
    donor_tables: list[tuple] | None = None,
    n_donors: int = 8,
    seed: int = 0,
) -> dict:
    """Baseline + ablate-each-layer per-depth metric trajectories for one table.

    One forward per condition feeds every metric. Returns::

        {"baseline": {metric: [per depth]},
         "skip": {layer: {metric: [per depth]}},
         "zscore": {"mu": float, "sigma": float}}

    ``metric`` in {auc, gt_logit, margin}; ``zscore`` = clean-baseline stats for
    cross-task GT-logit normalization. (Key stays ``"skip"`` for both modes so the
    plots read either unchanged.)

    ``ablation`` (#35):

    - ``"zero"`` — skip the layer (δ:=0). Off-distribution; kept as the cross-check.
    - ``"resample"`` — δ:= a role-matched real-donor δ (on-manifold, norm-preserving).
      Needs ``donor_tables`` (leave-one-out real tables, each ``(X_train, y_train,
      X_test)``); draws ``n_donors`` without replacement, **averages the metric over
      donors** (not the δ), and adds ``{metric}_p25/_p75`` per layer for the IQR band.
    """

    def _logits():
        return predict_layers_logits(adapter, decoders, X_train, y_train, X_test, n_classes)

    base_logits = _logits()
    baseline = _all_metrics(base_logits, y_test)
    mu, sigma = gt_logit_zscore_stats(base_logits, y_test)

    if ablation == "zero":
        skip = {}
        for m in range(adapter.n_layers):
            with skip_layer(adapter, m):
                skip[m] = _all_metrics(_logits(), y_test)
    elif ablation == "resample":
        if not donor_tables:
            raise ValueError("resample ablation needs donor_tables (leave-one-out real tables)")
        skip = _resample_skip(
            adapter,
            decoders,
            X_train,
            y_train,
            X_test,
            y_test,
            n_classes,
            donor_tables,
            n_donors,
            seed,
        )
    else:
        raise ValueError(f"unknown ablation {ablation!r}; expected 'zero' or 'resample'")

    return {"baseline": baseline, "skip": skip, "zscore": {"mu": mu, "sigma": sigma}}


def _clone_residual(r):
    return tuple(t.clone() for t in r) if isinstance(r, tuple) else r.clone()


def _target_residuals(adapter, decoders, X_train, y_train, X_test):
    """Clean per-layer *input* residuals of the target forward — the shape + row-split
    the donor draw fills against (values unused). One extra forward per task."""
    device = next(decoders[0].parameters()).device
    eval_pos = X_train.shape[0]
    X = torch.cat([X_train, X_test], dim=0).unsqueeze(0).to(device)
    y = y_train.unsqueeze(0).to(device)
    with torch.no_grad(), capture_layers(adapter) as cache:
        adapter.forward_frozen(X, y, eval_pos)
        return [_clone_residual(adapter.residual_of(cache[m])) for m in range(adapter.n_layers)]


def _average_over_donors(per_donor: list[dict], n_layers: int) -> dict:
    """Per layer, mean each per-depth metric across donors (+ p25/p75 for the IQR band)."""
    names = list(per_donor[0][0].keys())
    out = {}
    for m in range(n_layers):
        entry = {}
        for name in names:
            stack = np.array([d[m][name] for d in per_donor])  # [n_donors, n_depths]
            entry[name] = stack.mean(0).tolist()
            entry[f"{name}_p25"] = np.percentile(stack, 25, axis=0).tolist()
            entry[f"{name}_p75"] = np.percentile(stack, 75, axis=0).tolist()
        out[m] = entry
    return out


def _resample_skip(
    adapter, decoders, X_train, y_train, X_test, y_test, n_classes, donor_tables, n_donors, seed
) -> dict:
    """Resample-ablate each layer: for ``n_donors`` leave-one-out donor tables, swap
    the layer's δ for a role-matched donor δ, then average the metric over donors."""

    def _logits():
        return predict_layers_logits(adapter, decoders, X_train, y_train, X_test, n_classes)

    device = next(decoders[0].parameters()).device
    eval_pos_t = X_train.shape[0]
    target_res = _target_residuals(adapter, decoders, X_train, y_train, X_test)

    k = min(n_donors, len(donor_tables))
    picks = np.random.default_rng(seed).permutation(len(donor_tables))[:k]  # without replacement

    per_donor = []
    for di, i in enumerate(picks):
        Xd_train, yd_train, Xd_test = donor_tables[i]
        eval_pos_d = Xd_train.shape[0]
        Xd = torch.cat([Xd_train, Xd_test], dim=0).unsqueeze(0).to(device)
        yd = yd_train.unsqueeze(0).to(device)
        d_deltas = donor_deltas(adapter, Xd, yd, eval_pos_d)
        draw = np.random.default_rng(seed + 1 + di)  # per-donor position draw
        by_layer = {}
        for m in range(adapter.n_layers):
            donor_delta = build_donor_delta(
                target_res[m], d_deltas[m], eval_pos_t, eval_pos_d, adapter.label_token_index, draw
            )
            with resample_layer(adapter, m, donor_delta):
                by_layer[m] = _all_metrics(_logits(), y_test)
        per_donor.append(by_layer)

    return _average_over_donors(per_donor, adapter.n_layers)
