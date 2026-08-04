"""Direct (DE) and total (TE) effect on the **final native head** — path-patching
取法 B (issue #44, Part 1).

Two rulers for the Hydra DE–TE scatter, both read through the *same* native head
so they're comparable by construction (not per-layer tuned decoders):

- **DE(m) — path-patching direct effect.** The residual is additive, so freezing
  downstream at clean needs *no* per-layer forward:

      r_L^DE(m) = r_L^clean − a_m + ã_m

  One clean forward captures every contribution ``a_m`` (= r_m − r_{m-1}); DE(m)
  for all m is residual arithmetic through the TRUE head (``readout`` incl. final
  LN with σ recomputed + native decoder). Faithful 取法 B — the head runs on the
  whole residual. NOT logit-lens (取法 A = a fixed û on a single layer).

- **TE(m) — total effect.** Downstream must react → one real forward per m
  (``inject_delta`` / ``skip_layer``), read the ablated final residual through the
  same native head.

``ã_m`` (the replacement contribution): zero → 0; resample → a role-matched donor
δ, metric averaged over donors (mirrors ``self_repair._resample_skip``).

Both effects are reported as the **drop** ``clean − ablated`` on two fixed
coordinates (gt_logit, margin) — the scatter's x (DE) and y (TE).
"""

import numpy as np
import torch

from tfm_lens.adapters.base import ModelAdapter
from tfm_lens.core.capture import capture_layers
from tfm_lens.core.interventions import inject_delta, skip_layer
from tfm_lens.core.resample_ablation import build_donor_delta, donor_deltas, layer_deltas
from tfm_lens.evaluation.layerwise import (
    gt_logit_zscore_stats,
    layerwise_gt_logit,
    layerwise_margin,
)


def native_head_logits(adapter, residual, decoder, eval_pos, n_classes):
    """Final native head on a residual → ``[n_test, n_classes]`` logits (numpy).

    The **shared ruler** for DE and TE: ``adapter.readout`` (stream/token pick +
    final LN, σ recomputed on *this* residual) then the native decoder. 取法 B —
    the true head on the whole residual, not a fixed û on one layer's write.

    ``residual`` is in ``residual_of`` coordinates (a Tensor, or a
    ``(support, query)`` tuple for Mitra) — exactly what ``readout`` consumes.
    """
    with torch.no_grad():
        h = adapter.readout(residual, eval_pos)
        z = decoder(h.transpose(0, 1)).transpose(0, 1) if adapter.needs_transpose else decoder(h)
        return z[0, :, :n_classes].float().cpu().numpy()


def _clone(r):
    return tuple(t.clone() for t in r) if isinstance(r, tuple) else r.clone()


def _zeros_like(r):
    return tuple(torch.zeros_like(t) for t in r) if isinstance(r, tuple) else torch.zeros_like(r)


def _swap(r_full, a_m, delta):
    """r_full − a_m + delta, per stream (Tensor or ``(support, query)`` tuple)."""
    if isinstance(r_full, tuple):
        return tuple(r - a + d for r, a, d in zip(r_full, a_m, delta, strict=True))
    return r_full - a_m + delta


def _metrics(logits, y_test):
    """The two fixed coordinates off one final-head logit array."""
    return {
        "gt_logit": layerwise_gt_logit([logits], y_test)[0],
        "margin": layerwise_margin([logits], y_test)[0],
    }


def _drop(clean, ablated):
    """Effect = clean − ablated on each coordinate (the scatter value)."""
    return {k: clean[k] - ablated[k] for k in clean}


def _mean_metrics(per_donor_metrics):
    """Average each coordinate over donors."""
    keys = per_donor_metrics[0]
    return {k: float(np.mean([d[k] for d in per_donor_metrics])) for k in keys}


def direct_total_effect(
    adapter: ModelAdapter,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_test: torch.Tensor,
    y_test: np.ndarray,
    n_classes: int,
    ablation: str = "resample",
    donor_tables: list[tuple] | None = None,
    n_donors: int = 8,
    seed: int = 0,
) -> dict:
    """Per-layer DE(m) and TE(m) on the native head, for one table.

    Returns::

        {"clean": {"gt_logit", "margin"},          # final native head, no ablation
         "de":    {m: {"gt_logit", "margin"}},      # DE(m) = clean − path-patched
         "te":    {m: {"gt_logit", "margin"}},      # TE(m) = clean − ablate-and-react
         "zscore": {"mu", "sigma"}}                 # cross-task gt_logit normalizer

    DE and TE share one donor draw per (m, donor) → their difference (CE, Part 2)
    is a clean pairing. ``ablation``: ``"zero"`` (ã_m := 0) or ``"resample"``
    (needs ``donor_tables``; ã_m := role-matched donor δ, metric averaged).
    """
    y_test = np.asarray(y_test)
    decoder = adapter.decoder_template()
    device = next(decoder.parameters()).device
    eval_pos = X_train.shape[0]
    X = torch.cat([X_train, X_test], dim=0).unsqueeze(0).to(device)
    y = y_train.unsqueeze(0).to(device)

    # One clean forward: contributions a_m, per-layer input residuals (donor shaping),
    # final residual r_L, and the clean final-head metrics.
    with torch.no_grad(), capture_layers(adapter) as cache:
        adapter.forward_frozen(X, y, eval_pos)
        contributions = [_clone(d) for d in layer_deltas(adapter, cache)]
        target_inputs = [_clone(adapter.residual_of(cache[m])) for m in range(adapter.n_layers)]
        r_full = _clone(adapter.residual_of(cache[-1]))
        clean_logits = native_head_logits(adapter, r_full, decoder, eval_pos, n_classes)
    clean = _metrics(clean_logits, y_test)
    mu, sigma = gt_logit_zscore_stats([clean_logits], y_test)

    common = (adapter, decoder, X, y, eval_pos, n_classes, y_test, contributions, r_full)
    if ablation == "zero":
        de, te = _zero_effects(*common, clean)
    elif ablation == "resample":
        if not donor_tables:
            raise ValueError("resample ablation needs donor_tables (leave-one-out real tables)")
        de, te = _resample_effects(
            *common, clean, target_inputs, device, donor_tables, n_donors, seed
        )
    else:
        raise ValueError(f"unknown ablation {ablation!r}; expected 'zero' or 'resample'")

    return {"clean": clean, "de": de, "te": te, "zscore": {"mu": mu, "sigma": sigma}}


def _te_forward(adapter, decoder, X, y, eval_pos, n_classes, y_test, ctx):
    """Ablate one layer (``ctx`` = skip_layer/inject_delta), let downstream react,
    read the final residual through the native head → TE-side metrics."""
    with torch.no_grad(), ctx, capture_layers(adapter) as cache:
        adapter.forward_frozen(X, y, eval_pos)
        r_ab = adapter.residual_of(cache[-1])
        logits = native_head_logits(adapter, r_ab, decoder, eval_pos, n_classes)
    return _metrics(logits, y_test)


def _zero_effects(
    adapter, decoder, X, y, eval_pos, n_classes, y_test, contributions, r_full, clean
):
    """DE via ã_m := 0 (arithmetic); TE via ``skip_layer`` (real forward)."""
    de, te = {}, {}
    for m in range(adapter.n_layers):
        r_de = _swap(r_full, contributions[m], _zeros_like(contributions[m]))
        de_logits = native_head_logits(adapter, r_de, decoder, eval_pos, n_classes)
        de[m] = _drop(clean, _metrics(de_logits, y_test))
        te_ablated = _te_forward(
            adapter, decoder, X, y, eval_pos, n_classes, y_test, skip_layer(adapter, m)
        )
        te[m] = _drop(clean, te_ablated)
    return de, te


def _resample_effects(
    adapter,
    decoder,
    X,
    y,
    eval_pos,
    n_classes,
    y_test,
    contributions,
    r_full,
    clean,
    target_inputs,
    device,
    donor_tables,
    n_donors,
    seed,
):
    """DE and TE under resample: for each donor, build a role-matched δ per layer,
    use the **same** δ for DE (arithmetic) and TE (forward), average over donors."""
    k = min(n_donors, len(donor_tables))
    picks = np.random.default_rng(seed).permutation(len(donor_tables))[:k]

    de_donors, te_donors = [], []  # each: {m: {metric: val}}
    for di, i in enumerate(picks):
        Xd_train, yd_train, Xd_test = donor_tables[i]
        eval_pos_d = Xd_train.shape[0]
        Xd = torch.cat([Xd_train, Xd_test], dim=0).unsqueeze(0).to(device)
        yd = yd_train.unsqueeze(0).to(device)
        d_deltas = donor_deltas(adapter, Xd, yd, eval_pos_d)
        draw = np.random.default_rng(seed + 1 + di)  # per-donor position draw
        de_m, te_m = {}, {}
        for m in range(adapter.n_layers):
            delta = build_donor_delta(
                target_inputs[m], d_deltas[m], eval_pos, eval_pos_d, adapter.label_token_index, draw
            )
            r_de = _swap(r_full, contributions[m], delta)
            de_logits = native_head_logits(adapter, r_de, decoder, eval_pos, n_classes)
            de_m[m] = _metrics(de_logits, y_test)
            te_m[m] = _te_forward(
                adapter,
                decoder,
                X,
                y,
                eval_pos,
                n_classes,
                y_test,
                inject_delta(adapter, m, delta),
            )
        de_donors.append(de_m)
        te_donors.append(te_m)

    de = {
        m: _drop(clean, _mean_metrics([d[m] for d in de_donors])) for m in range(adapter.n_layers)
    }
    te = {
        m: _drop(clean, _mean_metrics([d[m] for d in te_donors])) for m in range(adapter.n_layers)
    }
    return de, te
