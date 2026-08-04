"""Direct (DE) and total (TE) effect on the **final decoder** — path-patching
取法 B (issue #44, Part 1).

Two rulers for the Hydra DE–TE scatter, both read through the *same* native decoder
so they're comparable by construction (not per-layer fine-tuned decoders):

- **DE(m) — path-patching direct effect.** The residual is additive, so freezing
  downstream at clean needs *no* per-layer forward:

      r_L^DE(m) = r_L^clean − a_m + ã_m

  ``a_m`` = the clean contribution layer m writes into the residual (r_m − r_{m-1});
  ``ã_m`` = what the ablation writes instead. One clean forward captures every
  ``a_m``; DE(m) for all m is residual arithmetic through the TRUE decoder
  (``readout`` incl. final LN with σ recomputed + native decoder). Faithful 取法 B —
  the decoder runs on the whole residual, NOT a fixed û on a single layer (取法 A).

- **TE(m) — total effect.** Downstream must react → one real forward per m
  (``inject_delta``), read the final layer through the same native decoder
  (reuses ``self_repair.native_final_logits``).

Ablation is **resample**: ``ã_m`` = a role-matched donor δ, metric averaged over
donors (reusing the #35 machinery). Both effects are the **drop** ``clean − ablated``
on two fixed coordinates (gt_logit, margin) — the scatter's x (DE) and y (TE).
"""

import numpy as np
import torch
import torch.nn as nn

from tfm_lens.adapters.base import ModelAdapter
from tfm_lens.core.capture import capture_layers
from tfm_lens.core.interventions import inject_delta
from tfm_lens.core.resample_ablation import build_donor_delta, donor_deltas, layer_deltas
from tfm_lens.evaluation.layerwise import (
    gt_logit_zscore_stats,
    layerwise_gt_logit,
    layerwise_margin,
)
from tfm_lens.evaluation.self_repair import native_final_logits
from tfm_lens.utils import clone_residual

# A residual stream: a Tensor, or a ``(support, query)`` pair for double-stream models.
Residual = torch.Tensor | tuple[torch.Tensor, ...]
Coords = dict[str, float]  # the two scatter axes, keyed {"gt_logit", "margin"}


def final_decoder_logits(
    adapter: ModelAdapter,
    residual: Residual,
    decoder: nn.Module,
    eval_pos: int,
    n_classes: int,
) -> np.ndarray:
    """The native decoder on a residual → ``[n_test, n_classes]`` logits (numpy).

    The **shared ruler** for DE and TE: ``adapter.readout`` (stream/token pick +
    final LN, σ recomputed on *this* residual) then the native decoder. 取法 B — the
    true decoder on the whole residual, not a fixed û on one layer's write.

    ``residual`` is in ``residual_of`` coordinates (a Tensor, or a
    ``(support, query)`` tuple for Mitra) — exactly what ``readout`` consumes.
    """
    with torch.no_grad():
        h = adapter.readout(residual, eval_pos)
        z = decoder(h.transpose(0, 1)).transpose(0, 1) if adapter.needs_transpose else decoder(h)
        return z[0, :, :n_classes].float().cpu().numpy()


def patched_residual(
    clean_residual: Residual, clean_contribution: Residual, replacement: Residual
) -> Residual:
    """The final residual with layer m's write swapped: ``r_L − a_m + ã_m`` (per stream).

    - ``clean_residual`` (r_L) — already holds upstream + the frozen-clean downstream writes.
    - ``clean_contribution`` (a_m) — what layer m actually wrote (r_m − r_{m-1}); removed.
    - ``replacement`` (ã_m) — what the ablation writes instead; added.
    """
    if isinstance(clean_residual, tuple):
        return tuple(
            r - a + d
            for r, a, d in zip(clean_residual, clean_contribution, replacement, strict=True)
        )
    return clean_residual - clean_contribution + replacement


def _coords(logits: np.ndarray, y_test: np.ndarray) -> Coords:
    """The two scatter axes off one final-decoder logit array. The metric math is
    reused from ``layerwise_*``; this only bundles gt_logit (x) + margin (y)."""
    return {
        "gt_logit": layerwise_gt_logit([logits], y_test)[0],
        "margin": layerwise_margin([logits], y_test)[0],
    }


def _effect(clean: Coords, ablated: Coords) -> Coords:
    """Effect = clean − ablated per coordinate (the scatter value)."""
    return {k: clean[k] - ablated[k] for k in clean}


def _mean_coords(per_donor: list[Coords]) -> Coords:
    """Average each coordinate over donors."""
    return {k: float(np.mean([c[k] for c in per_donor])) for k in per_donor[0]}


def direct_total_effect(
    adapter: ModelAdapter,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_test: torch.Tensor,
    y_test: np.ndarray,
    n_classes: int,
    donor_tables: list[tuple],
    n_donors: int = 8,
    seed: int = 0,
) -> dict:
    """Per-layer DE(m) and TE(m) on the native decoder, for one table (resample ablation).

    Returns::

        {"clean": {"gt_logit", "margin"},          # final decoder, no ablation
         "de":    {m: {"gt_logit", "margin"}},      # DE(m) = clean − path-patched
         "te":    {m: {"gt_logit", "margin"}},      # TE(m) = clean − ablate-and-react
         "zscore": {"mu", "sigma"}}                 # cross-task gt_logit normalizer

    DE and TE share one donor draw per (m, donor) → their difference (CE, Part 2) is a
    clean pairing. ``donor_tables`` = leave-one-out real tables; ``ã_m`` = a role-matched
    donor δ, metric averaged over ``n_donors`` draws.
    """
    if not donor_tables:
        raise ValueError("resample ablation needs donor_tables (leave-one-out real tables)")

    y_test = np.asarray(y_test)
    decoder = adapter.decoder_template()
    device = next(decoder.parameters()).device
    eval_pos = X_train.shape[0]
    X = torch.cat([X_train, X_test], dim=0).unsqueeze(0).to(device)
    y = y_train.unsqueeze(0).to(device)

    # One clean forward: clean contributions a_m, per-layer input residuals (donor
    # shaping), the final residual r_L, and the clean coordinates.
    with torch.no_grad(), capture_layers(adapter) as cache:
        adapter.forward_frozen(X, y, eval_pos)
        clean_contributions = [clone_residual(d) for d in layer_deltas(adapter, cache)]
        target_inputs = [
            clone_residual(adapter.residual_of(cache[m])) for m in range(adapter.n_layers)
        ]
        clean_residual = clone_residual(adapter.residual_of(cache[-1]))
        clean_logits = final_decoder_logits(adapter, clean_residual, decoder, eval_pos, n_classes)
    clean = _coords(clean_logits, y_test)
    mu, sigma = gt_logit_zscore_stats([clean_logits], y_test)

    k = min(n_donors, len(donor_tables))
    picks = np.random.default_rng(seed).permutation(len(donor_tables))[:k]

    de_donors: list[dict[int, Coords]] = []
    te_donors: list[dict[int, Coords]] = []
    for di, i in enumerate(picks):
        Xd_train, yd_train, Xd_test = donor_tables[i]
        eval_pos_d = Xd_train.shape[0]
        Xd = torch.cat([Xd_train, Xd_test], dim=0).unsqueeze(0).to(device)
        yd = yd_train.unsqueeze(0).to(device)
        donor_layer_deltas = donor_deltas(adapter, Xd, yd, eval_pos_d)
        draw = np.random.default_rng(seed + 1 + di)  # per-donor position draw
        de_m: dict[int, Coords] = {}
        te_m: dict[int, Coords] = {}
        for m in range(adapter.n_layers):
            replacement = build_donor_delta(
                target_inputs[m],
                donor_layer_deltas[m],
                eval_pos,
                eval_pos_d,
                adapter.label_token_index,
                draw,
            )
            # DE: arithmetic — swap m's write, read the native decoder (取法 B).
            r_patched = patched_residual(clean_residual, clean_contributions[m], replacement)
            de_logits = final_decoder_logits(adapter, r_patched, decoder, eval_pos, n_classes)
            de_m[m] = _coords(de_logits, y_test)
            # TE: real forward, m ablated + downstream free; same native decoder.
            with inject_delta(adapter, m, replacement):
                te_logits = native_final_logits(adapter, X_train, y_train, X_test, n_classes)
            te_m[m] = _coords(te_logits, y_test)
        de_donors.append(de_m)
        te_donors.append(te_m)

    layers = range(adapter.n_layers)
    de = {m: _effect(clean, _mean_coords([d[m] for d in de_donors])) for m in layers}
    te = {m: _effect(clean, _mean_coords([d[m] for d in te_donors])) for m in layers}
    return {"clean": clean, "de": de, "te": te, "zscore": {"mu": mu, "sigma": sigma}}
