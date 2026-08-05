"""Donor δ for resample ablation (#35).

Resample ablation replaces a layer's contribution δ with a **real donor's** δ,
role-matched (issue #35 "Design settled"):

- **row axis (all models):** context←context, query←query.
- **token axis (4D only):** label←label (decoder reads only the label token),
  feature←feature random. 3D models have no token axis (``label_token_index=None``).

Two steps:

- ``donor_deltas`` — one donor table's per-layer δ, ``δ = residual_of(out) − residual_of(in)``.
- ``build_donor_delta`` — a **target-shaped** δ: each position filled by a random
  draw from the donor's same-role positions (with replacement — donor/target
  row & feature counts differ).

The N=8 donor draws + metric averaging live in the sweep (``balef_exp6.py``);
this module builds one ``donor_delta`` per (donor, layer).
"""

import numpy as np
import torch

from tfm_lens.core.capture import capture_layers


def _sub(out, inp):
    """δ = out − in, preserving stream structure (Tensor, or ``(support, query)``)."""
    if isinstance(out, tuple):
        return tuple(o - i for o, i in zip(out, inp, strict=True))
    return out - inp


def layer_deltas(adapter, cache) -> list:
    """Per-layer contribution δ from a capture cache: ``residual_of(out) − residual_of(in)``.

    ``len == n_layers``; each entry a Tensor (single stream) or ``(support, query)``
    tuple (double stream), in raw residual coordinates (all rows / tokens, no norm).
    """
    return [
        _sub(adapter.residual_of(cache[m + 1]), adapter.residual_of(cache[m]))
        for m in range(adapter.n_layers)
    ]


def donor_deltas(adapter, X, y_train, eval_pos) -> list:
    """Run one donor table through the frozen model → its per-layer δ (``layer_deltas``)."""
    with torch.no_grad(), capture_layers(adapter) as cache:
        adapter.forward_frozen(X, y_train, eval_pos)
        return layer_deltas(adapter, cache)


def _blocks(residual, eval_pos) -> list:
    """Split a residual into ``[context, query]`` blocks by row-role.

    - single stream: split the seq axis (dim 1) at ``eval_pos``.
    - double stream ``(support, query)``: the streams *are* the split.
    """
    if isinstance(residual, tuple):
        return [residual[0], residual[1]]
    return [residual[:, :eval_pos], residual[:, eval_pos:]]


def _fill_like(target, pool, rng) -> torch.Tensor:
    """Tensor shaped like ``target`` (``[..., hidden]``), each cell = a random row of
    ``pool`` (``[N, hidden]``), drawn uniformly with replacement."""
    lead = target.shape[:-1]
    idx = torch.from_numpy(rng.integers(0, pool.shape[0], size=int(np.prod(lead)))).to(pool.device)
    return pool[idx].reshape(*lead, pool.shape[-1])


def _fill_block(t_block, d_block, label_index, rng) -> torch.Tensor:
    """Fill one role block: each target cell ← a random donor cell of matching token-role.

    - ``label_index is None`` (3D): one bucket over rows.
    - else (4D): label column vs feature columns are separate buckets.
    """
    hidden = t_block.shape[-1]
    if label_index is None:
        return _fill_like(t_block, d_block.reshape(-1, hidden), rng)

    # Target and donor token counts differ across tables (different #features), so
    # resolve the label position and feature indices *per block* — never index the
    # donor with the target's token indices (that overruns when the donor is narrower).
    t_label = label_index % t_block.shape[-2]
    d_label = label_index % d_block.shape[-2]
    t_feats = [t for t in range(t_block.shape[-2]) if t != t_label]
    d_feats = [t for t in range(d_block.shape[-2]) if t != d_label]
    out = torch.empty_like(t_block)
    out[..., t_label, :] = _fill_like(
        t_block[..., t_label, :], d_block[..., d_label, :].reshape(-1, hidden), rng
    )
    out[..., t_feats, :] = _fill_like(
        t_block[..., t_feats, :], d_block[..., d_feats, :].reshape(-1, hidden), rng
    )
    return out


def build_donor_delta(target_residual, donor_delta, eval_pos_t, eval_pos_d, label_index, rng):
    """A target-shaped donor δ: every position filled by a random draw from the
    donor's **same-role** positions (row-role always; token-role for 4D).

    - ``target_residual`` — ``residual_of(target layer input)``: gives shape + the row split.
    - ``donor_delta`` — one donor's δ at this layer (from :func:`donor_deltas`).
    - ``eval_pos_t`` / ``eval_pos_d`` — target / donor train/test split (row-role boundary).
    - ``label_index`` — ``adapter.label_token_index`` (None for 3D).
    """
    t_blocks = _blocks(target_residual, eval_pos_t)
    d_blocks = _blocks(donor_delta, eval_pos_d)
    filled = [_fill_block(t, d, label_index, rng) for t, d in zip(t_blocks, d_blocks, strict=True)]
    if isinstance(target_residual, tuple):
        return tuple(filled)
    return torch.cat(filled, dim=1)
