"""Path-patching DE / TE (取法 B) — ``evaluation.direct_effect``.

Headline invariant: the DE **arithmetic** ``patched_residual`` (``r_L − a_m + ã_m``,
no per-layer forward) reproduces a real **frozen-downstream forward** (ablate m to
ã_m, pin every downstream layer to its clean contribution). Holds for *any* ã_m, any
head, any stream layout ⇒ the O(1)-forward shortcut is exact.

Also: on a linear head DE collapses to the logit-lens closed form (取法 A == B); and
the resample sweep runs with the right shape/keys.
"""

from contextlib import ExitStack

import numpy as np
import pytest
import torch
import torch.nn as nn

from tfm_lens.core.capture import capture_layers
from tfm_lens.core.interventions import inject_delta
from tfm_lens.core.resample_ablation import layer_deltas
from tfm_lens.evaluation.direct_effect import (
    direct_total_effect,
    final_decoder_logits,
    patched_residual,
)
from tfm_lens.utils import clone_residual
from toys import ToyAdapter3D, ToyAdapter4D


# --- binary, stable-decoder toys (margin is binary-only; decoder must be one fixed instance) ---
def _binary(cls):
    """Subclass a toy so its decoder is binary and the *same* instance every call
    (real adapters return their shared decoder; the base toy makes a fresh Linear)."""

    class _Bin(cls):
        N_CLASSES = 2

        def __init__(self):
            super().__init__()
            self._dec = nn.Linear(cls.HIDDEN, 2)

        def decoder_template(self):
            return self._dec

    _Bin.__name__ = f"Binary{cls.__name__}"
    return _Bin


BinaryToy3D = _binary(ToyAdapter3D)
BinaryToy4D = _binary(ToyAdapter4D)


class _DoubleBlock(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.lin = nn.Linear(hidden, hidden)

    def forward(self, support, query):
        return self.lin(support), self.lin(query)


class _DoubleBackbone(nn.Module):
    def __init__(self, n_layers, hidden):
        super().__init__()
        self.blocks = nn.ModuleList(_DoubleBlock(hidden) for _ in range(n_layers))

    def forward(self, support, query):
        for blk in self.blocks:
            support, query = blk(support, query)
        return query


class BinaryDoubleStreamToy(ToyAdapter3D):
    """Binary double-stream toy (Mitra family): layers take/return (support, query),
    readout takes the query stream — exercises the tuple ``patched_residual`` path."""

    N_CLASSES = 2

    def __init__(self):
        self.backbone = _DoubleBackbone(self.N_LAYERS, self.HIDDEN)
        self._dec = nn.Linear(self.HIDDEN, 2)

    def forward_frozen(self, X, y_train, eval_pos):
        with torch.no_grad():
            self.backbone(X[:, :eval_pos], X[:, eval_pos:])

    def decoder_template(self):
        return self._dec

    def readout(self, out, eval_pos):
        return out[1]  # query stream (already the test rows)

    def identity_forward(self, *args, **kwargs):
        return args[0], args[1]

    def residual_of(self, layer_output):
        return layer_output[0], layer_output[1]

    def inject_delta_forward(self, donor_delta, *args, **kwargs):
        return args[0] + donor_delta[0], args[1] + donor_delta[1]


# (make_adapter, per-row input shape) — 3D / 4D / double-stream cover every layout.
_CASES = [
    pytest.param(BinaryToy3D, (ToyAdapter3D.HIDDEN,), id="3d"),
    pytest.param(BinaryToy4D, (ToyAdapter4D.TOKENS, ToyAdapter4D.HIDDEN), id="4d"),
    pytest.param(BinaryDoubleStreamToy, (ToyAdapter3D.HIDDEN,), id="double_stream"),
]


def _inputs(row_shape, n_train=4, n_test=3, seed=0):
    g = torch.Generator().manual_seed(seed)
    Xtr = torch.randn(n_train, *row_shape, generator=g)
    Xte = torch.randn(n_test, *row_shape, generator=g)
    ytr = torch.randint(0, 2, (n_train,), generator=g).float()
    y_test = torch.randint(0, 2, (n_test,), generator=g).numpy()
    return Xtr, ytr, Xte, y_test


def _rand_like(r, gen):
    """An arbitrary replacement ã_m shaped like a residual (Tensor or stream tuple)."""
    if isinstance(r, tuple):
        return tuple(torch.randn(t.shape, generator=gen) for t in r)
    return torch.randn(r.shape, generator=gen)


def _clean_forward(adapter, Xtr, ytr, Xte):
    """Clean per-layer contributions a_m + final residual r_L, off one forward."""
    eval_pos = Xtr.shape[0]
    X = torch.cat([Xtr, Xte], dim=0).unsqueeze(0)
    with torch.no_grad(), capture_layers(adapter) as cache:
        adapter.forward_frozen(X, ytr.unsqueeze(0), eval_pos)
        contribs = [clone_residual(d) for d in layer_deltas(adapter, cache)]
        r_full = clone_residual(adapter.residual_of(cache[-1]))
    return X, ytr.unsqueeze(0), eval_pos, contribs, r_full


def _frozen_forward_logits(adapter, X, y, eval_pos, decoder, m, replacement, contribs, n_classes):
    """DE the *slow* way: ablate m to ``replacement``, pin every downstream layer to
    its clean contribution, run a real forward, read the native decoder."""
    with ExitStack() as stack:
        stack.enter_context(inject_delta(adapter, m, replacement))
        for layer in range(m + 1, adapter.n_layers):
            stack.enter_context(inject_delta(adapter, layer, contribs[layer]))
        with torch.no_grad(), capture_layers(adapter) as cache:
            adapter.forward_frozen(X, y, eval_pos)
            r = adapter.residual_of(cache[-1])
            return final_decoder_logits(adapter, r, decoder, eval_pos, n_classes)


@pytest.mark.parametrize("make_adapter, row_shape", _CASES)
def test_patched_residual_equals_frozen_forward(make_adapter, row_shape):
    """The O(1)-forward arithmetic == the brute-force frozen-downstream forward, for
    an arbitrary replacement ã_m."""
    adapter = make_adapter()
    Xtr, ytr, Xte, _ = _inputs(row_shape)
    decoder = adapter.decoder_template()
    X, y, eval_pos, contribs, r_full = _clean_forward(adapter, Xtr, ytr, Xte)
    gen = torch.Generator().manual_seed(1)
    for m in range(adapter.n_layers):
        replacement = _rand_like(contribs[m], gen)
        arith = final_decoder_logits(
            adapter, patched_residual(r_full, contribs[m], replacement), decoder, eval_pos, 2
        )
        frozen = _frozen_forward_logits(
            adapter, X, y, eval_pos, decoder, m, replacement, contribs, 2
        )
        np.testing.assert_allclose(arith, frozen, atol=1e-5)


def test_de_matches_logit_lens_on_linear_head():
    """Linear head ⇒ clean − patched = decode(a_m − ã_m) − decode(0) (bias cancels) = 取法 A."""
    adapter = BinaryToy3D()
    Xtr, ytr, Xte, _ = _inputs((ToyAdapter3D.HIDDEN,))
    decoder = adapter.decoder_template()
    _, _, eval_pos, contribs, r_full = _clean_forward(adapter, Xtr, ytr, Xte)
    clean_logits = final_decoder_logits(adapter, r_full, decoder, eval_pos, 2)
    zero_logits = final_decoder_logits(adapter, torch.zeros_like(contribs[0]), decoder, eval_pos, 2)
    gen = torch.Generator().manual_seed(2)
    for m in range(adapter.n_layers):
        replacement = _rand_like(contribs[m], gen)
        patched = final_decoder_logits(
            adapter, patched_residual(r_full, contribs[m], replacement), decoder, eval_pos, 2
        )
        lens = final_decoder_logits(adapter, contribs[m] - replacement, decoder, eval_pos, 2)
        np.testing.assert_allclose(clean_logits - patched, lens - zero_logits, atol=1e-5)


def test_resample_runs_with_shape_and_keys():
    """Resample sweep: donor tables (other toy tables) → de/te for every layer, both
    coordinates, all finite."""
    adapter = BinaryToy3D()
    Xtr, ytr, Xte, y_test = _inputs((ToyAdapter3D.HIDDEN,), seed=1)
    donor_tables = [
        (dtr, dytr, dte)
        for dtr, dytr, dte, _ in (_inputs((ToyAdapter3D.HIDDEN,), seed=s) for s in (2, 3, 4))
    ]
    res = direct_total_effect(adapter, Xtr, ytr, Xte, y_test, 2, donor_tables, n_donors=3, seed=0)
    for effect in ("de", "te"):
        assert set(res[effect]) == set(range(adapter.n_layers))
        for m in range(adapter.n_layers):
            for coord in ("gt_logit", "margin"):
                assert np.isfinite(res[effect][m][coord])


def test_resample_needs_donor_tables():
    adapter = BinaryToy3D()
    Xtr, ytr, Xte, y_test = _inputs((ToyAdapter3D.HIDDEN,))
    with pytest.raises(ValueError, match="donor_tables"):
        direct_total_effect(adapter, Xtr, ytr, Xte, y_test, 2, donor_tables=[])
