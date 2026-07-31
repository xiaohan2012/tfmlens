"""Layer interventions via temporary forward replacement.

Replaces the old ``task_info`` weight-threading through the model source: a
context manager swaps a layer's ``.forward`` and restores it on exit (model
agnostic, works through capture). ``nn.Module.__call__`` dispatches to
``self.forward``, so overriding the instance attribute is enough.
"""

from contextlib import contextmanager


@contextmanager
def skip_layer(adapter, idx):
    """Make layer ``idx`` the identity for the duration of the context."""
    layer = adapter.layers[idx]
    original = layer.forward
    # Pass the layer's whole input to identity_forward; the adapter decides what a
    # skipped layer returns (single residual, or both streams for a double-stream
    # model like Mitra).
    layer.forward = lambda *args, **kwargs: adapter.identity_forward(*args, **kwargs)
    try:
        yield
    finally:
        layer.forward = original


@contextmanager
def resample_layer(adapter, idx, donor_delta):
    """Replace layer ``idx``'s contribution with ``donor_delta`` for the context:
    the layer returns ``input_residual + donor_delta`` instead of its own output
    (resample ablation, #35). Upstream is untouched; downstream reacts freely.

    ``donor_delta`` must match the layer's input-residual shape (a Tensor, or a
    per-stream tuple for double-stream models) — build it with
    ``core.donor.build_donor_delta``. ``skip_layer`` is the special case δ:=0.
    """
    layer = adapter.layers[idx]
    original = layer.forward
    layer.forward = lambda *args, **kwargs: adapter.resample_forward(donor_delta, *args, **kwargs)
    try:
        yield
    finally:
        layer.forward = original


# TODO(exp5): loop_layer(adapter, idx, n) — call a layer's forward n times in a
# row, for the looping / repeated-layer experiment.
