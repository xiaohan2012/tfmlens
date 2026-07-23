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
    layer.forward = lambda x, *args, **kwargs: x
    try:
        yield
    finally:
        layer.forward = original


# TODO(exp5): loop_layer(adapter, idx, n) — call a layer's forward n times in a
# row, for the looping / repeated-layer experiment.
