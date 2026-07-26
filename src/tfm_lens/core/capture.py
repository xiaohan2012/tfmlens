"""Capture the per-layer residual stream via forward hooks.

Replaces the old ``self.out_embeddings = ...`` source surgery: a context manager
arms hooks on every layer, yields a cache that fills during the forward, and
removes the hooks on exit. Captures ``n_layers + 1`` taps — the input embedding
(depth 0) plus each layer's output — and stores the raw layer output; picking the
stream / token, transpose, and norm are the adapter's ``readout`` job.
"""

from contextlib import contextmanager

import torch


def _detach(out):
    """Detach a layer's raw output for caching, preserving tuple structure so the
    adapter's ``readout`` can pick the stream it needs (e.g. Mitra's query)."""
    if isinstance(out, tuple):
        return tuple(t.detach() if torch.is_tensor(t) else t for t in out)
    return out.detach()


@contextmanager
def capture_layers(adapter):
    cache: list[torch.Tensor] = []
    handles = []

    def input_hook(module, args, kwargs):
        # residual is the layer's first input — positional (LimiX) or, when the
        # layer is called by keyword (TabICL: block(q=x, ...)), the first kwarg.
        x = args[0] if args else next(iter(kwargs.values()))
        cache.append(x.detach())  # depth 0: what enters the first layer

    def output_hook(module, inputs, output):
        cache.append(_detach(output))

    handles.append(adapter.layers[0].register_forward_pre_hook(input_hook, with_kwargs=True))
    for layer in adapter.layers:
        handles.append(layer.register_forward_hook(output_hook))
    try:
        yield cache
    finally:
        for handle in handles:
            handle.remove()
