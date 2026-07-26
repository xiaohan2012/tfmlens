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
        # Mirror output_hook: cache the whole input (positional + keyword args) so
        # the adapter's readout picks the stream — including at depth 0. A single
        # stream packs to (x,) -> readout takes [0]; a double-stream model packs to
        # (support, query, ...) -> readout takes [1] (query sits at index 1 in both
        # a layer's input and its output).
        cache.append(_detach(args + tuple(kwargs.values())))  # depth 0: layer 0's input

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
