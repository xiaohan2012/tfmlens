"""Capture the per-layer residual stream via forward hooks.

Replaces the old ``self.out_embeddings = ...`` source surgery: a context manager
arms hooks on every layer, yields a cache that fills during the forward, and
removes the hooks on exit. Captures ``n_layers + 1`` taps — the input embedding
(depth 0) plus each layer's output — and stores the raw residual; token
selection / transpose / norm are logit_lens's job.
"""

from contextlib import contextmanager

import torch


def _pick(out):
    """Layer outputs are sometimes a tuple; the residual is the first element."""
    return out[0] if isinstance(out, tuple) else out


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
        cache.append(_pick(output).detach())

    handles.append(adapter.layers[0].register_forward_pre_hook(input_hook, with_kwargs=True))
    for layer in adapter.layers:
        handles.append(layer.register_forward_hook(output_hook))
    try:
        yield cache
    finally:
        for handle in handles:
            handle.remove()
