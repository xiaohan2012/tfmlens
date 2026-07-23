"""Decode captured residuals into per-layer predictions.

The tabular logit lens: pair each captured residual with its trained decoder and
read out a prediction, using the adapter's capability declarations to bridge
model-specific quirks (which token holds the label, whether the decoder wants a
transposed layout, and any pre-decoder normalization).
"""


def logit_lens(cache, decoders, adapter):
    """One prediction per depth. ``len(cache) == len(decoders) == n_layers + 1``."""
    preds = []
    for emb, decoder in zip(cache, decoders, strict=True):
        h = emb[:, adapter.label_token_index]
        if adapter.needs_transpose:
            h = h.transpose(0, 1)
        h = adapter.post_norm(h)
        preds.append(decoder(h))
    return preds
