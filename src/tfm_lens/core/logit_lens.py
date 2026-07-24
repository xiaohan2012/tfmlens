"""Decode captured residuals into per-layer predictions.

The tabular logit lens: pair each captured residual with its trained decoder and
read out a prediction, using the adapter's capability declarations to bridge
model-specific quirks (which token holds the label, whether the decoder wants a
transposed layout, and any pre-decoder normalization).
"""


def logit_lens(cache, decoders, adapter, eval_pos):
    """One prediction per depth for the test rows (those at/after ``eval_pos``).

    ``len(cache) == len(decoders) == n_layers + 1``. Each prediction is
    ``[batch, n_test, n_classes]``.
    """
    preds = []
    for emb, decoder in zip(cache, decoders, strict=True):
        h = adapter.select_label_token(emb)  # -> [batch, seq, hidden]
        h = adapter.post_norm(h)
        h = h[:, eval_pos:]  # keep only the test rows
        if adapter.needs_transpose:
            preds.append(decoder(h.transpose(0, 1)).transpose(0, 1))
        else:
            preds.append(decoder(h))
    return preds
