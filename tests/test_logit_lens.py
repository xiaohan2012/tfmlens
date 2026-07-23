"""Block 4 — logit_lens.

Decodes each captured residual into a per-layer prediction, using the adapter's
capability declarations (which token to read, transpose, pre-decoder norm).
"""

import torch
import torch.nn as nn

from tfm_lens.core.capture import capture_layers
from tfm_lens.core.logit_lens import logit_lens
from toys import ToyAdapter


class TestLogitLens:
    def test_one_prediction_per_layer(self, toy_adapter, toy_input, toy_decoders):
        with capture_layers(toy_adapter) as cache:
            toy_adapter.forward_frozen(toy_input, None, 5)
        preds = logit_lens(cache, toy_decoders, toy_adapter)
        assert len(preds) == toy_adapter.n_layers + 1 == 4
        assert all(p.shape == (2, ToyAdapter.N_CLASSES) for p in preds)

    def test_reads_label_token(self, toy_adapter):
        # token t is filled with the value t, so the read-out reveals which token.
        hidden = ToyAdapter.HIDDEN
        emb = torch.stack([torch.full((hidden,), float(t)) for t in range(3)]).unsqueeze(0)
        decoder = nn.Identity()

        # default label_token_index = -1 -> last token (value 2)
        out = logit_lens([emb], [decoder], toy_adapter)[0]
        torch.testing.assert_close(out, torch.full((1, hidden), 2.0))

        # changing the index reads a different token
        toy_adapter.label_token_index = 0
        out0 = logit_lens([emb], [decoder], toy_adapter)[0]
        torch.testing.assert_close(out0, torch.full((1, hidden), 0.0))

    def test_post_norm_applied(self):
        class DoublingAdapter(ToyAdapter):
            def post_norm(self, emb):
                return emb * 2

        adapter = DoublingAdapter()
        emb = torch.ones(1, 3, ToyAdapter.HIDDEN)
        out = logit_lens([emb], [nn.Identity()], adapter)[0]
        torch.testing.assert_close(out, torch.full((1, ToyAdapter.HIDDEN), 2.0))
