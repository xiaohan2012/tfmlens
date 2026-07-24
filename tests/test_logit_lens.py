"""Block 4/5 — logit_lens across the 3D and 4D residual families.

The same logit_lens must handle both the toy (3D, [batch, seq, hidden]) and the
4D family (LimiX / TabPFN v2, [batch, seq, tokens, hidden]); the model-specific
bits live behind the adapter (select_label_token / post_norm / needs_transpose).
"""

import pytest
import torch
import torch.nn as nn

from tfm_lens.core.capture import capture_layers
from tfm_lens.core.logit_lens import logit_lens
from toys import ToyAdapter3D, ToyAdapter4D


class TestLogitLens:
    @pytest.mark.parametrize(
        "adapter_fx, input_fx, decoders_fx, n_classes",
        [
            ("toy_adapter", "toy_input", "toy_decoders", ToyAdapter3D.N_CLASSES),
            ("toy_adapter_4d", "toy_input_4d", "toy_decoders_4d", ToyAdapter4D.N_CLASSES),
        ],
    )
    def test_one_prediction_per_layer(self, request, adapter_fx, input_fx, decoders_fx, n_classes):
        adapter = request.getfixturevalue(adapter_fx)
        x = request.getfixturevalue(input_fx)
        decoders = request.getfixturevalue(decoders_fx)
        eval_pos = 3  # seq is 5 -> 2 test rows
        with capture_layers(adapter) as cache:
            adapter.forward_frozen(x, None, eval_pos)
        preds = logit_lens(cache, decoders, adapter, eval_pos)
        assert len(preds) == adapter.n_layers + 1 == 4
        assert all(p.shape == (2, 5 - eval_pos, n_classes) for p in preds)

    def test_decodes_only_test_rows(self, toy_adapter, toy_input, toy_decoders):
        # number of predicted rows = seq - eval_pos, not the whole sequence.
        with capture_layers(toy_adapter) as cache:
            toy_adapter.forward_frozen(toy_input, None, 4)
        preds = logit_lens(cache, toy_decoders, toy_adapter, eval_pos=4)
        assert preds[0].shape[1] == 5 - 4 == 1

    def test_4d_reads_the_label_token(self, toy_adapter_4d):
        # token t is filled with value t; the label token is the last one.
        hidden, tokens = ToyAdapter4D.HIDDEN, ToyAdapter4D.TOKENS
        per_token = torch.stack([torch.full((hidden,), float(t)) for t in range(tokens)])
        emb = per_token.reshape(1, 1, tokens, hidden).expand(1, 3, tokens, hidden).clone()
        h = toy_adapter_4d.select_label_token(emb)  # -> (1, 3, hidden)
        torch.testing.assert_close(h, torch.full((1, 3, hidden), float(tokens - 1)))

    def test_post_norm_applied(self):
        class DoublingAdapter(ToyAdapter3D):
            def post_norm(self, emb):
                return emb * 2

        adapter = DoublingAdapter()
        emb = torch.ones(1, 3, ToyAdapter3D.HIDDEN)  # (batch, seq, hidden)
        out = logit_lens([emb], [nn.Identity()], adapter, eval_pos=0)[0]
        torch.testing.assert_close(out, torch.full((1, 3, ToyAdapter3D.HIDDEN), 2.0))
