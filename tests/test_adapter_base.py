"""The ModelAdapter contract.

Each test pins a property some downstream module (capture / interventions /
logit_lens / training) will rely on; the comment names which one. capture etc.
do not exist yet, so this suite is red until base.py lands.
"""

import pytest
import torch

from tfm_lens.adapters.base import ModelAdapter
from toys import ToyAdapter3D, ToyAdapter4D


class TestModelAdapter:
    def test_layers_expose_the_backbone_blocks(self, toy_adapter):
        # capture hooks these and interventions swap their .forward, so `layers`
        # must be the real block modules, not copies.
        assert toy_adapter.layers == list(toy_adapter.backbone.blocks)
        assert len(toy_adapter.layers) == 3

    def test_forward_frozen_runs_the_layers(self, toy_adapter, toy_input):
        # capture's lifeline: if forward_frozen doesn't run the layers, its hooks
        # never fire and the cache stays empty.
        fired = []
        handle = toy_adapter.layers[0].register_forward_hook(lambda *_: fired.append(True))
        try:
            toy_adapter.forward_frozen(toy_input, y_train=None, eval_pos=5)
        finally:
            handle.remove()
        assert fired, "forward_frozen did not run the layers"

    def test_decoder_template_maps_hidden_to_classes(self, toy_adapter):
        # logit_lens / training deepcopy this head; it must map hidden -> n_classes.
        decoder = toy_adapter.decoder_template()
        out = decoder(torch.randn(2, ToyAdapter3D.HIDDEN))
        assert out.shape == (2, ToyAdapter3D.N_CLASSES)

    def test_readout_defaults_to_stream_and_test_rows(self, toy_adapter):
        # logit_lens calls readout to turn a raw layer output into decoder-ready
        # test-row residuals; the 3D default picks the stream and keeps test rows.
        emb = torch.randn(1, 5, ToyAdapter3D.HIDDEN)  # (batch, seq, hidden)
        torch.testing.assert_close(toy_adapter.readout(emb, eval_pos=3), emb[:, 3:])

    def test_to_is_chainable_and_moves_the_backbone(self, toy_adapter):
        # finetune_decoders co-locates the frozen backbone with the decoders and
        # inputs on config.device via this; it must move params and return self.
        assert toy_adapter.to("cpu") is toy_adapter
        assert all(p.device.type == "cpu" for p in toy_adapter.layers[0].parameters())

    def test_incomplete_subclass_cannot_instantiate(self):
        # the mold's teeth: a subclass missing an abstractmethod can't be built,
        # so no adapter can silently skip something a downstream module needs.
        class Incomplete(ModelAdapter):
            pass

        with pytest.raises(TypeError):
            Incomplete()


class TestResampleHooks:
    """residual_of / resample_forward — the coordinate resample ablation (#35)
    captures a contribution δ in and applies a donor δ in.

    Key invariant: ``δ = residual_of(out) − residual_of(in)`` and
    ``resample_forward(δ, in) == out`` — re-applying a layer's *own* δ reproduces
    its output, so a *donor* δ is a clean drop-in (only the contribution swaps).
    """

    # ---- residual_of: pick the stream, keep every row/token (no slice, no norm) ----
    def test_residual_of_single_stream(self, toy_adapter):
        x = torch.randn(2, 5, ToyAdapter3D.HIDDEN)
        # bare tensor (a layer output) -> itself; capture's input tuple (x,) -> x.
        torch.testing.assert_close(toy_adapter.residual_of(x), x)
        torch.testing.assert_close(toy_adapter.residual_of((x,)), x)

    def test_residual_of_4d_keeps_token_axis(self, toy_adapter_4d):
        # bucketing must be able to index the label token, so the token axis stays.
        res = torch.randn(2, 5, ToyAdapter4D.TOKENS, ToyAdapter4D.HIDDEN)
        r = toy_adapter_4d.residual_of((res, None, None))  # LimiX-style 3-tuple
        torch.testing.assert_close(r, res)
        assert r.shape == (2, 5, ToyAdapter4D.TOKENS, ToyAdapter4D.HIDDEN)

    def test_residual_of_double_stream_returns_both(self, toy_adapter_double_stream):
        s = torch.randn(2, 3, ToyAdapter3D.HIDDEN)
        q = torch.randn(2, 2, ToyAdapter3D.HIDDEN)
        rs, rq = toy_adapter_double_stream.residual_of((s, q))
        torch.testing.assert_close(rs, s)
        torch.testing.assert_close(rq, q)

    # ---- resample_forward: input + donor δ, in the layer's own output shape ----
    def test_resample_roundtrip_single_stream(self, toy_adapter):
        x = torch.randn(2, 5, ToyAdapter3D.HIDDEN)
        out = toy_adapter.layers[1](x)
        delta = toy_adapter.residual_of(out) - toy_adapter.residual_of((x,))
        torch.testing.assert_close(toy_adapter.resample_forward(delta, x), out)

    def test_resample_roundtrip_4d_keeps_tuple_shape(self, toy_adapter_4d):
        x = torch.randn(2, 5, ToyAdapter4D.TOKENS, ToyAdapter4D.HIDDEN)
        out = toy_adapter_4d.layers[1](x)  # (residual, None, None)
        delta = toy_adapter_4d.residual_of(out) - toy_adapter_4d.residual_of((x,))
        res = toy_adapter_4d.resample_forward(delta, x)
        assert isinstance(res, tuple) and len(res) == 3 and res[1] is None and res[2] is None
        torch.testing.assert_close(res[0], out[0])

    def test_resample_roundtrip_double_stream(self, toy_adapter_double_stream):
        a = toy_adapter_double_stream
        s = torch.randn(2, 3, ToyAdapter3D.HIDDEN)
        q = torch.randn(2, 2, ToyAdapter3D.HIDDEN)
        out = a.layers[1](s, q)  # (lin(s), lin(q))
        r_out, r_in = a.residual_of(out), a.residual_of((s, q))
        delta = (r_out[0] - r_in[0], r_out[1] - r_in[1])
        res = a.resample_forward(delta, s, q)
        torch.testing.assert_close(res[0], out[0])
        torch.testing.assert_close(res[1], out[1])

    def test_resample_applies_donor_delta(self, toy_adapter):
        # a *foreign* δ swaps the contribution: output = input + donor δ, exactly.
        x = torch.randn(2, 5, ToyAdapter3D.HIDDEN)
        donor = torch.randn(2, 5, ToyAdapter3D.HIDDEN)
        torch.testing.assert_close(toy_adapter.resample_forward(donor, x), x + donor)
