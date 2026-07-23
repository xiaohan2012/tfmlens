"""Block 1 — the ModelAdapter contract.

Each test pins a property some downstream module (capture / interventions /
logit_lens / training) will rely on; the comment names which one. capture etc.
do not exist yet, so this suite is red until base.py lands.
"""

import pytest
import torch

from tfm_lens.adapters.base import ModelAdapter
from toys import ToyAdapter


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
        out = decoder(torch.randn(2, ToyAdapter.HIDDEN))
        assert out.shape == (2, ToyAdapter.N_CLASSES)

    def test_post_norm_defaults_to_identity(self, toy_adapter):
        # logit_lens applies post_norm before decoding; the default must be a no-op.
        x = torch.randn(2, ToyAdapter.HIDDEN)
        torch.testing.assert_close(toy_adapter.post_norm(x), x)

    def test_incomplete_subclass_cannot_instantiate(self):
        # the mold's teeth: a subclass missing an abstractmethod can't be built,
        # so no adapter can silently skip something a downstream module needs.
        class Incomplete(ModelAdapter):
            pass

        with pytest.raises(TypeError):
            Incomplete()
