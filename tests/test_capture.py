"""Block 2 — capture_layers.

Captures the residual stream at every depth (n_layers + 1 taps: the input
embedding plus each layer's output), stores it raw, and cleans up its hooks.
"""

import torch

from tfm_lens.core.capture import capture_layers
from toys import ToyAdapter3D


class TestCaptureLayers:
    def test_one_tensor_per_depth(self, toy_adapter, toy_input):
        with capture_layers(toy_adapter) as cache:
            toy_adapter.forward_frozen(toy_input, None, 5)
        assert len(cache) == toy_adapter.n_layers + 1 == 4
        assert all(t.shape == (2, 5, ToyAdapter3D.HIDDEN) for t in cache)

    def test_empty_before_forward_filled_after(self, toy_adapter, toy_input):
        with capture_layers(toy_adapter) as cache:
            assert cache == []  # hooks are armed but nothing has run yet
            toy_adapter.forward_frozen(toy_input, None, 5)
            assert len(cache) == 4

    def test_depth_indexing(self, toy_adapter, toy_input):
        blocks = toy_adapter.layers
        with capture_layers(toy_adapter) as cache:
            toy_adapter.forward_frozen(toy_input, None, 5)
        # depth 0 = input to layer 0 (the raw embedding); depth 1 = layer 0's output.
        torch.testing.assert_close(cache[0], toy_input)
        torch.testing.assert_close(cache[1], blocks[0](toy_input))

    def test_hooks_removed_on_exit(self, toy_adapter, toy_input):
        with capture_layers(toy_adapter) as first:
            toy_adapter.forward_frozen(toy_input, None, 5)
        # a second capture must see exactly n+1 again — old hooks must not linger.
        with capture_layers(toy_adapter) as second:
            toy_adapter.forward_frozen(toy_input, None, 5)
        assert len(first) == len(second) == 4
