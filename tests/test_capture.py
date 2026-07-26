"""capture_layers.

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
        # depth 0 is the layer's packed input (x,); depths 1+ are the layer outputs.
        assert cache[0][0].shape == (2, 5, ToyAdapter3D.HIDDEN)
        assert all(t.shape == (2, 5, ToyAdapter3D.HIDDEN) for t in cache[1:])

    def test_empty_before_forward_filled_after(self, toy_adapter, toy_input):
        with capture_layers(toy_adapter) as cache:
            assert cache == []  # hooks are armed but nothing has run yet
            toy_adapter.forward_frozen(toy_input, None, 5)
            assert len(cache) == 4

    def test_depth_indexing(self, toy_adapter, toy_input):
        blocks = toy_adapter.layers
        with capture_layers(toy_adapter) as cache:
            toy_adapter.forward_frozen(toy_input, None, 5)
        # depth 0 = layer 0's packed input ((x,)); depth 1 = layer 0's output.
        torch.testing.assert_close(cache[0][0], toy_input)
        torch.testing.assert_close(cache[1], blocks[0](toy_input))

    def test_captures_keyword_called_layers(self, toy_adapter_keyword_call, toy_input):
        # TabICL drives blocks by keyword (blk(q=x)); the input pre-hook must still
        # grab depth 0 (the first layer's input) with no positional args present.
        adapter = toy_adapter_keyword_call
        blocks = adapter.layers
        with capture_layers(adapter) as cache:
            adapter.forward_frozen(toy_input, None, 5)
        assert len(cache) == adapter.n_layers + 1 == 4
        torch.testing.assert_close(cache[0][0], toy_input)  # depth 0 = raw input (packed)
        torch.testing.assert_close(cache[1], blocks[0](q=toy_input))  # depth 1 = layer 0 output

    def test_hooks_removed_on_exit(self, toy_adapter, toy_input):
        with capture_layers(toy_adapter) as first:
            toy_adapter.forward_frozen(toy_input, None, 5)
        # a second capture must see exactly n+1 again — old hooks must not linger.
        with capture_layers(toy_adapter) as second:
            toy_adapter.forward_frozen(toy_input, None, 5)
        assert len(first) == len(second) == 4
