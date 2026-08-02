"""skip_layer.

Temporarily makes a layer the identity, and must restore the original forward on
exit — including when the body raises. Verified through capture.
"""

import pytest
import torch

from tfm_lens.core.capture import capture_layers
from tfm_lens.core.interventions import inject_delta, skip_layer


class TestSkipLayer:
    def _capture(self, adapter, x):
        with capture_layers(adapter) as cache:
            adapter.forward_frozen(x, None, 5)
        return cache

    def test_skipped_layer_is_identity(self, toy_adapter, toy_input):
        with skip_layer(toy_adapter, 1):
            cache = self._capture(toy_adapter, toy_input)
        # output of layer 1 (depth 2) equals its input (depth 1).
        torch.testing.assert_close(cache[2], cache[1])

    def test_skipped_layer_is_identity_4d(self, toy_adapter_4d, toy_input_4d):
        # 4D layers return a 3-tuple; identity_forward must match that shape.
        with skip_layer(toy_adapter_4d, 1):
            cache = self._capture(toy_adapter_4d, toy_input_4d)
        torch.testing.assert_close(cache[2], cache[1])

    def test_skipped_layer_is_identity_keyword_call(self, toy_adapter_keyword_call, toy_input):
        # keyword-called layers (TabICL: blk(q=x)): skip must still neutralize the layer.
        with skip_layer(toy_adapter_keyword_call, 1):
            cache = self._capture(toy_adapter_keyword_call, toy_input)
        # output of layer 1 (depth 2) equals its input (depth 1).
        torch.testing.assert_close(cache[2], cache[1])

    def test_skipped_layer_is_identity_double_stream(self, toy_adapter_double_stream, toy_input):
        # double-stream layers return (support, query); skip must return both unchanged.
        adapter = toy_adapter_double_stream
        with skip_layer(adapter, 1), capture_layers(adapter) as cache:
            adapter.forward_frozen(toy_input, None, eval_pos=3)  # 3 support rows, 2 query rows
        # layer 1's query output (depth 2) equals its query input (depth 1).
        torch.testing.assert_close(cache[2][1], cache[1][1])

    def test_restores_forward_on_exit(self, toy_adapter, toy_input):
        baseline = self._capture(toy_adapter, toy_input)
        with skip_layer(toy_adapter, 1):
            pass
        after = self._capture(toy_adapter, toy_input)
        for a, b in zip(baseline, after, strict=True):
            torch.testing.assert_close(a, b)

    def test_restores_on_exception(self, toy_adapter, toy_input):
        baseline = self._capture(toy_adapter, toy_input)
        with pytest.raises(RuntimeError), skip_layer(toy_adapter, 1):
            raise RuntimeError("boom")
        after = self._capture(toy_adapter, toy_input)
        for a, b in zip(baseline, after, strict=True):
            torch.testing.assert_close(a, b)


class TestResampleLayer:
    """inject_delta makes a layer output ``input + donor_delta``; δ:=0 == skip."""

    def _capture(self, adapter, x, eval_pos=5):
        with capture_layers(adapter) as cache:
            adapter.forward_frozen(x, None, eval_pos)
        return cache

    def test_zero_delta_is_skip(self, toy_adapter, toy_input):
        # donor_delta == 0 → layer output equals its input (same as skip_layer).
        delta = torch.zeros_like(toy_input)
        with inject_delta(toy_adapter, 1, delta):
            cache = self._capture(toy_adapter, toy_input)
        torch.testing.assert_close(cache[2], cache[1])  # out(layer 1) == in(layer 1)

    def test_output_is_input_plus_delta(self, toy_adapter, toy_input):
        delta = torch.randn_like(toy_input)
        with inject_delta(toy_adapter, 1, delta):
            cache = self._capture(toy_adapter, toy_input)
        torch.testing.assert_close(cache[2], cache[1] + delta)  # residual + donor δ

    def test_4d_keeps_tuple_shape(self, toy_adapter_4d, toy_input_4d):
        delta = torch.randn_like(toy_input_4d)
        with inject_delta(toy_adapter_4d, 1, delta):
            cache = self._capture(toy_adapter_4d, toy_input_4d)
        r_in = toy_adapter_4d.residual_of(cache[1])
        r_out = toy_adapter_4d.residual_of(cache[2])
        torch.testing.assert_close(r_out, r_in + delta)

    def test_double_stream_delta_per_stream(self, toy_adapter_double_stream, toy_input):
        adapter = toy_adapter_double_stream
        # a (support_δ, query_δ) pair; support has 3 rows, query 2 (eval_pos=3).
        delta = (torch.randn(2, 3, adapter.HIDDEN), torch.randn(2, 2, adapter.HIDDEN))
        with inject_delta(adapter, 1, delta), capture_layers(adapter) as cache:
            adapter.forward_frozen(toy_input, None, eval_pos=3)
        torch.testing.assert_close(cache[2][0], cache[1][0] + delta[0])  # support
        torch.testing.assert_close(cache[2][1], cache[1][1] + delta[1])  # query

    def test_restores_forward_on_exit(self, toy_adapter, toy_input):
        baseline = self._capture(toy_adapter, toy_input)
        with inject_delta(toy_adapter, 1, torch.randn_like(toy_input)):
            pass
        after = self._capture(toy_adapter, toy_input)
        for a, b in zip(baseline, after, strict=True):
            torch.testing.assert_close(a, b)
