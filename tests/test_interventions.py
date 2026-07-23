"""Block 3 — skip_layer.

Temporarily makes a layer the identity, and must restore the original forward on
exit — including when the body raises. Verified through capture.
"""

import pytest
import torch

from tfm_lens.core.capture import capture_layers
from tfm_lens.core.interventions import skip_layer


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
