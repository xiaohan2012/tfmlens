"""Block 5 — LimixAdapter integration (needs the LimiX-2M ckpt).

Runs the real frozen LimiX through capture / skip / logit_lens to prove the
abstraction that worked on the toys also holds on a real 4D model. Skipped when
the ckpt can't be resolved (see the limix_ckpt fixture).
"""

import pytest
import torch

from tfm_lens.adapters.limix import LimixAdapter
from tfm_lens.core.capture import capture_layers
from tfm_lens.core.interventions import skip_layer
from tfm_lens.core.logit_lens import logit_lens


@pytest.fixture(scope="session")
def limix_adapter(limix_model):
    return LimixAdapter(limix_model)


def _table():
    # tiny synthetic classification table: 8 train rows + 4 test rows, 5 features.
    X = torch.randn(1, 12, 5)
    y_train = torch.randint(0, 3, (1, 8)).float()
    return X, y_train, 8  # X, y_train, eval_pos


class TestLimixAdapterIntegration:
    def test_layers(self, limix_adapter):
        assert limix_adapter.n_layers == 12

    def test_capture_one_4d_tap_per_depth(self, limix_adapter):
        X, y, eval_pos = _table()
        with capture_layers(limix_adapter) as cache:
            limix_adapter.forward_frozen(X, y, eval_pos)
        assert len(cache) == 13  # input embedding + 12 layer outputs
        assert cache[0].dim() == 4  # [batch, seq, tokens, hidden]

    def test_logit_lens_end_to_end(self, limix_adapter):
        X, y, eval_pos = _table()
        decoders = [limix_adapter.decoder_template() for _ in range(limix_adapter.n_layers + 1)]
        with capture_layers(limix_adapter) as cache:
            limix_adapter.forward_frozen(X, y, eval_pos)
        preds = logit_lens(cache, decoders, limix_adapter, eval_pos)
        assert len(preds) == 13
        n_test = X.shape[1] - eval_pos
        assert all(p.shape[0] == 1 and p.shape[1] == n_test for p in preds)

    def test_skip_layer_is_identity(self, limix_adapter):
        X, y, eval_pos = _table()
        with skip_layer(limix_adapter, 5), capture_layers(limix_adapter) as cache:
            limix_adapter.forward_frozen(X, y, eval_pos)
        # output of layer 5 (depth 6) equals its input (depth 5).
        torch.testing.assert_close(cache[6], cache[5])
