"""TabICLAdapter integration (needs the TabICLv2 ckpt + the `tabicl` package).

Runs the real frozen TabICL through capture / skip / logit_lens to prove the
abstraction that worked on the toys also holds on a real 3D model — and that the
core's keyword-called-layer handling works on the genuine ICL Encoder. Skipped
when `tabicl` isn't installed or the checkpoint can't be resolved.
"""

import os

import pytest
import torch

from tfm_lens.core.capture import capture_layers
from tfm_lens.core.interventions import skip_layer
from tfm_lens.core.logit_lens import logit_lens
from tfm_lens.evaluation.layerwise import layerwise_auc, predict_layers

BATCH = 4


@pytest.fixture(scope="session")
def tabicl_adapter():
    pytest.importorskip("tabicl")  # skip if the `tabicl` dependency group isn't installed
    from tfm_lens.adapters.tabicl import TabICLAdapter

    path = os.environ.get("TABICL_V2_CKPT")  # a local ckpt, else download from HF
    try:
        return TabICLAdapter.from_checkpoint(path, device="cpu")
    except Exception as exc:  # offline / HF unavailable
        pytest.skip(f"could not build TabICL adapter: {exc}")


@pytest.fixture
def table():
    # synthetic classification table: 8 train rows + 4 test rows, 5 features.
    # Seeded so it doesn't depend on global RNG state left by earlier tests.
    g = torch.Generator().manual_seed(0)
    X = torch.randn(BATCH, 12, 5, generator=g)
    y_train = torch.randint(0, 3, (BATCH, 8), generator=g).float()
    return X, y_train, 8  # X, y_train, eval_pos


class TestTabICLAdapterIntegration:
    def test_from_checkpoint_loads_on_requested_device(self, tabicl_adapter):
        assert all(p.device.type == "cpu" for p in tabicl_adapter.model.parameters())

    def test_layers(self, tabicl_adapter):
        assert tabicl_adapter.n_layers == 12  # 12 ICL blocks -> 13 capture depths

    def test_capture_one_3d_tap_per_depth(self, tabicl_adapter, table):
        X, y, eval_pos = table
        with capture_layers(tabicl_adapter) as cache:
            tabicl_adapter.forward_frozen(X, y, eval_pos)
        assert len(cache) == 13  # input embedding + 12 layer outputs
        assert cache[0].dim() == 3  # [batch, rows, hidden] — 3D residual

    def test_logit_lens_end_to_end(self, tabicl_adapter, table):
        X, y, eval_pos = table
        decoders = [tabicl_adapter.decoder_template() for _ in range(tabicl_adapter.n_layers + 1)]
        with capture_layers(tabicl_adapter) as cache:
            tabicl_adapter.forward_frozen(X, y, eval_pos)
        preds = logit_lens(cache, decoders, tabicl_adapter, eval_pos)
        assert len(preds) == 13
        n_test = X.shape[1] - eval_pos
        assert all(p.shape[0] == BATCH and p.shape[1] == n_test for p in preds)

    def test_skip_layer_is_identity(self, tabicl_adapter, table):
        # exercises the kwarg path on the real ICL Encoder (block(q=x, ...)).
        X, y, eval_pos = table
        with skip_layer(tabicl_adapter, 5), capture_layers(tabicl_adapter) as cache:
            tabicl_adapter.forward_frozen(X, y, eval_pos)
        # output of layer 5 (depth 6) equals its input (depth 5).
        torch.testing.assert_close(cache[6], cache[5])

    def test_native_decoder_decodes_separable_task(self, tabicl_adapter):
        # PR1 gate: the model's own decoder at the final layer decodes a separable
        # binary table above chance — proves the whole mapping is wired right end to
        # end (forward -> capture -> post_norm -> decoder), no fine-tuned weights.
        torch.manual_seed(0)
        f = 5
        X_train, X_test = torch.randn(60, f), torch.randn(40, f)
        w = torch.tensor([1.5, -1.0, 0.5, 0.0, 0.0])
        y_train = (X_train @ w > 0).float()
        y_test = (X_test @ w > 0).long().numpy()

        native = tabicl_adapter.decoder_template()
        decoders = [native] * (tabicl_adapter.n_layers + 1)
        aucs = layerwise_auc(
            predict_layers(tabicl_adapter, decoders, X_train, y_train, X_test, n_classes=2), y_test
        )
        assert len(aucs) == 13
        assert aucs[-1] > 0.5  # the native final layer decodes the separable task above chance
