"""TabFMAdapter integration (needs the TabFM checkpoint + the `tabfm` group).

Runs the real frozen TabFM (single-stream 3D, like TabICL) through capture /
skip / logit_lens: self-repair reads the ICL stage (``icl_predictor.tf_icl``),
readout applies its RMSNorm then slices the test rows. Skipped without the
`tabfm` group installed or a resolvable checkpoint.
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
def tabfm_adapter():
    pytest.importorskip("safetensors")  # part of the `tabfm` dependency group
    from tfm_lens.adapters.tabfm import TabFMAdapter

    path = os.environ.get("TABFM_CKPT")  # a local ckpt dir, else download from HF
    try:
        return TabFMAdapter.from_checkpoint(path, device="cpu")
    except Exception as exc:  # offline / HF unavailable
        pytest.skip(f"could not build TabFM adapter: {exc}")


@pytest.fixture
def table():
    # synthetic classification table: 8 train rows + 4 test rows, 5 features.
    # Seeded so it doesn't depend on global RNG state left by earlier tests.
    g = torch.Generator().manual_seed(0)
    X = torch.randn(BATCH, 12, 5, generator=g)
    y_train = torch.randint(0, 3, (BATCH, 8), generator=g).float()
    return X, y_train, 8  # X, y_train, eval_pos


class TestTabFMAdapterIntegration:
    def test_from_checkpoint_loads_on_requested_device(self, tabfm_adapter):
        assert all(p.device.type == "cpu" for p in tabfm_adapter.model.parameters())

    def test_layers(self, tabfm_adapter):
        assert tabfm_adapter.n_layers == 24  # 24 ICL blocks -> 25 capture depths

    def test_capture_one_3d_tap_per_depth(self, tabfm_adapter, table):
        X, y, eval_pos = table
        with capture_layers(tabfm_adapter) as cache:
            tabfm_adapter.forward_frozen(X, y, eval_pos)
        assert len(cache) == 25  # input embedding + 24 layer outputs
        assert cache[0][0].dim() == 3  # depth 0 is the packed input; [batch, rows, hidden]

    def test_logit_lens_end_to_end(self, tabfm_adapter, table):
        X, y, eval_pos = table
        decoders = [tabfm_adapter.decoder_template() for _ in range(tabfm_adapter.n_layers + 1)]
        with capture_layers(tabfm_adapter) as cache:
            tabfm_adapter.forward_frozen(X, y, eval_pos)
        preds = logit_lens(cache, decoders, tabfm_adapter, eval_pos)
        assert len(preds) == 25
        n_test = X.shape[1] - eval_pos
        assert all(p.shape[0] == BATCH and p.shape[1] == n_test for p in preds)

    def test_skip_layer_is_identity(self, tabfm_adapter, table):
        # exercises the layer path on the real ICL Encoder blocks.
        X, y, eval_pos = table
        with skip_layer(tabfm_adapter, 5), capture_layers(tabfm_adapter) as cache:
            tabfm_adapter.forward_frozen(X, y, eval_pos)
        # output of layer 5 (depth 6) equals its input (depth 5).
        torch.testing.assert_close(cache[6], cache[5])

    def test_native_decoder_decodes_separable_task(self, tabfm_adapter):
        # PR1 gate: the model's own decoder at the final layer decodes a separable
        # binary table above chance — proves the whole mapping is wired right end to
        # end (forward -> capture -> ln -> decoder), no fine-tuned weights.
        torch.manual_seed(0)
        f = 5
        X_train, X_test = torch.randn(60, f), torch.randn(40, f)
        w = torch.tensor([1.5, -1.0, 0.5, 0.0, 0.0])
        y_train = (X_train @ w > 0).float()
        y_test = (X_test @ w > 0).long().numpy()

        native = tabfm_adapter.decoder_template()
        decoders = [native] * (tabfm_adapter.n_layers + 1)
        aucs = layerwise_auc(
            predict_layers(tabfm_adapter, decoders, X_train, y_train, X_test, n_classes=2), y_test
        )
        assert len(aucs) == 25
        assert aucs[-1] > 0.5  # the native final layer decodes the separable task above chance
