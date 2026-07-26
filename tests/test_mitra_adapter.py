"""MitraAdapter integration (needs the Mitra checkpoint + the `mitra` group).

Runs the real frozen Tab2D (double-stream) through capture / skip / logit_lens:
each layer returns ``(support, query)``, and readout takes the query stream.
Skipped without the `mitra` group installed or a resolvable checkpoint.
"""

import os

import pytest
import torch

from tfm_lens.core.capture import capture_layers
from tfm_lens.core.interventions import skip_layer
from tfm_lens.core.logit_lens import logit_lens
from tfm_lens.evaluation.layerwise import layerwise_auc, predict_layers

BATCH = 1  # Mitra's forward_frozen splits a packed [1, seq, f] table into support/query


@pytest.fixture(scope="session")
def mitra_adapter():
    pytest.importorskip("einx")  # part of the `mitra` dependency group
    from tfm_lens.adapters.mitra import MitraAdapter

    path = os.environ.get("MITRA_CKPT")  # a local ckpt dir, else download from HF
    try:
        return MitraAdapter.from_checkpoint(path, device="cpu")
    except Exception as exc:  # offline / HF unavailable
        pytest.skip(f"could not build Mitra adapter: {exc}")


@pytest.fixture
def table():
    # 8 train rows + 4 test rows, 5 features; seeded (independent of global RNG).
    g = torch.Generator().manual_seed(0)
    X = torch.randn(BATCH, 12, 5, generator=g)
    y_train = torch.randint(0, 2, (BATCH, 8), generator=g).float()
    return X, y_train, 8  # X, y_train, eval_pos


class TestMitraAdapterIntegration:
    def test_from_checkpoint_loads_on_requested_device(self, mitra_adapter):
        assert all(p.device.type == "cpu" for p in mitra_adapter.model.parameters())

    def test_layers(self, mitra_adapter):
        assert mitra_adapter.n_layers == 12  # 12 Tab2D layers -> 13 capture depths

    def test_capture_double_stream_tap_per_depth(self, mitra_adapter, table):
        X, y, eval_pos = table
        with capture_layers(mitra_adapter) as cache:
            mitra_adapter.forward_frozen(X, y, eval_pos)
        assert len(cache) == 13
        # each layer output is (support, query); query is the 4D stream readout takes.
        assert isinstance(cache[1], tuple) and cache[1][1].dim() == 4

    def test_logit_lens_end_to_end(self, mitra_adapter, table):
        X, y, eval_pos = table
        decoders = [mitra_adapter.decoder_template() for _ in range(mitra_adapter.n_layers + 1)]
        with capture_layers(mitra_adapter) as cache:
            mitra_adapter.forward_frozen(X, y, eval_pos)
        preds = logit_lens(cache, decoders, mitra_adapter, eval_pos)
        assert len(preds) == 13
        n_test = X.shape[1] - eval_pos
        assert all(p.shape[0] == BATCH and p.shape[1] == n_test for p in preds)

    def test_skip_layer_is_identity(self, mitra_adapter, table):
        # skip must return both streams unchanged; the skipped layer's query output
        # (depth 6) equals its query input (depth 5).
        X, y, eval_pos = table
        with skip_layer(mitra_adapter, 5), capture_layers(mitra_adapter) as cache:
            mitra_adapter.forward_frozen(X, y, eval_pos)
        torch.testing.assert_close(cache[6][1], cache[5][1])

    def test_native_decoder_decodes_separable_task(self, mitra_adapter):
        torch.manual_seed(0)
        f = 5
        X_train, X_test = torch.randn(60, f), torch.randn(40, f)
        w = torch.tensor([1.5, -1.0, 0.5, 0.0, 0.0])
        y_train = (X_train @ w > 0).float()
        y_test = (X_test @ w > 0).long().numpy()

        native = mitra_adapter.decoder_template()
        decoders = [native] * (mitra_adapter.n_layers + 1)
        aucs = layerwise_auc(
            predict_layers(mitra_adapter, decoders, X_train, y_train, X_test, n_classes=2), y_test
        )
        assert len(aucs) == 13
        assert aucs[-1] > 0.5  # native final layer decodes the separable task above chance
