"""finetune_decoders.

Loop mechanics are tested on the toy adapter with injected data (fast, always
runs). The real path (LimiX + mix_scm prior) runs end-to-end on Linux CI and is
skipped on macOS (tree_scm/xgboost segfaults there).
"""

import sys

import pytest
import torch

from tfm_lens.finetune.config import PriorConfig, TrainConfig
from tfm_lens.finetune.finetune_decoders import finetune_decoders
from toys import ToyAdapter3D


@pytest.fixture
def toy_prior():
    """Factory for (X, y, eval_pos) macro-batches shaped for the toy (hidden-space)."""

    def _make(steps, batch=4, seq=6, eval_pos=4):
        for _ in range(steps):
            X = torch.randn(batch, seq, ToyAdapter3D.HIDDEN)
            y = torch.randint(0, ToyAdapter3D.N_CLASSES, (batch, seq)).float()
            yield X, y, eval_pos

    return _make


@pytest.fixture
def toy_config():
    """Factory for a fast CPU TrainConfig with toy-sized batches."""

    def _make(out_dir, **kw):
        kw = {
            "max_steps": 3,
            "prior_batch_size": 4,
            "micro_batch_size": 4,
            "training_batch_size": 4,
            "save_every": 1,
            "device": "cpu",
            **kw,
        }
        return TrainConfig(out_dir=out_dir, **kw)

    return _make


class TestFinetuneDecoders:
    def test_produces_and_saves_a_decoder_per_depth(self, tmp_path, toy_config, toy_prior):
        adapter = ToyAdapter3D()
        cfg = toy_config(tmp_path)
        decoders = finetune_decoders(adapter, cfg, prior=toy_prior(cfg.max_steps))
        assert len(decoders) == adapter.n_layers + 1 == 4
        for i in range(adapter.n_layers + 1):
            assert (tmp_path / f"decoder_layer_{i}.pth").exists()
        assert (tmp_path / "loss_per_step.json").exists()
        assert (tmp_path / "config.json").exists()

    def test_seed_makes_it_reproducible(self, tmp_path, toy_config, toy_prior):
        adapter = ToyAdapter3D()  # shared frozen backbone
        batches = list(toy_prior(3))  # identical data for both runs

        def run(sub):
            return finetune_decoders(
                adapter, toy_config(tmp_path / sub, seed=0), prior=iter(batches)
            )

        a = run("a")
        b = run("b")
        for da, db in zip(a, b, strict=True):
            for pa, pb in zip(da.parameters(), db.parameters(), strict=True):
                torch.testing.assert_close(pa, pb)

    @pytest.mark.skipif(
        sys.platform == "darwin", reason="tree_scm/xgboost segfaults on macOS arm64"
    )
    def test_real_prior_end_to_end(self, tmp_path, limix_model):
        from tfm_lens.adapters.limix import LimixAdapter

        adapter = LimixAdapter(limix_model)
        cfg = TrainConfig(
            out_dir=tmp_path,
            max_steps=2,
            prior_batch_size=8,
            micro_batch_size=4,
            training_batch_size=4,
            device="cpu",
            prior=PriorConfig(max_features=10, max_seq_len=128, n_jobs=1),
        )
        decoders = finetune_decoders(adapter, cfg)  # real mix_scm prior
        assert len(decoders) == adapter.n_layers + 1 == 13
        assert (tmp_path / "decoder_layer_12.pth").exists()
