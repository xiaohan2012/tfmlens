"""Block 7 — finetune_decoders.

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


def _toy_prior(steps, batch=4, seq=6, eval_pos=4):
    """Yield (X, y, eval_pos) macro-batches shaped for the toy (already hidden-space)."""
    for _ in range(steps):
        X = torch.randn(batch, seq, ToyAdapter3D.HIDDEN)
        y = torch.randint(0, ToyAdapter3D.N_CLASSES, (batch, seq)).float()
        yield X, y, eval_pos


def _toy_config(out_dir, **kw):
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


class TestFinetuneDecoders:
    def test_produces_and_saves_a_decoder_per_depth(self, tmp_path):
        adapter = ToyAdapter3D()
        cfg = _toy_config(tmp_path)
        decoders = finetune_decoders(adapter, cfg, prior=_toy_prior(cfg.max_steps))
        assert len(decoders) == adapter.n_layers + 1 == 4
        for i in range(adapter.n_layers + 1):
            assert (tmp_path / f"decoder_layer_{i}.pth").exists()
        assert (tmp_path / "loss_per_step.json").exists()
        assert (tmp_path / "config.json").exists()

    def test_seed_makes_it_reproducible(self, tmp_path):
        adapter = ToyAdapter3D()  # shared frozen backbone
        batches = list(_toy_prior(3))  # identical data for both runs

        def run(sub):
            return finetune_decoders(
                adapter, _toy_config(tmp_path / sub, seed=0), prior=iter(batches)
            )

        for a, b in zip(run("a"), run("b"), strict=True):
            for pa, pb in zip(a.parameters(), b.parameters(), strict=True):
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
