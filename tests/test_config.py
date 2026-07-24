"""Block 7 — TrainConfig (pydantic): defaults, validation, YAML override, snapshot."""

import json

import pytest
import torch
from pydantic import ValidationError

from tfm_lens.finetune.config import PriorConfig, TrainConfig


class TestTrainConfig:
    def test_defaults_match_paper_recipe(self, tmp_path):
        cfg = TrainConfig(out_dir=tmp_path)
        assert (cfg.lr, cfg.weight_decay, cfg.grad_clip) == (3e-5, 1e-4, 1.0)
        assert cfg.max_steps == 200
        assert cfg.prior.prior_type == "mix_scm"
        assert cfg.prior.mix_probs == (0.7, 0.3)
        assert (cfg.prior.min_features, cfg.prior.max_features) == (2, 30)

    def test_rejects_bad_prior_type(self, tmp_path):
        # the graph_scm footgun the old CLI shipped
        with pytest.raises(ValidationError):
            TrainConfig(out_dir=tmp_path, prior=PriorConfig(prior_type="graph_scm"))

    def test_mix_probs_must_sum_to_one(self):
        with pytest.raises(ValidationError):
            PriorConfig(mix_probs=(0.5, 0.2))

    def test_device_autodetects(self, tmp_path):
        cfg = TrainConfig(out_dir=tmp_path)
        assert cfg.device == ("cuda" if torch.cuda.is_available() else "cpu")

    def test_yaml_override_keeps_other_defaults(self, tmp_path):
        (tmp_path / "c.yaml").write_text("out_dir: out\nmax_steps: 5\nprior:\n  max_features: 10\n")
        cfg = TrainConfig.from_yaml(tmp_path / "c.yaml")
        assert cfg.max_steps == 5  # overridden
        assert cfg.prior.max_features == 10  # nested override
        assert cfg.lr == 3e-5  # default preserved

    def test_snapshot_roundtrips(self, tmp_path):
        TrainConfig(out_dir=tmp_path, max_steps=7).save_snapshot(tmp_path)
        data = json.loads((tmp_path / "config.json").read_text())
        assert data["max_steps"] == 7
