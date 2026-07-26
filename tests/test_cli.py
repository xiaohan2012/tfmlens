"""The `python -m tfm_lens.finetune` entrypoint — wiring only.

Patches out the heavy pieces (checkpoint load, training loop) and asserts main()
routes --config / --model / --ckpt / device into the adapter factory and
finetune_decoders. The factory imports adapters lazily from their real modules,
so the patches target those module paths.
"""

import sys

import pytest

from tfm_lens.finetune import __main__ as cli
from tfm_lens.finetune.__main__ import build_adapter


class TestFinetuneCli:
    def test_main_wires_config_model_ckpt_and_device(self, monkeypatch, tmp_path):
        (tmp_path / "c.yaml").write_text("out_dir: out\ndevice: cpu\n")
        seen: dict = {}

        def fake_from_checkpoint(path, device):
            seen["ckpt"], seen["device"] = path, device
            return "ADAPTER"

        def fake_finetune(adapter, config):
            seen["adapter"], seen["out_dir"] = adapter, str(config.out_dir)

        monkeypatch.setattr(
            "tfm_lens.adapters.limix.LimixAdapter.from_checkpoint", fake_from_checkpoint
        )
        monkeypatch.setattr(cli, "finetune_decoders", fake_finetune)
        argv = [
            "prog",
            "--config",
            str(tmp_path / "c.yaml"),
            "--model",
            "limix_2m",
            "--ckpt",
            "ck.pth",
        ]
        monkeypatch.setattr(sys, "argv", argv)

        cli.main()

        assert seen["ckpt"] == "ck.pth"
        assert seen["device"] == "cpu"  # from the yaml, threaded into from_checkpoint
        assert seen["adapter"] == "ADAPTER"
        assert seen["out_dir"] == "out"

    def test_build_adapter_routes_to_tabicl_passing_none_ckpt(self, monkeypatch):
        pytest.importorskip("tabicl")
        import tfm_lens.adapters.tabicl as tabicl_mod

        monkeypatch.setattr(
            tabicl_mod.TabICLAdapter,
            "from_checkpoint",
            lambda path, device: ("TABICL", path, device),
        )
        # ckpt omitted -> None flows through so the adapter downloads from HF.
        assert build_adapter("tabicl_v2", None, "cpu") == ("TABICL", None, "cpu")

    def test_build_adapter_routes_to_mitra_passing_none_ckpt(self, monkeypatch):
        pytest.importorskip("einx")  # the mitra group
        import tfm_lens.adapters.mitra as mitra_mod

        monkeypatch.setattr(
            mitra_mod.MitraAdapter,
            "from_checkpoint",
            lambda path, device: ("MITRA", path, device),
        )
        assert build_adapter("mitra", None, "cpu") == ("MITRA", None, "cpu")

    def test_build_adapter_routes_to_tabfm_passing_none_ckpt(self, monkeypatch):
        pytest.importorskip("safetensors")  # the tabfm group
        import tfm_lens.adapters.tabfm as tabfm_mod

        monkeypatch.setattr(
            tabfm_mod.TabFMAdapter,
            "from_checkpoint",
            lambda path, device: ("TABFM", path, device),
        )
        assert build_adapter("tabfm", None, "cpu") == ("TABFM", None, "cpu")

    def test_build_adapter_rejects_unknown_model(self):
        with pytest.raises(SystemExit):
            build_adapter("bogus", None, "cpu")
