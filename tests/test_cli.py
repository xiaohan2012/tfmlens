"""The `python -m tfm_lens.finetune` entrypoint — wiring only.

Patches out the heavy pieces (checkpoint load, training loop) and asserts main()
routes --config / --ckpt / device into from_checkpoint and finetune_decoders.
"""

import sys

from tfm_lens.finetune import __main__ as cli


def test_cli_wires_config_ckpt_and_device(monkeypatch, tmp_path):
    (tmp_path / "c.yaml").write_text("out_dir: out\ndevice: cpu\n")
    seen: dict = {}

    def fake_from_checkpoint(path, device):
        seen["ckpt"], seen["device"] = path, device
        return "ADAPTER"

    def fake_finetune(adapter, config):
        seen["adapter"], seen["out_dir"] = adapter, str(config.out_dir)

    monkeypatch.setattr(cli.LimixAdapter, "from_checkpoint", fake_from_checkpoint)
    monkeypatch.setattr(cli, "finetune_decoders", fake_finetune)
    monkeypatch.setattr(
        sys, "argv", ["prog", "--config", str(tmp_path / "c.yaml"), "--ckpt", "ck.pth"]
    )

    cli.main()

    assert seen["ckpt"] == "ck.pth"
    assert seen["device"] == "cpu"  # from the yaml, threaded into from_checkpoint
    assert seen["adapter"] == "ADAPTER"
    assert seen["out_dir"] == "out"
