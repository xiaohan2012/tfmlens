"""CLI entrypoint: ``python -m tfm_lens.finetune --config <yaml> --model <name>``.

Wires a YAML TrainConfig and a chosen backbone into the Exp4 finetune loop. The
backbone is picked by ``--model`` (a small adapter factory) so the same loop
trains decoders for either LimiX or TabICL; the checkpoint path is a CLI arg (not
in the config) so one config can drive different weight files. Device comes from
the config.

Adapters are imported lazily inside the factory: TabICL lives in the optional
``tabicl`` dependency group, so a LimiX-only install must not import it at module
load just to run ``--model limix_2m``.
"""

import argparse

from tfm_lens.finetune.config import TrainConfig
from tfm_lens.finetune.finetune_decoders import finetune_decoders


def build_adapter(model: str, ckpt: str | None, device: str):
    """Load the requested backbone as a ModelAdapter on ``device``.

    Both adapters' from_checkpoint take ``ckpt=None`` to download their default
    checkpoint from HF, so the two branches are symmetric — no per-model special
    casing.
    """
    if model == "limix_2m":
        from tfm_lens.adapters.limix import LimixAdapter

        return LimixAdapter.from_checkpoint(ckpt, device=device)
    if model == "tabicl_v2":
        from tfm_lens.adapters.tabicl import TabICLAdapter

        return TabICLAdapter.from_checkpoint(ckpt, device=device)
    raise SystemExit(f"unknown --model: {model}")


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m tfm_lens.finetune")
    parser.add_argument("--config", required=True, help="path to a TrainConfig YAML")
    parser.add_argument("--model", choices=["limix_2m", "tabicl_v2"], default="limix_2m")
    parser.add_argument(
        "--ckpt",
        default=None,
        help="checkpoint path; omit to download the model's default checkpoint from HF",
    )
    args = parser.parse_args()

    config = TrainConfig.from_yaml(args.config)
    adapter = build_adapter(args.model, args.ckpt, config.device)
    finetune_decoders(adapter, config)


if __name__ == "__main__":
    main()
