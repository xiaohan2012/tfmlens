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
    """Load the requested backbone as a ModelAdapter on ``device``."""
    if model == "limix_2m":
        from tfm_lens.adapters.limix import LimixAdapter

        if ckpt is None:
            raise SystemExit("--ckpt is required for --model limix_2m")
        return LimixAdapter.from_checkpoint(ckpt, device=device)
    if model == "tabicl_v2":
        from tfm_lens.adapters.tabicl import TabICLAdapter

        return TabICLAdapter.from_checkpoint(ckpt, device=device)  # ckpt=None -> HF download
    raise SystemExit(f"unknown --model: {model}")


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m tfm_lens.finetune")
    parser.add_argument("--config", required=True, help="path to a TrainConfig YAML")
    parser.add_argument("--model", choices=["limix_2m", "tabicl_v2"], default="limix_2m")
    parser.add_argument(
        "--ckpt",
        default=None,
        help="checkpoint path; required for limix_2m, optional for tabicl_v2 "
        "(downloads the v2 checkpoint from HF if omitted)",
    )
    args = parser.parse_args()

    config = TrainConfig.from_yaml(args.config)
    adapter = build_adapter(args.model, args.ckpt, config.device)
    finetune_decoders(adapter, config)


if __name__ == "__main__":
    main()
