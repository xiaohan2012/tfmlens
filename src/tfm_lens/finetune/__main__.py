"""CLI entrypoint: ``python -m tfm_lens.finetune --config <yaml> --ckpt <path>``.

Wires a YAML TrainConfig and a LimiX checkpoint into the Exp4 finetune loop. The
checkpoint path is a CLI arg (not in the config) so the same config can drive
different weight files; the device comes from the config.
"""

import argparse

from tfm_lens.adapters.limix import LimixAdapter
from tfm_lens.finetune.config import TrainConfig
from tfm_lens.finetune.finetune_decoders import finetune_decoders


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m tfm_lens.finetune")
    parser.add_argument("--config", required=True, help="path to a TrainConfig YAML")
    parser.add_argument("--ckpt", required=True, help="path to a LimiX checkpoint")
    args = parser.parse_args()

    config = TrainConfig.from_yaml(args.config)
    adapter = LimixAdapter.from_checkpoint(args.ckpt, device=config.device)
    finetune_decoders(adapter, config)


if __name__ == "__main__":
    main()
