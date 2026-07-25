"""Run the exp6 self-repair sweep over the TabArena binary tasks.

For each dataset: load -> (optional) subsample -> preprocess -> ablation_sweep
(the per-depth AUC trajectories) + native_final_auc (the per-dataset normalizer)
[+ ablation_diffs, the scatter data]. Results are dumped to JSON for the plotting
script. ``--model`` picks the backbone (limix_2m / tabicl_v2): its adapter, its
per-depth decoders (``weights/<model>/``), and its preprocessing.

Defaults subsample to a fast CPU pass. For the full run pass 0 to both subsample
flags; ``--skip-diffs`` drops the (expensive) scatter data so a full pass only
pays for the Figure-8 trajectories.

    uv run --group eval python scripts/run_self_repair_sweep.py --model limix_2m
    uv run --group tabicl --group eval python scripts/run_self_repair_sweep.py \
        --model tabicl_v2 --subsample-train 0 --subsample-test 0 --skip-diffs \
        --out out/self_repair_tabicl.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from tfm_lens.evaluation.datasets import TABARENA_BINARY_TASK_IDS, load_tabarena_task
from tfm_lens.evaluation.layerwise import load_decoders
from tfm_lens.evaluation.preprocess import limix_preprocess, tabicl_preprocess
from tfm_lens.evaluation.self_repair import ablation_diffs, ablation_sweep, native_final_auc
from tfm_lens.finetune.__main__ import build_adapter

SEED = 0
MODELS = {
    "limix_2m": {"weights": Path("weights/limix_2m"), "preprocess": limix_preprocess},
    "tabicl_v2": {"weights": Path("weights/tabicl_v2"), "preprocess": tabicl_preprocess},
}


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", choices=list(MODELS), default="limix_2m")
    p.add_argument("--subsample-train", type=int, default=500, help="max train rows; 0 = all")
    p.add_argument("--subsample-test", type=int, default=200, help="max test rows; 0 = all")
    p.add_argument("--out", type=Path, default=Path("out/self_repair.json"))
    p.add_argument("--skip-diffs", action="store_true", help="skip ablation_diffs (scatter data)")
    return p.parse_args()


def _subsample(X, y, n):
    """Take at most n rows (n <= 0 keeps all), seeded for reproducibility."""
    if n <= 0 or n >= len(X):
        return X, y
    idx = np.random.RandomState(SEED).choice(len(X), n, replace=False)
    return X[idx], y[idx]


def main():
    args = _parse_args()
    cfg = MODELS[args.model]
    adapter = build_adapter(args.model, None, "cpu")  # ckpt=None -> download / cache
    decoders = load_decoders(cfg["weights"], adapter)
    preprocess = cfg["preprocess"]

    n_tasks = len(TABARENA_BINARY_TASK_IDS)
    results = {}
    for i, task_id in enumerate(TABARENA_BINARY_TASK_IDS):
        try:
            X_train, y_train, X_test, y_test, cat_idx = load_tabarena_task(task_id)
            X_train, y_train = _subsample(X_train, y_train, args.subsample_train)
            X_test, y_test = _subsample(X_test, y_test, args.subsample_test)
            # 0-index the labels: TabICL's one-hot y_encoder needs contiguous ints;
            # harmless for LimiX since binary labels are already 0/1.
            classes = np.unique(y_train)
            y_train = np.searchsorted(classes, y_train)
            y_test = np.searchsorted(classes, y_test)
            X_train_p, X_test_p = preprocess(X_train, y_train, X_test, cat_idx)

            n_classes = len(classes)
            Xtr_t = torch.tensor(X_train_p)
            ytr_t = torch.tensor(y_train).float()
            Xte_t = torch.tensor(X_test_p)
            sweep_args = (adapter, decoders, Xtr_t, ytr_t, Xte_t, y_test, n_classes)

            sweep = ablation_sweep(*sweep_args)
            native_final = native_final_auc(adapter, Xtr_t, ytr_t, Xte_t, y_test, n_classes)
            entry = {
                "sweep": sweep,
                "native_final": native_final,
                "n_train": int(len(X_train)),
                "n_test": int(len(X_test)),
            }
            if not args.skip_diffs:
                entry["diffs"] = ablation_diffs(*sweep_args)
            results[str(task_id)] = entry
            print(
                f"[{i + 1}/{n_tasks}] task {task_id}: final AUC {sweep['baseline'][-1]:.3f} | "
                f"native {native_final:.3f} | rows {len(X_train)}+{len(X_test)}",
                flush=True,
            )
        except Exception as exc:  # keep going so one bad table can't sink the run
            print(
                f"[{i + 1}/{n_tasks}] task {task_id}: FAILED {type(exc).__name__}: {exc}",
                flush=True,
            )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2))
    print(f"wrote {args.out} ({len(results)}/{n_tasks} tasks)")


if __name__ == "__main__":
    main()
