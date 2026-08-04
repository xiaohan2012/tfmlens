"""Path-patching DE–TE sweep over the TabArena binary tasks (issue #44, Part 1).

For each dataset: load -> (optional) subsample -> preprocess -> ``direct_total_effect``
(per-layer DE and TE on the **native head**, same ruler) + ``native_final_auc``
(the per-dataset normalizer). Dumped to JSON for ``plot_de_te_scatter.py`` (the
Hydra Fig-2c analog).

DE = path-patching direct effect (取法 B, one clean forward + residual arithmetic);
TE = total effect (ablate-and-react). Default ablation is **resample** (on-manifold);
donors are the other loaded tasks (leave-one-out). Unlike the tuned-decoder sweep,
this reads only the model's own head — no ``weights/`` decoders needed.

    uv run --group eval python scripts/run_direct_effect_sweep.py --model limix_2m
    uv run --group eval python scripts/run_direct_effect_sweep.py --model mitra \
        --device cuda --out out/de_mitra.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from tfm_lens.evaluation.datasets import TABARENA_BINARY_TASK_IDS, load_tabarena_task
from tfm_lens.evaluation.direct_effect import direct_total_effect
from tfm_lens.evaluation.preprocess import (
    limix_preprocess,
    mitra_preprocess,
    tabfm_preprocess,
    tabicl_preprocess,
)
from tfm_lens.evaluation.self_repair import native_final_auc
from tfm_lens.finetune.__main__ import build_adapter

SEED = 0
MODELS = {
    "limix_2m": limix_preprocess,
    "tabicl_v2": tabicl_preprocess,
    "mitra": mitra_preprocess,
    "tabfm": tabfm_preprocess,
}


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", choices=list(MODELS), default="limix_2m")
    p.add_argument("--subsample-train", type=int, default=500, help="max train rows; 0 = all")
    p.add_argument("--subsample-test", type=int, default=200, help="max test rows; 0 = all")
    p.add_argument("--out", type=Path, default=Path("out/direct_effect.json"))
    p.add_argument("--device", default="cpu", help="cpu or cuda (Mitra's full tables need cuda)")
    p.add_argument("--tasks", default=None, help="comma-separated task ids; default all 15")
    p.add_argument(
        "--ablation", choices=["zero", "resample"], default="resample", help="ablation mode"
    )
    p.add_argument(
        "--n-donors", type=int, default=8, help="resample: donors drawn per target (LOO)"
    )
    return p.parse_args()


def _subsample(X, y, n):
    if n <= 0 or n >= len(X):
        return X, y
    idx = np.random.RandomState(SEED).choice(len(X), n, replace=False)
    return X[idx], y[idx]


def _load_record(task_id, preprocess, args):
    X_train, y_train, X_test, y_test, cat_idx = load_tabarena_task(task_id)
    X_train, y_train = _subsample(X_train, y_train, args.subsample_train)
    X_test, y_test = _subsample(X_test, y_test, args.subsample_test)
    classes = np.unique(y_train)
    y_train = np.searchsorted(classes, y_train)
    y_test = np.searchsorted(classes, y_test)
    X_train_p, X_test_p = preprocess(X_train, y_train, X_test, cat_idx)
    return {
        "Xtr": torch.tensor(X_train_p),
        "ytr": torch.tensor(y_train).float(),
        "Xte": torch.tensor(X_test_p),
        "y_test": y_test,
        "n_classes": len(classes),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
    }


def main():
    args = _parse_args()
    preprocess = MODELS[args.model]
    adapter = build_adapter(args.model, None, args.device)  # ckpt=None -> download / cache

    task_ids = [int(t) for t in args.tasks.split(",")] if args.tasks else TABARENA_BINARY_TASK_IDS
    n_tasks = len(task_ids)

    # Load every task up front — resample needs the others as leave-one-out donors.
    records = {}
    for task_id in task_ids:
        try:
            records[task_id] = _load_record(task_id, preprocess, args)
        except Exception as exc:  # keep going so one bad table can't sink the run
            print(f"load task {task_id}: FAILED {type(exc).__name__}: {exc}", flush=True)

    results = {}
    for i, (task_id, rec) in enumerate(records.items()):
        try:
            kw = {}
            if args.ablation == "resample":
                kw = dict(
                    ablation="resample",
                    donor_tables=[
                        (r["Xtr"], r["ytr"], r["Xte"])
                        for tid, r in records.items()
                        if tid != task_id
                    ],
                    n_donors=args.n_donors,
                    seed=SEED,
                )
            effects = direct_total_effect(
                adapter, rec["Xtr"], rec["ytr"], rec["Xte"], rec["y_test"], rec["n_classes"], **kw
            )
            native_final = native_final_auc(
                adapter, rec["Xtr"], rec["ytr"], rec["Xte"], rec["y_test"], rec["n_classes"]
            )
            results[str(task_id)] = {
                "effects": effects,
                "native_final": native_final,
                "n_train": rec["n_train"],
                "n_test": rec["n_test"],
            }
            print(
                f"[{i + 1}/{len(records)}] task {task_id} ({args.ablation}): "
                f"native AUC {native_final:.3f} | rows {rec['n_train']}+{rec['n_test']}",
                flush=True,
            )
        except Exception as exc:  # keep going so one bad table can't sink the run
            print(
                f"[{i + 1}/{len(records)}] task {task_id}: FAILED {type(exc).__name__}: {exc}",
                flush=True,
            )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2))
    print(f"wrote {args.out} ({len(results)}/{n_tasks} tasks)")


if __name__ == "__main__":
    main()
