"""Run the exp6 self-repair sweep over the TabArena binary tasks.

For each dataset: load -> (optional) subsample -> preprocess -> ablation_sweep
(the per-depth AUC trajectories) + native_final_auc (the per-dataset normalizer)
[+ ablation_diffs, the scatter data]. Results are dumped to JSON for the plotting
script. ``--model`` picks the backbone (limix_2m / tabicl_v2): its adapter, its
per-depth decoders (``weights/<model>/``), and its preprocessing.

Defaults subsample to a fast CPU pass. For the full run pass 0 to both subsample
flags; ``--skip-diffs`` drops the (expensive) scatter data so a full pass only
pays for the Figure-8 trajectories.

``--ablation resample`` (#35) swaps zero/skip for **cross-table resample**: each
target's donors are the other loaded tasks (leave-one-out), ``--n-donors`` drawn
without replacement, metric averaged over donors. On-distribution + norm-preserving;
run it alongside the default ``zero`` for the skip-vs-resample cross-check.

    uv run --group eval python scripts/run_self_repair_sweep.py --model limix_2m
    uv run --group tabicl --group eval python scripts/run_self_repair_sweep.py \
        --model tabicl_v2 --subsample-train 0 --subsample-test 0 --skip-diffs \
        --out out/self_repair_tabicl.json
    uv run --group eval python scripts/run_self_repair_sweep.py --model limix_2m \
        --ablation resample --out out/self_repair_limix_resample.json
"""

import argparse
import json
from pathlib import Path

from tfm_lens.evaluation.datasets import TABARENA_BINARY_TASK_IDS, load_task_record
from tfm_lens.evaluation.layerwise import load_decoders
from tfm_lens.evaluation.preprocess import (
    limix_preprocess,
    mitra_preprocess,
    tabfm_preprocess,
    tabicl_preprocess,
)
from tfm_lens.evaluation.self_repair import ablation_diffs, ablation_sweep, native_final_auc
from tfm_lens.finetune.__main__ import build_adapter

SEED = 0
MODELS = {
    "limix_2m": {"weights": Path("weights/limix_2m"), "preprocess": limix_preprocess},
    "tabicl_v2": {"weights": Path("weights/tabicl_v2"), "preprocess": tabicl_preprocess},
    "mitra": {"weights": Path("weights/mitra"), "preprocess": mitra_preprocess},
    "tabfm": {"weights": Path("weights/tabfm"), "preprocess": tabfm_preprocess},
}


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", choices=list(MODELS), default="limix_2m")
    p.add_argument("--subsample-train", type=int, default=500, help="max train rows; 0 = all")
    p.add_argument("--subsample-test", type=int, default=200, help="max test rows; 0 = all")
    p.add_argument("--out", type=Path, default=Path("out/self_repair.json"))
    p.add_argument("--skip-diffs", action="store_true", help="skip ablation_diffs (scatter data)")
    p.add_argument("--device", default="cpu", help="cpu or cuda (Mitra's full tables need cuda)")
    p.add_argument("--tasks", default=None, help="comma-separated task ids; default all 15")
    p.add_argument(
        "--ablation", choices=["zero", "resample"], default="zero", help="ablation mode (#35)"
    )
    p.add_argument(
        "--donor",
        choices=["table", "row", "feature"],
        default="table",
        help="resample donor source",
    )
    p.add_argument(
        "--n-donors", type=int, default=8, help="resample: donors drawn per target (LOO)"
    )
    return p.parse_args()


def main():
    args = _parse_args()
    if args.ablation == "resample" and args.donor != "table":
        raise NotImplementedError(
            f"--donor {args.donor} not implemented; only 'table' (cross-table)"
        )
    cfg = MODELS[args.model]
    adapter = build_adapter(args.model, None, args.device)  # ckpt=None -> download / cache
    decoders = [d.to(args.device) for d in load_decoders(cfg["weights"], adapter)]
    preprocess = cfg["preprocess"]

    task_ids = [int(t) for t in args.tasks.split(",")] if args.tasks else TABARENA_BINARY_TASK_IDS
    n_tasks = len(task_ids)

    # Load every task up front — resample needs the other tables as leave-one-out donors.
    records = {}
    for task_id in task_ids:
        try:
            records[task_id] = load_task_record(
                task_id, preprocess, args.subsample_train, args.subsample_test, SEED
            )
        except Exception as exc:  # keep going so one bad table can't sink the run
            print(f"load task {task_id}: FAILED {type(exc).__name__}: {exc}", flush=True)

    results = {}
    for i, (task_id, rec) in enumerate(records.items()):
        try:
            sweep_args = (
                adapter,
                decoders,
                rec["Xtr"],
                rec["ytr"],
                rec["Xte"],
                rec["y_test"],
                rec["n_classes"],
            )
            sweep_kw = {}
            if args.ablation == "resample":
                donor_tables = [
                    (r["Xtr"], r["ytr"], r["Xte"]) for tid, r in records.items() if tid != task_id
                ]
                sweep_kw = dict(
                    ablation="resample",
                    donor_tables=donor_tables,
                    n_donors=args.n_donors,
                    seed=SEED,
                )

            sweep = ablation_sweep(*sweep_args, **sweep_kw)
            native_final = native_final_auc(
                adapter, rec["Xtr"], rec["ytr"], rec["Xte"], rec["y_test"], rec["n_classes"]
            )
            entry = {
                "sweep": sweep,
                "native_final": native_final,
                "n_train": rec["n_train"],
                "n_test": rec["n_test"],
            }
            # diffs is a zero-ablation scatter concept; only meaningful in zero mode.
            if not args.skip_diffs and args.ablation == "zero":
                entry["diffs"] = ablation_diffs(*sweep_args)
            results[str(task_id)] = entry
            final_auc = sweep["baseline"]["auc"][-1]
            print(
                f"[{i + 1}/{len(records)}] task {task_id} ({args.ablation}): "
                f"final AUC {final_auc:.3f} | native {native_final:.3f} | "
                f"rows {rec['n_train']}+{rec['n_test']}",
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
