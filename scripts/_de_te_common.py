"""Shared helpers for the DE/TE/CE plot scripts (``plot_de_te_scatter``,
``plot_de_te_perrow``, ``plot_ce_de_law``).

One reader for the ``de_<model>.json`` contract emitted by
``direct_total_effect`` — so the model labels, the loader, and the per-task
z-score rule live in one place instead of drifting across three scripts.
"""

import json
from pathlib import Path

import numpy as np

MODEL_LABELS = {
    "limix_2m": "LimiX-2M",
    "tabicl_v2": "TabICLv2",
    "mitra": "Mitra",
    "tabfm": "TabFM",
}


def load_de_json(in_dir: Path, models: list[str]) -> dict:
    """Load ``in_dir/de_<model>.json`` for each model (skip missing). Raise if none found."""
    loaded = {}
    for m in models:
        p = in_dir / f"de_{m}.json"
        if p.exists():
            loaded[m] = json.loads(p.read_text())
        else:
            print(f"skip {m}: {p} not found")
    if not loaded:
        raise SystemExit("no input files found (expected in-dir/de_<model>.json)")
    return loaded


def de_scale(eff: dict, coord: str, *, agg: bool = False) -> float:
    """Per-task z-score unit for a coordinate (floored at 1e-6).

    - ``gt_logit`` → σ of the clean final-layer true-class logit.
    - ``margin`` → ``|clean margin|`` when ``agg`` (matches the D1 aggregate scatter),
      else the spread of the clean per-row margin (per-row figures).
    """
    if coord == "gt_logit":
        return max(eff["zscore"]["sigma"], 1e-6)
    if agg:
        return max(abs(eff["clean"]["margin"]), 1e-6)
    return max(float(np.std(eff["clean_rows"]["margin"])), 1e-6)
