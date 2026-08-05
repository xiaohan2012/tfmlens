"""Hydra Fig-b analog: for the **best-repair layer** of each model, scatter DE vs CE
**across the 15 tasks** + OLS fit (Hydra's Layer-23 plot: r²=0.92, slope=0.69).

Per model: pick the layer (excluding the last, which is degenerate — no downstream so
CE≡0) with the highest mean CE = the strongest repair candidate. Each point = one task.

Caveat: our CE = DE − TE (mechanical DE term); Hydra measures the downstream response
independently. A clean replication needs the per-layer CE decomposition (Part 2).

    uv run --group viz python scripts/plot_hydra_layer.py --coord margin
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_MODEL_LABELS = {
    "limix_2m": "LimiX-2M",
    "tabicl_v2": "TabICLv2",
    "mitra": "Mitra",
    "tabfm": "TabFM",
}


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models", nargs="+", default=["limix_2m", "mitra", "tabicl_v2", "tabfm"])
    p.add_argument("--in-dir", type=Path, default=Path("out"))
    p.add_argument("--out", type=Path, default=Path("out/hydra_layer.png"))
    p.add_argument("--coord", choices=["gt_logit", "margin"], default="margin")
    return p.parse_args()


def _layer_points(results, layer, coord):
    de, ce = [], []
    for r in results.values():
        e = r["effects"]
        sc = max(e["zscore"]["sigma"] if coord == "gt_logit" else abs(e["clean"]["margin"]), 1e-6)
        x = e["de"][str(layer)][coord] / sc
        de.append(x)
        ce.append(x - e["te"][str(layer)][coord] / sc)
    return np.array(de), np.array(ce)


def _best_repair_layer(results, coord):
    """The non-final layer with the highest mean CE (strongest repair candidate)."""
    n = len(next(iter(results.values()))["effects"]["de"])
    means = [(L, _layer_points(results, L, coord)[1].mean()) for L in range(n - 1)]
    return max(means, key=lambda t: t[1])[0]


def _draw(ax, de, ce, title):
    lim = float(np.nanmax(np.abs(np.concatenate([de, ce, [0.5]])))) * 1.1
    ax.axhline(0, color="0.6", lw=0.7, ls="--", zorder=1)
    ax.plot([-lim, lim], [-lim, lim], "--", color="0.4", lw=1, zorder=1, label="1:1 (TE=0)")
    ax.scatter(de, ce, s=28, alpha=0.75, color="tab:blue", zorder=2)
    slope, intercept = np.polyfit(de, ce, 1)
    pred = slope * de + intercept
    r2 = 1 - np.sum((ce - pred) ** 2) / max(np.sum((ce - ce.mean()) ** 2), 1e-12)
    xs = np.array([-lim, lim])
    ax.plot(xs, slope * xs + intercept, "-", color="crimson", lw=1.6, zorder=3)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_xlabel("Direct effect of ablated layer (DE)")
    ax.set_title(f"{title}\nslope={slope:+.2f}  r²={r2:.2f}", fontsize=10)
    return slope, r2


def main():
    args = _parse_args()
    loaded = {}
    for m in args.models:
        p = args.in_dir / f"de_{m}.json"
        if p.exists():
            loaded[m] = json.loads(p.read_text())
    if not loaded:
        raise SystemExit("no input files")

    fig, axes = plt.subplots(
        1, len(loaded), figsize=(4.4 * len(loaded), 4.8), squeeze=False, constrained_layout=True
    )
    print(f"\nbest-repair layer per model ({args.coord}), DE vs CE across 15 tasks:")
    for ax, (m, res) in zip(axes[0], loaded.items(), strict=True):
        L = _best_repair_layer(res, args.coord)
        de, ce = _layer_points(res, L, args.coord)
        slope, r2 = _draw(ax, de, ce, f"{_MODEL_LABELS.get(m, m)} · L{L}")
        print(f"  {m:10s} best-repair L{L}: slope {slope:+.2f}  r² {r2:.2f}")
    axes[0][0].set_ylabel("Compensatory response  CE = DE − TE")
    axes[0][0].legend(loc="lower right", fontsize=8)
    fig.suptitle(
        f"Hydra Fig-b analog · best-repair layer, across tasks · {args.coord}", fontweight="bold"
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=140)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
