"""D2 — CE vs DE (Hydra's compensation-law plot), from ``run_path_patching_sweep.py``.

Puts the two quantities on one axis: x = DE (direct effect), y = CE = DE − TE
(compensation). Self-repair = CE>0 among DE>0 layers. A Hydra-style compensation
law shows as a positive slope (fraction of DE compensated) with high R².

- grey stripe |DE|≤tol = overshoot zone (CE>0 there is not self-repair) — excluded from the fit.
- dashed y=x  = full compensation (TE=0); y=0 = no compensation.
- fit on |DE|>tol only → slope + R² printed per model (compare Hydra 0.69 / 0.92).

    uv run --group viz python scripts/plot_ce_vs_de.py \
        --models limix_2m mitra tabicl_v2 tabfm --in-dir out --coord margin
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
_TOL = 0.1  # |DE| below this = overshoot zone, not self-repair; excluded from the fit


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models", nargs="+", default=["limix_2m", "mitra", "tabicl_v2", "tabfm"])
    p.add_argument("--in-dir", type=Path, default=Path("out"))
    p.add_argument("--out", type=Path, default=Path("out/ce_vs_de.png"))
    p.add_argument("--coord", choices=["gt_logit", "margin"], default="margin")
    return p.parse_args()


def _points(results, coord):
    de, ce, layer = [], [], []
    for r in results.values():
        e = r["effects"]
        scale = max(
            e["zscore"]["sigma"] if coord == "gt_logit" else abs(e["clean"]["margin"]), 1e-6
        )
        for k, d in e["de"].items():
            x = d[coord] / scale
            de.append(x)
            ce.append(x - e["te"][k][coord] / scale)
            layer.append(int(k))
    return np.array(de), np.array(ce), np.array(layer)


def _fit(de, ce):
    """OLS CE~DE on the DE>tol points → (slope, r2, n). None if too few."""
    keep = np.abs(de) > _TOL
    if keep.sum() < 3:
        return None
    x, y = de[keep], ce[keep]
    slope, intercept = np.polyfit(x, y, 1)
    pred = slope * x + intercept
    ss_res = np.sum((y - pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return slope, intercept, r2, int(keep.sum())


def _draw(ax, de, ce, layer, title):
    lim = float(np.nanmax(np.abs(np.concatenate([de, ce])))) * 1.1 + 1e-6
    ax.axvspan(-_TOL, _TOL, color="0.88", zorder=0)  # overshoot zone
    ax.axhline(0, color="0.6", lw=0.7, zorder=1)
    ax.plot([-lim, lim], [-lim, lim], "--", color="0.5", lw=1, zorder=1, label="CE=DE (TE=0)")
    ax.scatter(de, ce, c=layer, cmap="viridis", s=14, alpha=0.7, zorder=2)
    fit = _fit(de, ce)
    if fit:
        slope, intercept, r2, n = fit
        xs = np.array([_TOL, lim])
        ax.plot(xs, slope * xs + intercept, "-", color="crimson", lw=1.8, zorder=3)
        title = f"{title}\nslope={slope:+.2f}  R²={r2:.2f}  (n={n}, DE>{_TOL})"
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_xlabel("DE (direct effect)")
    ax.set_title(title, fontsize=10)


def main():
    args = _parse_args()
    loaded = {}
    for m in args.models:
        p = args.in_dir / f"de_{m}.json"
        if p.exists():
            loaded[m] = json.loads(p.read_text())
        else:
            print(f"skip {m}: {p} not found")
    if not loaded:
        raise SystemExit("no input files (expected in-dir/de_<model>.json)")

    fig, axes = plt.subplots(
        1, len(loaded), figsize=(4.6 * len(loaded), 5.0), squeeze=False, constrained_layout=True
    )
    print(f"\nCE~DE fit ({args.coord}, on DE>{_TOL}) — vs Hydra slope 0.69 / R² 0.92:")
    for ax, (m, res) in zip(axes[0], loaded.items(), strict=True):
        de, ce, layer = _points(res, args.coord)
        _draw(ax, de, ce, layer, _MODEL_LABELS.get(m, m))
        fit = _fit(de, ce)
        msg = (
            f"slope {fit[0]:+.2f}  R² {fit[2]:.2f}  n={fit[3]}" if fit else "too few DE>tol points"
        )
        print(f"  {m:10s} {msg}")
    axes[0][0].set_ylabel("CE = DE − TE   (compensation)")
    axes[0][0].legend(loc="lower right", fontsize=8)
    fig.suptitle(f"Compensation law: CE vs DE · {args.coord}", fontweight="bold")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=140)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
