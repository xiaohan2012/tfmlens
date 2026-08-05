"""Per-row DE–TE — does row-aggregation hide a self-repair signal?

The aggregate D1 (``plot_de_te_scatter.py``) collapses each table's ~500 test rows to
one number. A layer's aggregate CE≈0 can mean **all-zero** OR **canceling rows** (some
repair, some break). This un-aggregates: each point = one (task, layer, test-row).

Needs ``de_<model>.json`` written with ``--per-row`` (has ``de_rows``/``te_rows``).

Per model, rows pooled over layers/tasks, z-scored per task by the clean-row spread:

- **top:** hexbin DE vs TE. Diagonal ``y = x`` = no compensation. Below-diagonal mass
  (DE > 0, TE < DE) = per-row self-repair — a lobe here that the aggregate mean washes out.
- **bottom:** histogram of per-row CE = DE − TE over rows with |DE| > tol. Symmetric about
  0 → cancellation, not repair; fat positive tail → hidden repair the mean hides.

    uv run --group viz python scripts/plot_de_te_perrow.py \
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
_REDUNDANT_TOL = 0.1  # |DE| below this (σ units) = redundant stripe, excluded from CE stats


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models", nargs="+", default=["limix_2m", "mitra", "tabicl_v2", "tabfm"])
    p.add_argument("--in-dir", type=Path, default=Path("out"))
    p.add_argument("--out", type=Path, default=Path("out/de_te_perrow.png"))
    p.add_argument("--coord", choices=["gt_logit", "margin"], default="margin")
    p.add_argument(
        "--hist-only",
        action="store_true",
        help="draw only the per-row CE histogram row (drop the hexbin)",
    )
    return p.parse_args()


def _task_scale(eff, coord):
    """Per-task z-score unit = spread of the clean per-row coordinate."""
    if coord == "gt_logit":
        return max(eff["zscore"]["sigma"], 1e-6)
    return max(float(np.std(eff["clean_rows"]["margin"])), 1e-6)


def _points(results, coord):
    """Pool per-row (DE, TE) over all (task, layer), each z-scored by its clean-row spread."""
    de, te = [], []
    for r in results.values():
        eff = r["effects"]
        if "de_rows" not in eff:
            raise SystemExit("json has no de_rows — re-run the sweep with --per-row")
        scale = _task_scale(eff, coord)
        for m_str, d in eff["de_rows"].items():
            de.append(np.asarray(d[coord]) / scale)
            te.append(np.asarray(eff["te_rows"][m_str][coord]) / scale)
    return np.concatenate(de), np.concatenate(te)


def _draw_hexbin(ax, de, te, title):
    keep = np.isfinite(de) & np.isfinite(te)
    de, te = de[keep], te[keep]
    lim = float(np.percentile(np.abs(np.concatenate([de, te])), 99)) * 1.1 + 1e-6
    diag = np.array([-lim, lim])
    ax.hexbin(de, te, gridsize=60, bins="log", cmap="viridis", extent=(-lim, lim, -lim, lim))
    ax.plot(diag, diag, "--", color="w", lw=1, zorder=3, label="TE = DE")
    ax.axhline(0, color="w", lw=0.7, alpha=0.6, zorder=2)  # y = 0
    ax.axvline(0, color="w", lw=0.7, alpha=0.6, zorder=2)  # x = 0
    ax.axvspan(-_REDUNDANT_TOL, _REDUNDANT_TOL, color="w", alpha=0.15, zorder=1)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_xlabel("DE  (per-row, z-scored)")
    ax.set_title(title, fontsize=10)


def _draw_ce_hist(ax, de, te):
    keep = np.isfinite(de) & np.isfinite(te) & (np.abs(de) > _REDUNDANT_TOL)
    ce = (de - te)[keep]
    lim = float(np.percentile(np.abs(ce), 99)) * 1.1 + 1e-6
    ax.hist(ce, bins=80, range=(-lim, lim), color="tab:blue", alpha=0.8)
    ax.axvline(0, color="0.4", lw=0.8, ls="--")
    ax.axvline(ce.mean(), color="crimson", lw=1.4, label=f"mean {ce.mean():+.2f}")
    ax.axvline(np.median(ce), color="darkorange", lw=1.4, label=f"median {np.median(ce):+.2f}")
    ax.set_xlim(-lim, lim)
    ax.set_xlabel("per-row CE = DE − TE  (|DE| > tol)")
    ax.legend(fontsize=8)
    return ce


def _summary(de, te):
    keep = np.isfinite(de) & np.isfinite(te) & (np.abs(de) > _REDUNDANT_TOL)
    if keep.sum() == 0:
        return "no non-redundant rows"
    ce = (de - te)[keep]
    pos = float(np.mean(de[keep] > 0))  # share of non-redundant rows with DE>0
    below = float(np.mean(te[keep] < de[keep]))  # CE>0 (repair direction)
    return (
        f"rows {keep.sum():>6}/{len(de)} | DE>0 {pos:.0%} | below-diag(CE>0) {below:.0%} | "
        f"mean CE {ce.mean():+.2f} | median CE {np.median(ce):+.2f}"
    )


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
        raise SystemExit("no input files (expected in-dir/de_<model>.json with --per-row)")

    n = len(loaded)
    nrow = 1 if args.hist_only else 2
    fig, axes = plt.subplots(
        nrow, n, figsize=(4.2 * n, 3.8 * nrow), squeeze=False, constrained_layout=True
    )
    print(f"\nper-row DE–TE ({args.coord}) — is aggregate CE≈0 real or cancellation?")
    for j, (m, res) in enumerate(loaded.items()):
        de, te = _points(res, args.coord)
        if args.hist_only:
            _draw_ce_hist(axes[0][j], de, te)
            axes[0][j].set_title(_MODEL_LABELS.get(m, m), fontsize=11)
        else:
            _draw_hexbin(axes[0][j], de, te, _MODEL_LABELS.get(m, m))
            _draw_ce_hist(axes[1][j], de, te)
        print(f"  {m:10s}  {_summary(de, te)}")
    if not args.hist_only:
        axes[0][0].set_ylabel("TE  (per-row, z-scored)")
    axes[-1][0].set_ylabel("row count")
    title = (
        f"Per-row CE distribution · {args.coord} — unimodal at 0 (not cancellation/dilution)"
        if args.hist_only
        else f"Per-row DE vs TE · {args.coord} — row-level sign cancellation check"
    )
    fig.suptitle(title, fontweight="bold")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=140)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
