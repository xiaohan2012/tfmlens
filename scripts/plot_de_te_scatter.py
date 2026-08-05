"""D1 — the DE–TE scatter (Hydra Fig 2c analog), from ``run_path_patching_sweep.py``.

Each point = one (layer, dataset). Axes (both through the native decoder, same ruler):

- x = DE (direct effect, path-patching (method B))
- y = TE (total effect, ablate-and-react)

Read on gt_logit, z-scored per task by its clean-baseline σ (Hydra's logit units) so
tasks/models pool. Diagonal ``y = x`` = no compensation (TE = DE).

- **below diagonal** (y < x, DE > 0)  -> downstream **repair** (self-repair): C1.
- **DE ≈ 0 stripe**                    -> redundant, not self-repair: excluded (C2).
- **above diagonal**                   -> downstream breakage (amplification).

    uv run --group viz python scripts/plot_de_te_scatter.py \
        --models limix_2m mitra tabicl_v2 tabfm --in-dir out
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from tfm_lens.evaluation.de_results import MODEL_LABELS, de_scale, load_de_json

# |DE| below this (σ units) is treated as the redundant stripe — flagged, not counted.
_REDUNDANT_TOL = 0.1


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models", nargs="+", default=["limix_2m", "mitra", "tabicl_v2", "tabfm"])
    p.add_argument("--in-dir", type=Path, default=Path("out"))
    p.add_argument("--out", type=Path, default=Path("out/de_te_scatter.png"))
    p.add_argument("--coord", choices=["gt_logit", "margin"], default="gt_logit")
    return p.parse_args()


def _points(results, coord):
    """-> (de, te, layer) arrays over all (task, layer). gt_logit is z-scored per task
    by its clean σ; margin by its clean final margin."""
    de, te, layer = [], [], []
    for r in results.values():
        eff = r["effects"]
        scale = de_scale(eff, coord, agg=True)
        for m_str, d in eff["de"].items():
            t = eff["te"][m_str]
            de.append(d[coord] / scale)
            te.append(t[coord] / scale)
            layer.append(int(m_str))
    return np.array(de), np.array(te), np.array(layer)


def _draw(ax, de, te, layer, title):
    lim = float(np.nanmax(np.abs(np.concatenate([de, te])))) * 1.1 + 1e-6
    diag = np.array([-lim, lim])
    ax.plot(diag, diag, "--", color="0.4", lw=1, zorder=1, label="TE = DE")
    ax.axvspan(-_REDUNDANT_TOL, _REDUNDANT_TOL, color="0.85", zorder=0)  # redundant stripe
    sc = ax.scatter(de, te, c=layer, cmap="viridis", s=14, alpha=0.7, zorder=2)
    ax.axhline(0, color="0.7", lw=0.6, zorder=1)
    ax.axvline(0, color="0.7", lw=0.6, zorder=1)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_xlabel("DE  (direct effect, path-patching)")
    ax.set_title(title)
    return sc


def _summary(de, te, layer):
    """Repair signal, redundant stripe excluded: fraction below diagonal + mean CE."""
    keep = np.abs(de) > _REDUNDANT_TOL
    if keep.sum() == 0:
        return "no non-redundant points"
    ce = de[keep] - te[keep]  # CE = DE − TE > 0 ⇒ repair
    below = float(np.mean(te[keep] < de[keep]))
    return (
        f"non-redundant {keep.sum()}/{len(de)} | below-diag {below:.0%} | "
        f"mean CE {ce.mean():+.2f}σ | redundant(|DE|≤{_REDUNDANT_TOL}) {int((~keep).sum())}"
    )


def main():
    args = _parse_args()
    loaded = load_de_json(args.in_dir, args.models)

    fig, axes = plt.subplots(
        1, len(loaded), figsize=(4.6 * len(loaded), 4.8), squeeze=False, constrained_layout=True
    )
    print(f"\nDE–TE ({args.coord}) — repair signal per model:")
    sc = None
    for ax, (model, results) in zip(axes[0], loaded.items(), strict=True):
        de, te, layer = _points(results, args.coord)
        sc = _draw(ax, de, te, layer, MODEL_LABELS.get(model, model))
        print(f"  {model:10s}  {_summary(de, te, layer)}")
    axes[0][0].set_ylabel("TE  (total effect, ablate-and-react)")
    fig.colorbar(sc, ax=axes[0], label="layer", shrink=0.8)
    fig.suptitle(f"Direct vs total effect · {args.coord} (z-scored per task)", fontweight="bold")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=140)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
