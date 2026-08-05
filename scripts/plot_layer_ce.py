"""D4 — per-layer compensation CE = DE − TE, from ``run_path_patching_sweep.py``.

Mean CE over all layers ≈ 0 can still hide a few layers with real CE>0 (self-repair).
This resolves CE **per layer**: for each layer, the per-task CE points + their mean,
z-scored per task (same normalization as the scatter). CE>0 (above the 0 line) =
downstream compensates = self-repair; CE<0 = amplification.

    uv run --group viz python scripts/plot_layer_ce.py \
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


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models", nargs="+", default=["limix_2m", "mitra", "tabicl_v2", "tabfm"])
    p.add_argument("--in-dir", type=Path, default=Path("out"))
    p.add_argument("--out", type=Path, default=Path("out/layer_ce.png"))
    p.add_argument("--coord", choices=["gt_logit", "margin"], default="margin")
    p.add_argument(
        "--shared-y", action="store_true", help="share y across panels (default: per-panel)"
    )
    return p.parse_args()


def _layer_ce(results, coord):
    """-> ce[layer] = list of per-task CE (normalized per task). Keyed by int layer."""
    ce = {}
    for r in results.values():
        e = r["effects"]
        sigma = e["zscore"]["sigma"] if coord == "gt_logit" else abs(e["clean"]["margin"])
        scale = max(sigma, 1e-6)
        for m_str, d in e["de"].items():
            m = int(m_str)
            val = (d[coord] - e["te"][m_str][coord]) / scale
            ce.setdefault(m, []).append(val)
    return ce


def _draw(ax, ce, title):
    layers = sorted(ce)
    means = [float(np.mean(ce[m])) for m in layers]
    p25 = [float(np.percentile(ce[m], 25)) for m in layers]
    p75 = [float(np.percentile(ce[m], 75)) for m in layers]
    ax.axhspan(0, max(max(p75), 0.01) * 1.2, color="tab:green", alpha=0.06)  # repair side
    ax.axhline(0, color="0.5", lw=0.8)
    for m in layers:
        ax.plot([m] * len(ce[m]), ce[m], ".", color="0.7", ms=3, alpha=0.5, zorder=1)
    ax.fill_between(layers, p25, p75, color="tab:blue", alpha=0.18, zorder=2)
    ax.plot(layers, means, "-o", color="tab:blue", lw=1.5, ms=3, zorder=3, label="mean CE")
    ax.set_title(title)
    ax.set_xlabel("layer")


def main():
    args = _parse_args()
    loaded = {}
    for model in args.models:
        path = args.in_dir / f"de_{model}.json"
        if path.exists():
            loaded[model] = json.loads(path.read_text())
        else:
            print(f"skip {model}: {path} not found")
    if not loaded:
        raise SystemExit("no input files (expected in-dir/de_<model>.json)")

    fig, axes = plt.subplots(
        1,
        len(loaded),
        figsize=(4.5 * len(loaded), 4.2),
        squeeze=False,
        sharey=args.shared_y,
        constrained_layout=True,
    )
    print(f"\nper-layer CE ({args.coord}) — layers with mean CE > 0 (self-repair candidates):")
    for ax, (model, results) in zip(axes[0], loaded.items(), strict=True):
        ce = _layer_ce(results, args.coord)
        _draw(ax, ce, _MODEL_LABELS.get(model, model))
        pos = [(m, float(np.mean(ce[m]))) for m in sorted(ce) if np.mean(ce[m]) > 0]
        pos.sort(key=lambda x: -x[1])
        top = ", ".join(f"L{m}:{v:+.2f}" for m, v in pos[:5]) or "none"
        print(f"  {model:10s} {len(pos)}/{len(ce)} layers CE>0 | top: {top}")
    axes[0][0].set_ylabel(f"CE = DE − TE   ({args.coord}, z-scored)")
    axes[0][0].legend(loc="upper right", fontsize=8)
    fig.suptitle(f"Per-layer compensation CE · {args.coord}  (>0 = self-repair)", fontweight="bold")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=140)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
