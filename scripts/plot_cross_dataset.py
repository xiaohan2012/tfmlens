"""Cross-dataset / cross-model views of the exp6 self-repair sweep.

Drills the per-depth sweep down to the task level and plots the net output
effect of skipping each layer:

    TE = skip_final - base_final   (per task, per layer; fraction-of-final margin)

- **red / TE > 0** — delete layer -> model *more* confident (layer net-suppresses).
- **blue / TE < 0** — delete layer -> output *damaged* (layer does irreplaceable work).
- **white / TE ~ 0** — output unchanged (repaired *or* redundant).

Reuses ``plot_self_repair._normalize_task`` so numbers match the trajectory plots.
Reads one sweep JSON per model (``run_self_repair_sweep.py`` output).

Draws three figures:
- ``cross_dataset_heatmap.png``  — 2x2 task x layer TE heatmap, one panel per model,
  shared diverging color scale (magnitudes comparable model-to-model).
- ``cross_dataset_lineplot.png`` — TE vs *relative depth* (layer / max, 0..1 so
  shallow and deep models align), median +/- IQR over tasks, all models overlaid.
- ``cross_dataset_<model>_L<k>.png`` — per-task fan for one (model, layer); shows
  when a near-zero *average* TE hides a bimodal help/hurt split.

    uv run --group viz python scripts/plot_cross_dataset.py
"""

import argparse
import importlib.util
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D

# Model -> (sweep JSON, plot color). Order fixes heatmap panels + line legend.
_MODELS = [
    ("LimiX-2M", "out/logit_limix_2m_median.json", "tab:blue"),
    ("TabICLv2", "out/logit_tabicl_v2_median.json", "tab:orange"),
    ("Mitra", "out/logit_mitra_median.json", "tab:green"),
    ("TabFM", "out/logit_tabfm_median.json", "tab:red"),
]


def _load_psr():
    """Import the sibling ``plot_self_repair.py`` script for its normalizers."""
    here = Path(__file__).resolve().parent
    spec = importlib.util.spec_from_file_location("psr", here / "plot_self_repair.py")
    psr = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(psr)
    return psr


_PSR = _load_psr()


def _load(path, metric):
    """One model's normalized trajectories -> (tasks, base[T,D], skips[T,L,D])."""
    res = json.loads(Path(path).read_text())
    tasks, B, S = [], [], []
    for t in res:
        base, skips = _PSR._normalize_task(res[t]["sweep"], res[t]["native_final"], metric)
        tasks.append(t)
        B.append(base)
        S.append(skips)
    return tasks, np.array(B), np.array(S)


def _te_matrix(path, metric):
    """Net output effect of skipping each layer -> (tasks, TE[T,L])."""
    tasks, B, S = _load(path, metric)
    return tasks, S[:, :, -1] - B[:, -1:]  # skip_final - base_final


def plot_heatmap(metric, out):
    """2x2 task x layer TE heatmap, one panel per model, shared diverging scale."""
    data = [(name, *_te_matrix(path, metric)) for name, path, _ in _MODELS]
    gvmax = max(np.nanpercentile(np.abs(TE), 98) for _, _, TE in data)  # shared scale
    norm = TwoSlopeNorm(vmin=-gvmax, vcenter=0, vmax=gvmax)
    print(f"shared vmax = {gvmax:.3f}")

    fig, axes = plt.subplots(2, 2, figsize=(17, 11), constrained_layout=True)
    im = None
    for ax, (name, tasks, TE) in zip(axes.ravel(), data, strict=True):
        im = ax.imshow(TE, aspect="auto", cmap="RdBu_r", norm=norm)
        ax.set_title(f"{name} · net output effect (TE) of skipping each layer", fontsize=11)
        ax.set_xlabel("ablated layer")
        ax.set_ylabel("TabArena binary task")
        ax.set_xticks(range(TE.shape[1]))
        ax.set_xticklabels(range(TE.shape[1]), fontsize=7)
        ax.set_yticks(range(len(tasks)))
        ax.set_yticklabels([t[:8] for t in tasks], fontsize=7)
    fig.colorbar(
        im,
        ax=axes,
        label="TE (skip_final − base_final), fraction-of-final margin  ·  "
        "red = overshoot/suppressive, blue = damage",
        fraction=0.03,
        pad=0.02,
    )
    fig.suptitle(
        "Cross-dataset (shared color scale): same layer, different data → "
        "different (often opposite-signed) reaction",
        fontsize=13,
        fontweight="bold",
    )
    fig.savefig(out, dpi=140)
    print(f"wrote {out}")


def plot_lineplot(metric, out):
    """TE vs relative depth (median +/- IQR over tasks), all models overlaid.

    x = layer / (n_layers - 1) so shallow (12L) and deep (24L) models align:
    0 = first layer, 1 = last.
    """
    fig, ax = plt.subplots(figsize=(11, 6), constrained_layout=True)
    ax.axhline(0, ls="--", color="0.4", lw=1.2, zorder=1)
    for name, path, c in _MODELS:
        _, TE = _te_matrix(path, metric)
        n = TE.shape[1]
        x = np.arange(n) / (n - 1)  # relative depth 0..1
        med = np.median(TE, 0)
        lo = np.percentile(TE, 25, 0)
        hi = np.percentile(TE, 75, 0)
        ax.fill_between(x, lo, hi, color=c, alpha=0.15, zorder=2)
        ax.plot(x, med, "-o", color=c, ms=4, lw=2, label=f"{name} ({n}L)", zorder=3)
    ax.text(
        0.012,
        0.97,
        "↑ TE > 0  = suppressive / over-confident (delete → more confident)",
        transform=ax.transAxes,
        va="top",
        fontsize=9,
        color="0.25",
    )
    ax.text(
        0.012,
        0.03,
        "↓ TE < 0  = damage (delete → worse)",
        transform=ax.transAxes,
        va="bottom",
        fontsize=9,
        color="0.25",
    )
    ax.set_xlabel("relative depth  (layer index / max,  0 = first layer · 1 = last)")
    ax.set_ylabel("TE = skip_final − base_final  (fraction-of-final margin)")
    ax.set_title(
        "Cross-dataset net output effect vs relative depth — median ± IQR over tasks, all models"
    )
    ax.set_xlim(-0.02, 1.02)
    ax.legend(loc="upper right", fontsize=9)
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


def plot_task_fan(model, layer, metric, out):
    """Per-task fan for one (model, layer): a near-zero average TE can hide a split.

    Colors each task's post-skip trajectory by the sign of its TE (red = removing
    the layer helps, blue = layer irreplaceable); overlays the task-average.
    """
    path = dict((n, p) for n, p, _ in _MODELS)[model]
    tasks, B, S = _load(path, metric)
    depths = np.arange(B.shape[1])
    base_avg = B.mean(0)
    TE = S[:, layer, -1] - B[:, -1]

    fig, ax = plt.subplots(figsize=(8, 5.5), constrained_layout=True)
    for i in range(len(tasks)):
        c = "tab:red" if TE[i] > 0.05 else ("tab:blue" if TE[i] < -0.05 else "0.7")
        ax.plot(depths[layer + 1 :], S[i, layer, layer + 1 :], "-", color=c, lw=1, alpha=0.6)
    ax.plot(depths, base_avg, "k-", lw=2.5, marker="o", ms=3, zorder=5)
    ax.plot(
        depths[layer + 1 :],
        S[:, layer, layer + 1 :].mean(0),
        "--",
        color="0.25",
        lw=2.5,
        marker="s",
        ms=4,
        zorder=6,
    )
    ax.axhline(1.0, color="0.5", ls=":", lw=1)
    n_pos, n_neg = int((TE > 0.05).sum()), int((TE < -0.05).sum())
    n_flat = int((np.abs(TE) <= 0.05).sum())
    pos_lbl = f"TE>0 (removing L{layer} helps)  n={n_pos}"
    neg_lbl = f"TE<0 (L{layer} irreplaceable)  n={n_neg}"
    ax.legend(
        handles=[
            Line2D([], [], color="k", lw=2.5, label="baseline (avg)"),
            Line2D([], [], color="0.25", ls="--", lw=2.5, label=f"skip L{layer} (avg)"),
            Line2D([], [], color="tab:red", lw=1.5, label=pos_lbl),
            Line2D([], [], color="tab:blue", lw=1.5, label=neg_lbl),
            Line2D([], [], color="0.7", lw=1.5, label=f"|TE|<0.05  n={n_flat}"),
        ],
        loc="lower right",
        fontsize=8,
    )
    ax.set_xlabel("Layer (forward-pass order)")
    ax.set_ylabel("Median margin (fraction of final)")
    ax.set_title(
        f"{model} · skip layer {layer} · per task\n"
        f"avg TE={TE.mean():+.2f} (~redundant) but split {n_pos} help / {n_neg} hurt"
    )
    fig.savefig(out, dpi=140)
    print(f"wrote {out}")


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--metric", default="margin", choices=["margin", "gt_logit"], help="TE metric")
    p.add_argument("--out-dir", type=Path, default=Path("out"), help="output directory")
    p.add_argument("--fan-model", default="Mitra", help="model for the per-task fan figure")
    p.add_argument("--fan-layer", type=int, default=7, help="layer for the per-task fan figure")
    return p.parse_args()


def main():
    args = _parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    plot_heatmap(args.metric, args.out_dir / "cross_dataset_heatmap.png")
    plot_lineplot(args.metric, args.out_dir / "cross_dataset_lineplot.png")
    plot_task_fan(
        args.fan_model,
        args.fan_layer,
        args.metric,
        args.out_dir / f"cross_dataset_{args.fan_model.lower()}_L{args.fan_layer}.png",
    )


if __name__ == "__main__":
    main()
