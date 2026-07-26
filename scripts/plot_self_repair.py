"""Plot the exp6 self-repair trajectories (paper Figure 8) from the sweep JSON.

Reads ``out/self_repair.json`` (written by run_self_repair_sweep.py). Per dataset every
trajectory is normalized by that dataset's native final-layer AUC (floored 0.5),
then averaged across datasets. Draws:

  - black baseline: the fine-tuned decode per depth, no ablation
  - one colored line per ablated layer m, from depth m+1 onward (post-skip)
  - a dashed red-x connector marking the immediate drop at depth m+1

Self-repair shows up as a sharp drop right after the skipped layer that the
later layers partly recover.

    uv run --group viz python scripts/plot_self_repair.py
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
    p.add_argument("--in", dest="inp", type=Path, default=Path("out/self_repair.json"))
    p.add_argument("--out", type=Path, default=Path("out/self_repair_fig8.png"))
    p.add_argument("--model", choices=list(_MODEL_LABELS), default="limix_2m", help="title label")
    return p.parse_args()


def _normalized(results):
    """-> (baseline_mean, skip_mean) arrays averaged over datasets.

    ``baseline_mean`` is ``[n_depths]``; ``skip_mean`` is ``[n_layers, n_depths]``
    (row m = trajectory with layer m skipped). Each dataset normalized by its
    native final AUC (floored 0.5) before averaging.
    """
    baselines, skips = [], []
    for r in results.values():
        norm = max(r["native_final"], 0.5)
        baselines.append(np.array(r["sweep"]["baseline"]) / norm)
        skip = r["sweep"]["skip"]
        skips.append(np.array([skip[str(m)] for m in range(len(skip))]) / norm)
    return np.mean(baselines, axis=0), np.mean(skips, axis=0)


def main():
    args = _parse_args()
    results = json.loads(args.inp.read_text())
    baseline, skip = _normalized(results)
    n_depths = len(baseline)
    n_layers = skip.shape[0]
    depths = np.arange(n_depths)
    colors = plt.get_cmap("viridis")(np.linspace(0, 1, n_layers))

    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    ax.plot(
        depths,
        baseline,
        "-",
        color="black",
        linewidth=2,
        marker="o",
        markersize=3,
        label="baseline (no ablation)",
        zorder=5,
    )

    for m in range(n_layers):
        xs = depths[m + 1 :]
        ys = skip[m][m + 1 :]
        ax.plot(xs, ys, "-", color=colors[m], linewidth=1, marker="o", markersize=2, alpha=0.7)
        # immediate drop at depth m+1: baseline -> ablated
        ax.plot(
            [m + 1, m + 1],
            [baseline[m + 1], skip[m][m + 1]],
            "--",
            color=colors[m],
            linewidth=1.2,
            alpha=0.9,
        )
        ax.plot(m + 1, skip[m][m + 1], "x", color="red", markersize=4, markeredgewidth=1)

    ax.set_xlabel("Layer (forward-pass order)")
    ax.set_ylabel("Normalized performance (ROC-AUC)")
    ax.set_title(f"{_MODEL_LABELS[args.model]} self-repair ({len(results)} TabArena binary tasks)")
    ax.legend(loc="lower right")

    sm = plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(0, n_layers - 1))
    fig.colorbar(sm, ax=ax, label="ablated layer")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
