"""Plot the exp6 self-repair trajectories (paper Figure 8) from the sweep JSON.

Reads the sweep JSON (run_self_repair_sweep.py); per dataset normalize each
trajectory for ``--metric`` (auc: / native final AUC floored 0.5; gt_logit:
per-task z-score with the clean-baseline mu/sigma), then average across datasets.
Draws:

- black baseline: fine-tuned decode per depth, no ablation.
- one colored line per ablated layer m, from depth m+1 onward (post-skip).
- a dashed red-x connector marking the immediate drop at depth m+1.

Self-repair = a sharp drop right after the skipped layer that later layers recover.

    uv run --group viz python scripts/plot_self_repair.py --metric gt_logit
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

_METRIC_YLABEL = {
    "auc": "Normalized performance (ROC-AUC)",
    "gt_logit": "GT-logit (z-scored, σ)",
    "margin": "Median margin z_true - z_other (fraction of final)",
}


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--in", dest="inp", type=Path, default=Path("out/self_repair.json"))
    p.add_argument("--out", type=Path, default=Path("out/self_repair_fig8.png"))
    p.add_argument("--model", choices=list(_MODEL_LABELS), default="limix_2m", help="title label")
    p.add_argument("--metric", choices=list(_METRIC_YLABEL), default="auc", help="y-axis metric")
    return p.parse_args()


def _normalize_task(sweep, native_final, metric):
    """One task's ``(baseline, skips)`` trajectories, normalized for ``metric``.

    - auc: divide by native final AUC (floored 0.5).
    - gt_logit: per-task z-score with the stored clean-baseline (mu, sigma).
    - margin: divide by the baseline final-layer margin (keeps 0 = decision
      boundary; final -> 1, like auc's "fraction of final performance").

    Margin divisor is one per-task scalar (mean row margin at the final layer),
    so this is ratio-of-means, not mean-of-ratios: we do *not* divide each test
    row by its own final margin. Per-row division explodes on boundary rows
    (final margin ~ 0) and isn't the population quantity we want. With a per-task
    scalar the order is moot anyway (dividing by a constant commutes with the
    row-mean); we just normalize the already-averaged trajectory.
    """
    margin_final = max(abs(sweep["baseline"]["margin"][-1]), 1e-6)

    def norm(arr):
        arr = np.array(arr)
        if metric == "auc":
            return arr / max(native_final, 0.5)
        if metric == "margin":
            return arr / margin_final
        z = sweep["zscore"]
        return (arr - z["mu"]) / z["sigma"]

    skip = sweep["skip"]
    baseline = norm(sweep["baseline"][metric])
    skips = np.array([norm(skip[str(m)][metric]) for m in range(len(skip))])
    return baseline, skips


def _normalized(results, metric):
    """-> (baseline_mean, skip_mean) averaged over datasets, normalized for ``metric``.

    ``baseline_mean`` is ``[n_depths]``; ``skip_mean`` is ``[n_layers, n_depths]``
    (row m = trajectory with layer m skipped).
    """
    baselines, skips = [], []
    for r in results.values():
        b, s = _normalize_task(r["sweep"], r["native_final"], metric)
        baselines.append(b)
        skips.append(s)
    return np.mean(baselines, axis=0), np.mean(skips, axis=0)


def main():
    args = _parse_args()
    results = json.loads(args.inp.read_text())
    baseline, skip = _normalized(results, args.metric)
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
    ax.set_ylabel(_METRIC_YLABEL[args.metric])
    if args.metric == "auc":
        ax.set_ylim(0.5, 1.0)  # fixed chance..perfect band; auto-zoom hides ceiling effects
    ax.set_title(
        f"{_MODEL_LABELS[args.model]} self-repair · {args.metric} "
        f"({len(results)} TabArena binary tasks)"
    )
    ax.legend(loc="lower right")

    sm = plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(0, n_layers - 1))
    fig.colorbar(sm, ax=ax, label="ablated layer")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
