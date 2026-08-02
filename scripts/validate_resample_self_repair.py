"""V2 self-repair cross-check: does the *faithful* (resample) ablation independently
show self-repair, and more stably than zero? (#35)

V1 (``validate_resample_donor_health.py``) showed resample is on-manifold +
norm-preserving. V2 reruns the self-repair sweep under **both** ablations and
asks, on the decision **margin**:

- **Claim 1 — self-repair holds:** under resample each ablated layer dips
  (immediate hit) then climbs back toward baseline -> a property of the model,
  not an artifact of zero's off-manifold shock. Zero is shown **for reference**,
  not ground truth.
- **Claim 2 — stability:** resample perturbs *consistently* (small per-layer
  spread, no overshoot); zero, off-manifold, occasionally sends the readout off
  (big swings + overshoots = ablation makes the margin *better* than baseline).

We deliberately do **not** frame this as "resample reproduces zero" — resample is
the better method that happens to land on the same self-repair, so no zero-vs-
resample correlation stat.

Inputs: ``out/v2_{model}_{zero,resample}.json`` (``run_self_repair_sweep.py``
with ``--ablation zero|resample``). Each stores baseline + per-ablated-layer
trajectories under ``sweep["skip"]`` (the generic *ablated* condition, whatever
the ablation) for metrics {auc, gt_logit, margin}.

Per ablated layer ``m`` (metric-normalized, averaged over tasks):

- **immediate** ``imm(m) = baseline[m+1] - ablated_m[m+1]`` — the direct hit.
- **total** ``TE(m) = baseline[-1] - ablated_m[-1]`` — what survives to the final layer.
- **self-repair** ``SR(m) = imm(m) - TE(m)`` — recovered downstream.

Figures:

- ``v2_pair_{model}.png``          margin, zero | resample side-by-side (shared y).
- ``v2_pair_gtlogit_{model}.png``  same, gt_logit (off-manifold blow-up in raw-logit space).
- ``v2_stability.png``             per-layer std(imm) + overshoot count, all models.

    uv run --group viz python scripts/validate_resample_self_repair.py \
        --models limix_2m mitra tabicl_v2 tabfm --in-dir out
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
    "margin": "Margin z_true - z_other (fraction of final)",
}

# overshoot = ablation makes the margin *better* than baseline (imm < 0); a small
# tolerance keeps averaging noise from counting as an overshoot.
_OVERSHOOT_TOL = 0.02


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models", nargs="+", default=["limix_2m", "mitra", "tabicl_v2", "tabfm"])
    p.add_argument("--in-dir", type=Path, default=Path("out"))
    p.add_argument("--out-dir", type=Path, default=Path("out"))
    return p.parse_args()


def _load(in_dir, model, mode):
    """One (model, ablation) sweep: ``out/v2_{model}_{mode}.json`` (mode = zero|resample)."""
    return json.loads((in_dir / f"v2_{model}_{mode}.json").read_text())


def _normalize_task(sweep, native_final, metric):
    """One task's ``(baseline, skips)`` trajectories, normalized for ``metric``.

    - auc: divide by native final AUC (floored 0.5).
    - gt_logit: per-task z-score with the stored clean-baseline (mu, sigma).
    - margin: divide by the baseline final-layer margin (0 = boundary, final -> 1).
    """
    margin_final = max(abs(sweep["baseline"]["margin"][-1]), 1e-6)

    def norm(arr):
        arr = np.array(arr, dtype=float)
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
    """-> (baseline_mean [n_depths], skip_mean [n_layers, n_depths]) averaged over tasks."""
    baselines, skips = [], []
    for r in results.values():
        b, s = _normalize_task(r["sweep"], r["native_final"], metric)
        baselines.append(b)
        skips.append(s)
    return np.mean(baselines, axis=0), np.mean(skips, axis=0)


def _imm(baseline, skip):
    """imm(m) = baseline[m+1] - ablated_m[m+1], the immediate drop right after layer m."""
    return np.array([baseline[m + 1] - skip[m][m + 1] for m in range(skip.shape[0])])


def _draw_panel(ax, baseline, skip, title, metric, ylim=None):
    """Baseline (black) + one trajectory per ablated layer, each with its red-x immediate drop."""
    n_depths = len(baseline)
    n_layers = skip.shape[0]
    depths = np.arange(n_depths)
    colors = plt.get_cmap("viridis")(np.linspace(0, 1, n_layers))
    ax.plot(depths, baseline, "-o", color="black", lw=2, ms=3, zorder=5, label="baseline")
    for m in range(n_layers):
        ax.plot(depths[m + 1 :], skip[m][m + 1 :], "-o", color=colors[m], lw=1, ms=2, alpha=0.7)
        ax.plot(
            [m + 1, m + 1],
            [baseline[m + 1], skip[m][m + 1]],
            "--",
            color=colors[m],
            lw=1.2,
            alpha=0.9,
        )
        ax.plot(m + 1, skip[m][m + 1], "x", color="red", ms=4, mew=1)
    ax.set_title(title)
    ax.set_xlabel("Layer (forward-pass order)")
    if metric == "auc":
        ax.set_ylim(0.5, 1.0)
    elif ylim is not None:
        ax.set_ylim(*ylim)
    ax.legend(loc="lower right", fontsize=8)


def _pair_fig(in_dir, model, metric, out):
    """zero | resample side-by-side for one model, shared y."""
    zb, zs = _normalized(_load(in_dir, model, "zero"), metric)
    rb, rs = _normalized(_load(in_dir, model, "resample"), metric)
    lo = min(zb.min(), zs.min(), rb.min(), rs.min())
    hi = max(zb.max(), zs.max(), rb.max(), rs.max())
    pad = 0.05 * (hi - lo + 1e-9)
    ylim = None if metric == "auc" else (lo - pad, hi + pad)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True, constrained_layout=True)
    _draw_panel(axes[0], zb, zs, "zero (reference)", metric, ylim)
    _draw_panel(axes[1], rb, rs, "resample (faithful)", metric, ylim)
    axes[0].set_ylabel(_METRIC_YLABEL[metric])
    fig.suptitle(f"{_MODEL_LABELS[model]} · self-repair · {metric}", fontweight="bold")
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"wrote {out}")


def _stability(in_dir, model):
    """Per-model stability of the immediate effect, on the margin (the decision metric)."""
    out = {}
    for mode in ("zero", "resample"):
        b, s = _normalized(_load(in_dir, model, mode), "margin")
        imm = _imm(b, s)
        out[mode] = dict(
            std=float(np.std(imm)),
            n_overshoot=int(np.sum(imm < -_OVERSHOOT_TOL)),  # ablation improved the margin
            max_overshoot=float(max(0.0, (-imm).max())),
        )
    return out


def _stability_fig(stats, out):
    """Grouped bars: per-model std(imm), zero vs resample; overshoot count annotated."""
    models = list(stats)
    x = np.arange(len(models))
    w = 0.38
    fig, ax = plt.subplots(figsize=(1.8 * len(models) + 2, 5), constrained_layout=True)
    zb = ax.bar(
        x - w / 2, [stats[m]["zero"]["std"] for m in models], w, color="tab:red", label="zero"
    )
    rb = ax.bar(
        x + w / 2,
        [stats[m]["resample"]["std"] for m in models],
        w,
        color="tab:blue",
        label="resample",
    )
    for m, bz, br in zip(models, zb, rb, strict=False):
        ax.annotate(
            f"{stats[m]['zero']['n_overshoot']} ovr",
            (bz.get_x() + bz.get_width() / 2, bz.get_height()),
            ha="center",
            va="bottom",
            fontsize=7,
            color="tab:red",
        )
        ax.annotate(
            f"{stats[m]['resample']['n_overshoot']} ovr",
            (br.get_x() + br.get_width() / 2, br.get_height()),
            ha="center",
            va="bottom",
            fontsize=7,
            color="tab:blue",
        )
    ax.set_xticks(x, [_MODEL_LABELS[m] for m in models])
    ax.set_ylabel("std(imm) over layers  ·  margin (fraction of final)")
    ax.set_title(
        "Stability: resample perturbs consistently; zero swings + overshoots", fontweight="bold"
    )
    ax.legend()
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"wrote {out}")


def main():
    args = _parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Claim 1 — self-repair holds (margin), plus the gt_logit off-manifold view.
    for model in args.models:
        _pair_fig(args.in_dir, model, "margin", args.out_dir / f"v2_pair_{model}.png")
        _pair_fig(args.in_dir, model, "gt_logit", args.out_dir / f"v2_pair_gtlogit_{model}.png")

    # Claim 2 — stability.
    stats = {m: _stability(args.in_dir, m) for m in args.models}
    _stability_fig(stats, args.out_dir / "v2_stability.png")

    print("\nstability (margin) — std(imm) over layers, overshoot count / max:")
    for m in args.models:
        z, r = stats[m]["zero"], stats[m]["resample"]
        print(
            f"  {m:10s}  std: zero {z['std']:.3f}  resample {r['std']:.3f}  |  "
            f"overshoot layers: zero {z['n_overshoot']:2d} resample {r['n_overshoot']:2d}  |  "
            f"max overshoot: zero {z['max_overshoot']:.2f} resample {r['max_overshoot']:.2f}"
        )


if __name__ == "__main__":
    main()
