"""Hydra Fig-4d analog — per-layer compensation law across depth.

Hydra's real claim isn't one hand-picked layer: it's that CE~DE regression **R² + slope**
form a systematic band across middle-late layers (Fig 4d), with layer 23 the apex
(R²=0.92, slope=0.69). This plots that whole curve for the TFMs.

Per layer L, per model: pool **per-row** (DE, CE) over all 15 tasks (z-scored per task
by clean-row spread — Hydra's per-prompt analog, thousands of points), regress CE on DE.

- **top row:** R²(L) vs depth — dashed line = Hydra 0.92.
- **bottom row:** slope(L) vs depth — shaded (0,1) = self-repair band, dashed = Hydra 0.69.

Self-repair at a layer ⟺ **both** high R² AND slope in (0,1). A layer with tiny DE spread
(redundant stripe) is hollow-marked — its slope is not meaningful.

⚠️ CE = DE − TE is **method-A** (mechanical coupling: CE carries a +DE term → the
regression is *biased toward* a spurious positive law). So this is a generous test;
failing it is robust. A coupling-free version needs the independent CE (Part 2).

⚠️ margin: ``--agg`` reduces by median (median-of-medians, D1 convention), the per-row
default by mean-over-rows → **different (non-commuting) estimators** (see layerwise_margin).
Use ``--coord gt_logit`` for the exact aggregate == mean(per-row) correspondence.

    uv run --group viz python scripts/plot_ce_de_law.py --coord margin
    uv run --group viz python scripts/plot_ce_de_law.py --coord margin --agg --apex-scatter
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
_DE_SPREAD_MIN = 0.1  # σ; below this a layer has no DE variation → slope not meaningful


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models", nargs="+", default=["limix_2m", "mitra", "tabicl_v2", "tabfm"])
    p.add_argument("--in-dir", type=Path, default=Path("out"))
    p.add_argument("--out", type=Path, default=Path("out/ce_de_law.png"))
    p.add_argument("--coord", choices=["gt_logit", "margin"], default="margin")
    p.add_argument(
        "--agg",
        action="store_true",
        help="regress on task-aggregated DE/CE (15 pts/layer, independent) instead of per-row",
    )
    p.add_argument(
        "--apex-scatter",
        action="store_true",
        help="instead of the depth curve, scatter DE vs CE at each model's apex (peak-R²) layer",
    )
    return p.parse_args()


def _scale(eff, coord, agg=False):
    if coord == "gt_logit":
        return max(eff["zscore"]["sigma"], 1e-6)
    if agg:  # match D1 aggregate scatter
        return max(abs(eff["clean"]["margin"]), 1e-6)
    return max(float(np.std(eff["clean_rows"]["margin"])), 1e-6)


def _n_layers(results):
    eff = next(iter(results.values()))["effects"]
    return len(eff["de"])


def _layer_pts(results, L, coord, agg=False):
    """Pooled (DE, CE) points at layer ``L``: one per task (agg) or one per test-row."""
    de, ce = [], []
    for r in results.values():
        eff = r["effects"]
        s = _scale(eff, coord, agg)
        if agg:
            d = np.array([eff["de"][str(L)][coord] / s])
            t = np.array([eff["te"][str(L)][coord] / s])
        else:
            d = np.asarray(eff["de_rows"][str(L)][coord]) / s
            t = np.asarray(eff["te_rows"][str(L)][coord]) / s
        k = np.isfinite(d) & np.isfinite(t)
        de.append(d[k])
        ce.append((d - t)[k])
    return np.concatenate(de), np.concatenate(ce)


def _fit(de, ce):
    """(slope, intercept, R²) of CE~DE; R²=nan if DE or CE has ~no spread (÷0)."""
    if float(np.std(de)) < 1e-6 or float(np.std(ce)) < 1e-4:
        return np.nan, np.nan, np.nan
    m, b = np.polyfit(de, ce, 1)
    pred = m * de + b
    r2 = 1 - np.sum((ce - pred) ** 2) / max(np.sum((ce - ce.mean()) ** 2), 1e-12)
    return m, b, r2


def _layer_curve(results, coord, agg=False):
    """Per layer: slope, R², DE spread. ``agg``: one pt/task (independent) else per-row."""
    slope, r2, spread = [], [], []
    for L in range(_n_layers(results)):
        de, ce = _layer_pts(results, L, coord, agg)
        m, _, r = _fit(de, ce)
        slope.append(m)
        r2.append(r)
        spread.append(float(np.std(de)))
    return np.array(slope), np.array(r2), np.array(spread)


def _apex_layer(results, coord, agg=False):
    """Peak-R² layer among those with meaningful DE spread, final layer excluded
    (no downstream ⇒ CE≡0). Hydra's Fig-4b apex criterion. -1 if none."""
    slope, r2, spread = _layer_curve(results, coord, agg)
    cand = r2.copy()
    cand[spread < _DE_SPREAD_MIN] = -np.inf
    cand[-1] = -np.inf  # final layer has no downstream
    cand[~np.isfinite(cand)] = -np.inf
    return int(np.argmax(cand)) if np.isfinite(cand).any() and cand.max() > -np.inf else -1


def _draw_apex_scatter(ax, de, ce, m, b, r2, title, agg):
    lim = (
        float(np.max(np.abs(np.concatenate([de, ce, [0.3]]))))
        if agg
        else float(np.percentile(np.abs(np.concatenate([de, ce])), 99))
    ) * 1.15
    xs = np.array([-lim, lim])
    ax.plot(xs, xs, "--", color="0.4", lw=1.2, label="1:1 (CE=DE, full repair)")
    ax.axhline(0, color="0.7", lw=0.6)
    ax.axvline(0, color="0.7", lw=0.6)
    if agg:
        ax.scatter(de, ce, s=45, color="tab:blue", alpha=0.8, zorder=3)
    else:
        ax.hexbin(de, ce, gridsize=50, bins="log", cmap="viridis", extent=(-lim, lim, -lim, lim))
    ax.plot(xs, m * xs + b, "-", color="crimson", lw=2.0, label=f"fit slope={m:+.2f} R²={r2:.2f}")
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("DE (z-scored)")
    ax.legend(fontsize=8, loc="upper left")


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

    if args.apex_scatter:
        _main_apex_scatter(loaded, args)
        return

    n = len(loaded)
    fig, axes = plt.subplots(2, n, figsize=(4.0 * n, 6.4), squeeze=False, sharex="col")
    print(f"\nper-layer CE~DE law ({args.coord}) — peak R² per model:")
    for j, (m, res) in enumerate(loaded.items()):
        slope, r2, spread = _layer_curve(res, args.coord, args.agg)
        L = np.arange(len(slope))
        solid = spread >= _DE_SPREAD_MIN  # meaningful DE variation
        apex = _apex_layer(res, args.coord, args.agg)
        print(
            f"  {m:10s} apex L{apex}: R²={r2[apex]:.2f} slope={slope[apex]:+.2f}"
            if apex >= 0
            else f"  {m:10s} no layer with DE spread"
        )

        ax_r, ax_s = axes[0][j], axes[1][j]
        ax_r.axhline(0.92, ls="--", color="crimson", lw=1, label="Hydra 0.92")
        ax_r.plot(L[solid], r2[solid], "o-", color="tab:blue", ms=4)
        ax_r.plot(L[~solid], r2[~solid], "o", mfc="white", mec="tab:blue", ms=4)
        ax_r.set_ylim(-0.05, 1.0)
        ax_r.set_title(_MODEL_LABELS.get(m, m), fontsize=11)
        if apex >= 0:
            ax_r.axvline(apex, color="0.7", lw=0.8, zorder=0)

        ax_s.axhspan(0, 1, color="tab:green", alpha=0.10)  # self-repair band
        ax_s.axhline(0.69, ls="--", color="crimson", lw=1, label="Hydra 0.69")
        ax_s.axhline(0, color="0.6", lw=0.7)
        ax_s.axhline(1, color="0.6", lw=0.7, ls=":")
        ax_s.plot(L[solid], slope[solid], "o-", color="tab:blue", ms=4)
        ax_s.plot(L[~solid], slope[~solid], "o", mfc="white", mec="tab:blue", ms=4)
        ax_s.set_ylim(-1.5, 2.0)
        ax_s.set_xlabel("ablated layer")
        if apex >= 0:
            ax_s.axvline(apex, color="0.7", lw=0.8, zorder=0)

    axes[0][0].set_ylabel("R²  (CE~DE across rows)")
    axes[1][0].set_ylabel("slope  (CE/DE)")
    axes[0][-1].legend(fontsize=8, loc="upper left")
    axes[1][-1].legend(fontsize=8, loc="upper left")
    unit = "task-aggregated (15 pts/layer)" if args.agg else "per-row (pooled, correlated)"
    fig.suptitle(
        f"Per-layer compensation law · {args.coord} · {unit} — Hydra Fig-4d analog "
        "(hollow = redundant layer, no DE spread; CE=DE−TE is method-A, coupling-biased)",
        fontweight="bold",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=140)
    print(f"\nwrote {args.out}")


def _main_apex_scatter(loaded, args):
    """Hydra Fig-4b analog: DE vs CE scatter + fit at each model's apex (peak-R²) layer."""
    n = len(loaded)
    ncol = min(n, 2)
    nrow = (n + ncol - 1) // ncol
    fig, axes = plt.subplots(
        nrow, ncol, figsize=(6.0 * ncol, 5.5 * nrow), squeeze=False, constrained_layout=True
    )
    print(f"\napex-layer DE vs CE fit ({args.coord}):")
    for ax, (m, res) in zip([a for row in axes for a in row], loaded.items(), strict=False):
        L = _apex_layer(res, args.coord, args.agg)
        de, ce = _layer_pts(res, L, args.coord, args.agg)
        slope, b, r2 = _fit(de, ce)
        print(f"  {m:10s} apex L{L}: slope={slope:+.2f} R²={r2:.2f} (n={len(de)})")
        _draw_apex_scatter(
            ax, de, ce, slope, b, r2, f"{_MODEL_LABELS.get(m, m)} · apex L{L} (peak R²)", args.agg
        )
    for ax in [a for row in axes for a in row][n:]:
        ax.axis("off")
    unit = "task-aggregated (15 pts)" if args.agg else "per-row (per-prompt analog)"
    fig.suptitle(
        f"DE vs CE + fit at apex layer (peak R²) · {args.coord} · {unit}\n"
        "Hydra L23: slope +0.69, R² 0.92 (tight, below 1:1)",
        fontweight="bold",
        fontsize=13,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=140)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
