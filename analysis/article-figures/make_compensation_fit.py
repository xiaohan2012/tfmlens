"""Figure 12 — the compensation law, as one plane instead of eight panels.

The criterion is joint: a layer passes only if its CE~DE fit has BOTH a high
R^2 AND a slope in (0, 1). Splitting the two numbers across two rows of panels
makes the reader check the conjunction by eye. Here each layer is one point in
the (slope, R^2) plane, so the criterion is a region and the language-model
reference is a point in it.

Data: tfmlens out/de_{model}.json, same fit as scripts/plot_compensation_fit.py.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]
FIGDIR = REPO / "out/article-figures"
FIGDIR.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(REPO / "src"))
from tfm_lens.evaluation.de_results import MODEL_LABELS, de_scale, load_de_json  # noqa: E402

COORD = "margin"
SPREAD_MIN = 0.1  # below this a layer has no DE variation -> slope meaningless
IN = REPO / "out"
MODELS = ["limix_2m", "mitra", "tabicl_v2", "tabfm"]
COLORS = {"limix_2m": "#3b6ea5", "mitra": "#e0912a", "tabicl_v2": "#4c9a6b", "tabfm": "#a2559c"}
MARKERS = {"limix_2m": "o", "mitra": "s", "tabicl_v2": "^", "tabfm": "D"}
HYDRA = (0.69, 0.92)


def layer_fits(results):
    """-> list of (slope, r2, de_spread) per layer, one (DE, CE) point per task."""
    n = len(next(iter(results.values()))["effects"]["de"])
    out = []
    for L in range(n):
        de, ce = [], []
        for r in results.values():
            eff = r["effects"]
            s = de_scale(eff, COORD, agg=True)
            d = eff["de"][str(L)][COORD] / s
            t = eff["te"][str(L)][COORD] / s
            if np.isfinite(d) and np.isfinite(t):
                de.append(d)
                ce.append(d - t)
        de, ce = np.array(de), np.array(ce)
        if len(de) < 3 or np.std(de) < 1e-6 or np.std(ce) < 1e-4:
            out.append((np.nan, np.nan, float(np.std(de)) if len(de) else 0.0))
            continue
        m, b = np.polyfit(de, ce, 1)
        r2 = 1 - np.sum((ce - (m * de + b)) ** 2) / max(np.sum((ce - ce.mean()) ** 2), 1e-12)
        out.append((m, r2, float(np.std(de))))
    return out


fig, ax = plt.subplots(figsize=(8.8, 6.4))

# the region a layer must land in to count as a compensation law
ax.axvspan(0, 1, color="#4c9a6b", alpha=0.07, zorder=0)
ax.axhline(0, color="0.85", lw=0.8, zorder=1)
ax.axvline(0, color="0.85", lw=0.8, zorder=1)

loaded = load_de_json(IN, MODELS)
for model, results in loaded.items():
    pts = [(sl, r) for sl, r, sp in layer_fits(results) if np.isfinite(sl)]
    xs, ys = zip(*pts, strict=True)
    ax.scatter(
        xs,
        ys,
        s=66,
        color=COLORS[model],
        marker=MARKERS[model],
        alpha=0.85,
        edgecolors="white",
        linewidths=0.6,
        zorder=3,
        label=f"{MODEL_LABELS.get(model, model)}  ({len(pts)} layers)",
    )

ax.plot(*HYDRA, "*", ms=26, color="#d1462f", zorder=5)
ax.annotate(
    "Chinchilla 7B, layer 23\n(0.69, 0.92)",
    xy=HYDRA,
    xytext=(-14, -6),
    textcoords="offset points",
    ha="right",
    va="top",
    fontsize=11,
    color="#d1462f",
    fontweight="bold",
    linespacing=1.35,
    zorder=5,
)

ax.set_xlim(-1.6, 1.6)
ax.set_ylim(-0.03, 1.0)
ax.set_xlabel(
    r"slope of $CE$ on $DE$   (share of the direct contribution that comes back)", fontsize=11.5
)
ax.set_ylabel(r"$R^2$   (how reliably it comes back, across tasks)", fontsize=11.5)
ax.set_title(
    "A compensation law would put a layer in the shaded band, high up",
    fontsize=13.5,
    fontweight="bold",
    pad=12,
)
ax.spines[["top", "right"]].set_visible(False)
ax.grid(alpha=0.15, lw=0.7)
ax.legend(fontsize=10.5, frameon=False, loc="upper left")
fig.tight_layout()

out = FIGDIR.joinpath("compensation-fit.png")
fig.savefig(out, dpi=170, facecolor="white")
print(f"wrote {out}")

for model, results in loaded.items():
    fits = layer_fits(results)
    ok = [(i, s, r) for i, (s, r, sp) in enumerate(fits) if np.isfinite(s) and sp >= SPREAD_MIN]
    best = max(ok, key=lambda t: t[2]) if ok else None
    print(
        f"{MODEL_LABELS.get(model, model):10s} layers plotted {len(ok):2d} | "
        f"peak R^2 at L{best[0]}: slope {best[1]:+.2f}, R2 {best[2]:.2f}"
    )
