"""Figure 3: the layer-skipping experiment — schematic, then the real thing.

Left  : one ablated layer, idealised. The reading key: black baseline, a drop
        at the depth right after the skipped layer, then recovery.
Right : the same experiment on a real model, every layer ablated in turn.
        One coloured curve per skipped layer, so the left panel is what a
        single curve on the right means.

Real data: tfmlens out/self_repair_full.json (LimiX-2M, 15 TabArena binary
tasks). Per task the trajectory is divided by the native final ROC AUC
(floored at 0.5), then averaged over tasks — the normalisation used by
scripts/plot_balef_exp6_trajectory.py.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

REPO = Path(__file__).resolve().parents[2]
FIGDIR = REPO / "out/article-figures"
FIGDIR.mkdir(parents=True, exist_ok=True)
BLACK = "#1a1a1a"
RED = "#d1462f"
BLUE = "#3b6ea5"
GREY = "#8a8a8a"

SRC = REPO / "out/self_repair_full.json"
MODEL = "LimiX-2M"

# ------------------------------------------------------------------ real data
res = json.loads((SRC).read_text())
base, skips = [], []
for r in res.values():
    nf = max(r["native_final"], 0.5)
    base.append(np.array(r["sweep"]["baseline"]) / nf)
    skips.append(
        np.array(
            [np.array(r["sweep"]["skip"][str(m)]) / nf for m in range(len(r["sweep"]["skip"]))]
        )
    )
base = np.mean(base, axis=0)
skips = np.mean(skips, axis=0)  # (n_layers, n_depths)
L = len(base) - 1
n_layers = skips.shape[0]

# ------------------------------------------------------------------ schematic
M = 4
xs = np.arange(L + 1)
sch_base = 0.60 + 0.40 * (1 - np.exp(-0.75 * xs))
sch_base = sch_base / sch_base[-1] * 0.998
dip = 0.66
sch_abl = sch_base.copy()
sch_abl[M + 1] = dip
for i in range(M + 2, L + 1):
    f = (i - (M + 1)) / (L - (M + 1))
    sch_abl[i] = dip + (sch_base[i] - dip) * f**0.62

fig, (axL, axR) = plt.subplots(
    1, 2, figsize=(12.4, 5.0), sharey=True, gridspec_kw=dict(wspace=0.06)
)

# ---- left: one ablated layer, idealised
axL.plot(xs, sch_base, "-o", color=BLACK, lw=2.4, ms=4.5, zorder=3, label="no ablation")
keep = np.arange(M + 1, L + 1)
axL.plot(
    keep, sch_abl[keep], "-o", color=BLUE, lw=2.2, ms=4.2, zorder=3, label=f"layer {M} skipped"
)
axL.plot([M, M + 1], [sch_base[M], dip], "--", color=BLUE, lw=1.6, zorder=2)
axL.plot([M], [sch_base[M]], "x", color=RED, ms=12, mew=2.8, zorder=5)
axL.annotate(
    "",
    xy=(M + 1, dip + 0.012),
    xytext=(M + 1, sch_base[M + 1] - 0.012),
    arrowprops=dict(arrowstyle="-|>", color=GREY, lw=1.3),
)
axL.text(M + 1.25, (dip + sch_base[M + 1]) / 2, "drop", fontsize=10.5, color="#555", va="center")
axL.annotate(
    "",
    xy=(L - 0.15, sch_base[L] - 0.008),
    xytext=(L - 2.8, sch_abl[L - 2] + 0.005),
    arrowprops=dict(arrowstyle="-|>", color=GREY, lw=1.3, connectionstyle="arc3,rad=-0.3"),
)
axL.text(L - 3.2, sch_abl[L - 2] - 0.042, "recovery", fontsize=10.5, color="#555", ha="center")
axL.set_title("one ablated layer, for illustration", fontsize=12.5, fontweight="bold", pad=10)
axL.set_ylabel("normalised ROC AUC decoded at each depth", fontsize=10.5)
axL.legend(loc="lower right", fontsize=10, frameon=False)

# ---- right: the real experiment, every layer ablated
cmap = plt.get_cmap("viridis")
norm = Normalize(0, n_layers - 1)
axR.plot(np.arange(L + 1), base, "-o", color=BLACK, lw=2.4, ms=4.0, zorder=4)
for m in range(n_layers):
    c = cmap(norm(m))
    d = np.arange(m + 1, L + 1)
    axR.plot(d, skips[m][m + 1 :], "-o", color=c, lw=1.5, ms=3.0, alpha=0.95, zorder=3)
    axR.plot([m, m + 1], [base[m], skips[m][m + 1]], "--", color=c, lw=1.1, alpha=0.8, zorder=2)
    axR.plot([m], [base[m]], "x", color=RED, ms=6, mew=1.4, alpha=0.85, zorder=5)
axR.set_title(
    f"{MODEL}, 15 tasks — every layer ablated in turn", fontsize=12.5, fontweight="bold", pad=10
)

cb = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=axR, pad=0.015, fraction=0.045)
cb.set_label("ablated layer", fontsize=10)

for ax in (axL, axR):
    ax.set_xlabel("layer (forward-pass order)", fontsize=10.5)
    ax.set_xlim(-0.5, L + 0.5)
    ax.set_ylim(0.5, 1.045)
    ax.set_xticks(range(0, L + 1, 2))
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.18, lw=0.7)

fig.subplots_adjust(left=0.075, right=0.965, top=0.90, bottom=0.125)
out = FIGDIR.joinpath("exp6-transition.png")
fig.savefig(out, dpi=170, facecolor="white", bbox_inches="tight")
print(f"wrote {out}")
