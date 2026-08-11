"""The three figures for section 5.3 — zero versus resample ablation.

All three are regenerated from the current local data so that the numbers on
the figures agree with the table in the text:

- ablation-norm.png       TabFM, residual-norm ratio per depth      (magnitude)
- ablation-stability.png  spread and largest negative hit, 4 models (stability)
- ablation-pair.png       Mitra, trajectories under both ablations  (cross-check)

Sources: tfmlens out/rr_v1_all/v1_stats.json (norm and direction diagnostics)
and out/v2_{model}_{zero,resample}.json (Exp6 sweep under each ablation).
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
OUT = REPO / "out"
RED, BLUE, BLACK = "#d1462f", "#3b6ea5", "#1a1a1a"
MODELS = ["limix_2m", "mitra", "tabicl_v2", "tabfm"]
LABEL = {"limix_2m": "LimiX-2M", "mitra": "Mitra", "tabicl_v2": "TabICL", "tabfm": "TabFM"}


def norm_margin(sweep):
    """baseline and per-ablated-layer margin trajectories, as a fraction of the clean final."""
    base = np.array(sweep["baseline"]["margin"], float)
    n = abs(base[-1]) or 1.0
    skip = np.array([sweep["skip"][str(k)]["margin"] for k in range(len(sweep["skip"]))], float)
    return base / n, skip / n


def immediate_hits(model, mode):
    """imm(m) = baseline[m+1] - ablated_m[m+1], averaged over tasks."""
    d = json.loads((OUT / f"v2_{model}_{mode}.json").read_text())
    hits = []
    for r in d.values():
        base, skip = norm_margin(r["sweep"])
        hits.append([base[m + 1] - skip[m][m + 1] for m in range(len(skip))])
    return np.mean(hits, axis=0)


# ---------------------------------------------------------------- 1. magnitude
stats = json.loads((OUT / "rr_v1_all/v1_stats.json").read_text())["tabfm"]
rz, rr = np.array(stats["ratio_zero"]), np.array(stats["ratio_resample"])
depth = np.linspace(0, 1, len(rz))

fig, ax = plt.subplots(figsize=(7.4, 4.3))
ax.axhline(1.0, color="#999", lw=1, ls=":")
ax.plot(depth, rz, "-o", color=RED, lw=2, ms=4, label="zero ablation")
ax.plot(depth, rr, "-o", color=BLUE, lw=2, ms=4, label="resample ablation")
ax.set_xlabel("relative depth (0 = first layer, 1 = last)", fontsize=10.5)
ax.set_ylabel(
    "residual norm after ablation,\nas a fraction of the clean norm", fontsize=10.5, linespacing=1.4
)
ax.set_title(
    "TabFM: resample keeps the residual norm, zero drops it", fontsize=12, fontweight="bold", pad=10
)
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="y", alpha=0.18, lw=0.7)
ax.legend(fontsize=10, frameon=False, loc="lower right")
fig.tight_layout()
fig.savefig(FIGDIR / "ablation-norm.png", dpi=170, facecolor="white")
print("wrote ablation-norm.png")

# --------------------------------------------------------------- 2. stability
spread = {m: (immediate_hits(m, "zero").std(), immediate_hits(m, "resample").std()) for m in MODELS}
worst = {
    m: (max(0.0, -immediate_hits(m, "zero").min()), max(0.0, -immediate_hits(m, "resample").min()))
    for m in MODELS
}

fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.2))
x = np.arange(len(MODELS))
for ax, data, title, ylab in [
    (axes[0], spread, "standard deviation of the immediate drop, across layers", "margin"),
    (axes[1], worst, "largest negative drop", "margin"),
]:
    z = [data[m][0] for m in MODELS]
    r = [data[m][1] for m in MODELS]
    ax.bar(x - 0.19, z, 0.36, color=RED, label="zero")
    ax.bar(x + 0.19, r, 0.36, color=BLUE, label="resample")
    for xi, (a, b) in enumerate(zip(z, r, strict=True)):
        ax.text(xi - 0.19, a, f"{a:.2f}", ha="center", va="bottom", fontsize=9, color=RED)
        ax.text(xi + 0.19, b, f"{b:.2f}", ha="center", va="bottom", fontsize=9, color=BLUE)
    ax.set_xticks(x, [LABEL[m] for m in MODELS], fontsize=10)
    ax.set_ylabel(ylab, fontsize=10)
    ax.set_title(title, fontsize=11.5, pad=8)
    ax.set_ylim(0, max(z + r) * 1.18)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.18, lw=0.7)
axes[0].legend(fontsize=10, frameon=False)
fig.suptitle(
    "Resample ablation perturbs consistently; zero ablation does not",
    fontsize=13,
    fontweight="bold",
    y=0.99,
)
fig.tight_layout(rect=(0, 0, 1, 0.94))
fig.savefig(FIGDIR / "ablation-stability.png", dpi=170, facecolor="white")
print("wrote ablation-stability.png")

# -------------------------------------------------------------- 3. cross-check
MODEL = "mitra"
fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.6), sharey=True)
cmap = plt.get_cmap("viridis")
for ax, mode, title in [
    (axes[0], "zero", "zero ablation"),
    (axes[1], "resample", "resample ablation"),
]:
    d = json.loads((OUT / f"v2_{MODEL}_{mode}.json").read_text())
    B, S = [], []
    for r in d.values():
        b, s = norm_margin(r["sweep"])
        B.append(b)
        S.append(s)
    base, skip = np.mean(B, 0), np.mean(S, 0)
    L = len(base) - 1
    nrm = Normalize(0, skip.shape[0] - 1)
    ax.plot(np.arange(L + 1), base, "-o", color=BLACK, lw=2.4, ms=4, zorder=4, label="no ablation")
    for m in range(skip.shape[0]):
        c = cmap(nrm(m))
        ax.plot(np.arange(m + 1, L + 1), skip[m][m + 1 :], "-o", color=c, lw=1.4, ms=3, zorder=3)
        ax.plot([m, m + 1], [base[m], skip[m][m + 1]], "--", color=c, lw=1.1, zorder=2)
        ax.plot([m], [base[m]], "x", color=RED, ms=7, mew=1.6, zorder=5)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=9)
    ax.set_xlabel("layer (forward-pass order)", fontsize=10.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.18, lw=0.7)
axes[0].set_ylabel("margin, as a fraction of the clean final", fontsize=10.5)
axes[0].legend(fontsize=10, frameon=False, loc="lower right")
fig.colorbar(
    ScalarMappable(norm=Normalize(0, 11), cmap=cmap), ax=axes[1], pad=0.015, fraction=0.045
).set_label("ablated layer", fontsize=10)
fig.suptitle(
    f"{LABEL[MODEL]}: the dip and recovery survives the change of ablation",
    fontsize=13,
    fontweight="bold",
    y=0.99,
)
fig.tight_layout(rect=(0, 0, 1, 0.94))
fig.savefig(FIGDIR / "ablation-pair.png", dpi=170, facecolor="white")
print("wrote ablation-pair.png")

# ------------------------------------------------------------------ the table
print("\nmodel      min norm ratio z/r   std(imm) z/r     largest negative hit z/r")
for m in MODELS:
    v = json.loads((OUT / "rr_v1_all/v1_stats.json").read_text())[m]
    print(
        f"{LABEL[m]:9s} {min(v['ratio_zero']):.2f} / {min(v['ratio_resample']):.2f}"
        f"        {spread[m][0]:.2f} / {spread[m][1]:.2f}"
        f"      {worst[m][0]:.2f} / {worst[m][1]:.2f}"
    )


# ------------------------------------------- 4. the other models, for the appendix
REST = ["limix_2m", "tabicl_v2", "tabfm"]
fig, axes = plt.subplots(len(REST), 2, figsize=(11.6, 3.9 * len(REST)))
for row, model in enumerate(REST):
    for col, mode in enumerate(["zero", "resample"]):
        ax = axes[row, col]
        d = json.loads((OUT / f"v2_{model}_{mode}.json").read_text())
        B, S = [], []
        for r in d.values():
            b, sk = norm_margin(r["sweep"])
            B.append(b)
            S.append(sk)
        base, skip = np.mean(B, 0), np.mean(S, 0)
        L = len(base) - 1
        nrm = Normalize(0, skip.shape[0] - 1)
        ax.plot(np.arange(L + 1), base, "-o", color=BLACK, lw=2.2, ms=3.4, zorder=4)
        for m in range(skip.shape[0]):
            c = cmap(nrm(m))
            ax.plot(
                np.arange(m + 1, L + 1), skip[m][m + 1 :], "-o", color=c, lw=1.2, ms=2.6, zorder=3
            )
            ax.plot([m, m + 1], [base[m], skip[m][m + 1]], "--", color=c, lw=1.0, zorder=2)
            ax.plot([m], [base[m]], "x", color=RED, ms=6, mew=1.4, zorder=5)
        ax.set_title(f"{LABEL[model]} — {mode} ablation", fontsize=11.5, fontweight="bold", pad=8)
        ax.set_xlabel("layer (forward-pass order)", fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.18, lw=0.7)
    lo = min(a.get_ylim()[0] for a in axes[row])
    hi = max(a.get_ylim()[1] for a in axes[row])
    for a in axes[row]:
        a.set_ylim(lo, hi)
    axes[row, 0].set_ylabel(
        "margin, as a fraction\nof the clean final", fontsize=10, linespacing=1.4
    )
fig.tight_layout()
fig.savefig(FIGDIR / "ablation-pair-rest.png", dpi=150, facecolor="white")
print("wrote ablation-pair-rest.png")
