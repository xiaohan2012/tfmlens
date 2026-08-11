"""Figure 11 — the direct effect against the total effect, 2x2.

Same content as tfmlens scripts/plot_de_te_scatter.py, laid out two by two so the
panels are large enough to read. One point is one (layer, task) pair; the dashed
diagonal is TE = DE, i.e. no downstream reaction; the grey stripe is the band in
which the direct effect is treated as zero.
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
TOL = 0.1
MODELS = ["limix_2m", "mitra", "tabicl_v2", "tabfm"]
IN = REPO / "out"


def points(results):
    de, te, layer = [], [], []
    for r in results.values():
        eff = r["effects"]
        scale = de_scale(eff, COORD, agg=True)
        for m, d in eff["de"].items():
            de.append(d[COORD] / scale)
            te.append(eff["te"][m][COORD] / scale)
            layer.append(int(m))
    layer = np.array(layer, float)
    return np.array(de), np.array(te), layer / max(layer.max(), 1)


loaded = load_de_json(IN, MODELS)
fig, axes = plt.subplots(2, 2, figsize=(10.4, 10.0), constrained_layout=True)
sc = None
for ax, (model, results) in zip(axes.ravel(), loaded.items(), strict=True):
    de, te, layer = points(results)
    lim = float(np.nanmax(np.abs(np.concatenate([de, te])))) * 1.1 + 1e-6
    ax.plot([-lim, lim], [-lim, lim], "--", color="0.4", lw=1.1, zorder=1)
    ax.axvspan(-TOL, TOL, color="0.88", zorder=0)
    sc = ax.scatter(de, te, c=layer, cmap="viridis", vmin=0, vmax=1, s=26, alpha=0.75, zorder=2)
    ax.axhline(0, color="0.75", lw=0.7, zorder=1)
    ax.axvline(0, color="0.75", lw=0.7, zorder=1)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_title(MODEL_LABELS.get(model, model), fontsize=13, fontweight="bold", pad=8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.text(
        0.03, 0.965, "repair lives here", transform=ax.transAxes, fontsize=9, color="0.45", va="top"
    ) if False else None

cb = fig.colorbar(sc, ax=axes, shrink=0.55, pad=0.02)
cb.set_label("relative depth of the ablated layer  (0 = first, 1 = last)", fontsize=11)
fig.supxlabel("DE  (direct effect, path patching)", fontsize=12)
fig.supylabel("TE  (total effect, ablate and react)", fontsize=12)
fig.suptitle(
    "Direct effect against total effect, margin coordinate", fontsize=14, fontweight="bold"
)

out = FIGDIR.joinpath("de-te-scatter.png")
fig.savefig(out, dpi=170, facecolor="white")
print(f"wrote {out}")
