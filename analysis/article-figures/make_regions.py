"""The reading key for the DE-TE plane — which region each label names.

A schematic, not data. It fixes the five names used in the text and in the
region table: redundant, indirectly important, load-bearing, repaired,
amplified.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]
FIGDIR = REPO / "out/article-figures"
FIGDIR.mkdir(parents=True, exist_ok=True)
TOL = 0.12
LIM = 1.75
RED, BLUE, GREY, AMBER = "#d1462f", "#3b6ea5", "#8a8a8a", "#e0912a"

fig, ax = plt.subplots(figsize=(9.2, 7.0))

# the wedge below the diagonal, where repair lives
xs = np.linspace(TOL, LIM, 50)
ax.fill_between(xs, 0 * xs, xs, color="#e0912a", alpha=0.09, zorder=0)
ax.fill_between(xs, xs, LIM + 0 * xs, color="#3b6ea5", alpha=0.07, zorder=0)

ax.axvspan(-TOL, TOL, color="0.88", zorder=0)
ax.plot([-LIM, LIM], [-LIM, LIM], "--", color="0.35", lw=1.3, zorder=2)
ax.axhline(0, color="0.75", lw=0.8, zorder=1)
ax.axvline(0, color="0.75", lw=0.8, zorder=1)
ax.text(
    0.62,
    0.66,
    "TE = DE",
    fontsize=12.5,
    color="0.35",
    ha="center",
    va="bottom",
    rotation=45,
    rotation_mode="anchor",
)


def region(x, y, name, gloss, color, dx, dy, ha, va):
    ax.plot(x, y, "o", ms=11, color=color, zorder=4)
    ax.annotate(
        f"{name}\n{gloss}",
        xy=(x, y),
        xytext=(dx, dy),
        textcoords="offset points",
        ha=ha,
        va=va,
        fontsize=12.5,
        color=color,
        linespacing=1.35,
        fontweight="bold",
        zorder=5,
    )


region(0.0, 0.0, "redundant", "writes nothing,\nnothing is lost", GREY, -16, -12, "right", "top")
region(
    0.0,
    1.05,
    "indirectly important",
    "writes no decision,\nbut later layers need it",
    GREY,
    -14,
    0,
    "right",
    "center",
)
region(
    1.30,
    1.30,
    "load-bearing",
    "writes the decision,\nnobody compensates",
    "#1a1a1a",
    10,
    4,
    "left",
    "bottom",
)
region(
    1.30,
    0.22,
    "repaired",
    "writes the decision,\ndownstream absorbs the loss",
    AMBER,
    0,
    -16,
    "center",
    "top",
)
region(0.55, 1.42, "amplified", "downstream makes\nthe loss worse", BLUE, -12, 2, "right", "bottom")

# the gap that is the compensation effect
ax.annotate(
    "", xy=(1.30, 0.27), xytext=(1.30, 1.25), arrowprops=dict(arrowstyle="<->", color=AMBER, lw=1.6)
)
ax.text(1.37, 0.76, r"$CE = DE - TE$", fontsize=13, color=AMBER, va="center", fontweight="bold")

ax.set_xlim(-0.62, LIM + 0.55)
ax.set_ylim(-0.62, LIM)
ax.set_aspect("equal")
ax.set_xlabel("DE   (direct effect)", fontsize=13)
ax.set_ylabel("TE   (total effect)", fontsize=13)
ax.set_xticks([0])
ax.set_yticks([0])
ax.spines[["top", "right"]].set_visible(False)
ax.set_title(
    "The five regions of the direct-versus-total-effect plane",
    fontsize=15,
    fontweight="bold",
    pad=12,
)
fig.tight_layout()
out = FIGDIR.joinpath("de-te-regions.png")
fig.savefig(out, dpi=170, facecolor="white")
print(f"wrote {out}")
