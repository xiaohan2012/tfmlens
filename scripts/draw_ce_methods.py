"""Draw the two CE-computation methods in the Pearl/Hydra causal-graph style
(⊕ = residual add, red = changed by ablation). For the notes, not an experiment.

Residual stream:  x → (⊕ with branch a) → (⊕ with branch b) → y
  a = the ablated layer m ;  b = a downstream layer ;  y = final logit (readout f).

    uv run --group viz python scripts/draw_ce_methods.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

RED = "#e8563f"
BLACK = "#1a1a1a"

# node positions (shared layout): spine at x=0.9, branch column at x=0.0
POS = {
    "x": (0.90, 0.20),
    "a": (0.00, 1.05),
    "p1": (0.90, 1.50),
    "b": (0.00, 2.55),
    "p2": (0.90, 3.00),
    "y": (0.90, 3.85),
}


def _arrow(ax, p, q, color, rad=0.0, lw=1.6):
    ax.add_patch(
        FancyArrowPatch(
            POS[p],
            POS[q],
            connectionstyle=f"arc3,rad={rad}",
            arrowstyle="-|>",
            mutation_scale=11,
            lw=lw,
            color=color,
            zorder=2,
        )
    )


# where each node's letter label sits, relative to the node (dx, dy, ha, va)
_LAB = {
    "x": (0.18, 0.0, "left", "center"),
    "y": (0.18, 0.0, "left", "center"),
    "a": (0.0, 0.30, "center", "bottom"),  # above (branch column, left side)
    "b": (0.0, 0.30, "center", "bottom"),
}


def _node(ax, key, color, filled=True, label=None, lab_color=None):
    x, y = POS[key]
    if key in ("p1", "p2"):  # the ⊕ residual-add node
        ax.scatter([x], [y], s=230, facecolor="white", edgecolor=BLACK, lw=1.4, zorder=3)
        ax.text(x, y, "+", ha="center", va="center", fontsize=13, zorder=4)
    else:
        ax.scatter(
            [x],
            [y],
            s=150,
            facecolor=(color if filled else "white"),
            edgecolor=color,
            lw=1.8,
            zorder=3,
        )
    if label:
        dx, dy, ha, va = _LAB[key]
        ax.text(
            x + dx, y + dy, label, ha=ha, va=va, fontsize=12, color=lab_color or color, zorder=4
        )


def _skeleton(ax, red=(), read_y=False, read_edge=False, do_b=None):
    """Draw the residual graph. red = set of nodes shown changed.
    read_y: highlight the output read (method A). read_edge: highlight b→y (method B)."""

    def col(n):
        return RED if n in red else BLACK

    # branch edges (curved) + skip edges (straight)
    _arrow(ax, "x", "a", col("a"), rad=0.45)
    _arrow(ax, "a", "p1", col("a"), rad=0.45)
    _arrow(ax, "x", "p1", col("x") if "x" in red else BLACK)  # skip x→⊕
    _arrow(ax, "p1", "b", col("b"), rad=0.45)
    _arrow(
        ax,
        "b",
        "p2",
        RED if read_edge or "b" in red else BLACK,
        rad=0.45,
        lw=3.0 if read_edge else 1.6,
    )  # b→⊕ (method B reads here)
    _arrow(ax, "p1", "p2", col("a") if "a" in red else BLACK)  # skip ⊕→⊕
    _arrow(ax, "p2", "y", RED if red else BLACK)

    _node(ax, "x", BLACK, label="x")
    _node(ax, "a", col("a"), label=("ã" if "a" in red else "a"))
    _node(ax, "p1", BLACK)
    _node(ax, "b", col("b"), label=("b'" if "b" in red else "b"))
    _node(ax, "p2", BLACK)
    _node(ax, "y", RED if red else BLACK, label="y")

    if do_b:
        bx, by = POS["b"]
        ax.annotate(
            do_b,
            (bx, by),
            xytext=(bx - 1.55, by - 0.05),
            fontsize=10,
            va="center",
            color=BLACK,
            arrowprops=dict(arrowstyle="-|>", color=BLACK, lw=1.2),
        )
    # ablation label on a
    ax.annotate(
        "do(m=ã)",
        POS["a"],
        xytext=(POS["a"][0] - 1.55, POS["a"][1] - 0.05),
        fontsize=10,
        color=RED,
        va="center",
        arrowprops=dict(arrowstyle="-|>", color=RED, lw=1.2),
    )

    if read_y:
        yx, yy = POS["y"]
        ax.scatter([yx], [yy], s=520, facecolor="none", edgecolor="tab:blue", lw=2.2, zorder=5)
        ax.text(yx + 0.30, yy + 0.02, "read here", color="tab:blue", fontsize=10, va="center")
    if read_edge:
        ex, ey = 0.34, POS["b"][1] + 0.42  # on the b→⊕ curve, well below ⊕
        ax.scatter([ex], [ey], s=430, facecolor="none", edgecolor="tab:blue", lw=2.2, zorder=5)
        ax.annotate(
            "read b's\nwrite here",
            (ex, ey),
            xytext=(ex - 1.9, ey),
            fontsize=9.5,
            color="tab:blue",
            va="center",
            arrowprops=dict(arrowstyle="-|>", color="tab:blue", lw=1.2),
        )

    ax.set_xlim(-1.75, 1.55)
    ax.set_ylim(-0.15, 4.30)
    ax.axis("off")
    ax.set_aspect("equal")


def main():
    fig, axes = plt.subplots(1, 4, figsize=(16, 5.2))

    # Panel 1 — clean
    _skeleton(axes[0])
    axes[0].set_title("① Clean\ny = x + a + b", fontsize=11)

    # Panel 2 — Method A, run 1: freeze b at clean (DE side)
    _skeleton(axes[1], red={"a"}, read_y=True, do_b="do(b = clean)")
    axes[1].set_title("② Method A · run FROZEN\nablate m, hold b at clean → read y", fontsize=11)

    # Panel 3 — Method A, run 2: b reacts (TE side)
    _skeleton(axes[2], red={"a", "b"}, read_y=True)
    axes[2].set_title(
        "③ Method A · run FREE\nablate m, b reacts (b') → read y\nCE_A = y[FREE] − y[FROZEN]",
        fontsize=11,
    )

    # Panel 4 — Method B: read b's own write, clean vs reacted
    _skeleton(axes[3], red={"a", "b"}, read_edge=True)
    axes[3].set_title(
        "④ Method B · independent\nablate m, b reacts (b')\n"
        "CE_B = (b' 's write to y) − (b's write to y)",
        fontsize=11,
    )

    fig.suptitle(
        "Two ways to compute CE —— A: read output y (two runs, subtract)   "
        "vs   B: read b's own write (one run, per-layer)",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = Path("out/ce_methods_diagram.png")
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=140)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
