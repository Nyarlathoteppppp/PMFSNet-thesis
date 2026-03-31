# -*- coding: utf-8 -*-
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


OUTPUT_DIR = Path("/root/autodl-tmp/xbw/PMFSNet-master/runs/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def add_box(ax, x, y, w, h, text, fc, ec="#334155", fontsize=11, lw=1.8, rounding=0.03):
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.012,rounding_size={rounding}",
        linewidth=lw,
        edgecolor=ec,
        facecolor=fc,
    )
    ax.add_patch(box)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize, color="#0f172a")
    return box


def add_arrow(ax, start, end, color="#475569", lw=2.0, style="->", connectionstyle="arc3"):
    arr = FancyArrowPatch(
        start, end,
        arrowstyle=style,
        mutation_scale=14,
        linewidth=lw,
        color=color,
        connectionstyle=connectionstyle,
    )
    ax.add_patch(arr)
    return arr


def main():
    fig, ax = plt.subplots(figsize=(16, 9), dpi=220)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    ax.text(0.03, 0.95, "PMFSNetEdgeAux Architecture", fontsize=22, fontweight="bold", color="#0f172a")
    ax.text(
        0.03,
        0.91,
        "Global context is preserved by the original PMFS bottleneck, while local boundary detail is enhanced by a shallow edge auxiliary head.",
        fontsize=11.5,
        color="#475569",
    )

    colors = {
        "input": "#e2e8f0",
        "local": "#dbeafe",
        "skip": "#dcfce7",
        "global": "#ffedd5",
        "decoder": "#ede9fe",
        "output": "#fee2e2",
        "aux": "#fce7f3",
        "loss": "#fef3c7",
    }

    add_box(ax, 0.05, 0.54, 0.11, 0.12, "Input Image\n3 x 224 x 224", colors["input"], fontsize=12)

    add_box(ax, 0.21, 0.68, 0.13, 0.11, "Stage 1 Encoder\nLocal block", colors["local"])
    add_box(ax, 0.39, 0.68, 0.13, 0.11, "Stage 2 Encoder\nLocal block", colors["local"])
    add_box(ax, 0.57, 0.68, 0.13, 0.11, "Stage 3 Encoder\nLocal block", colors["local"])

    add_box(ax, 0.22, 0.48, 0.11, 0.08, "x1 skip", colors["skip"])
    add_box(ax, 0.40, 0.48, 0.11, 0.08, "x2 skip", colors["skip"])

    add_box(ax, 0.74, 0.68, 0.15, 0.11, "PMFS Block\n(Global context)", colors["global"], fontsize=11.5)

    add_box(ax, 0.70, 0.46, 0.12, 0.09, "Up2 + Fuse\nwith x2 skip", colors["decoder"])
    add_box(ax, 0.54, 0.30, 0.12, 0.09, "Up1 + Fuse\nwith x1 skip", colors["decoder"])
    add_box(ax, 0.72, 0.18, 0.14, 0.10, "Segmentation Head", colors["output"])
    add_box(ax, 0.88, 0.18, 0.10, 0.10, "Segmentation\nMask", colors["output"])

    add_box(ax, 0.17, 0.22, 0.16, 0.10, "Edge Auxiliary Head\n3x3 Conv + 1x1 Conv", colors["aux"])
    add_box(ax, 0.09, 0.06, 0.18, 0.10, "Boundary Logits", colors["aux"])
    add_box(ax, 0.31, 0.06, 0.18, 0.10, "Boundary Target\n(from GT mask)", colors["loss"])
    add_box(ax, 0.53, 0.06, 0.18, 0.10, "Edge Loss\nBCEWithLogits", colors["loss"])
    add_box(ax, 0.75, 0.06, 0.18, 0.10, "Total Loss\nL = L_seg + lambda L_edge", colors["loss"], fontsize=10.8)

    add_arrow(ax, (0.16, 0.60), (0.21, 0.735))
    add_arrow(ax, (0.34, 0.735), (0.39, 0.735))
    add_arrow(ax, (0.52, 0.735), (0.57, 0.735))
    add_arrow(ax, (0.70, 0.735), (0.74, 0.735))
    add_arrow(ax, (0.815, 0.68), (0.76, 0.55))
    add_arrow(ax, (0.70, 0.505), (0.66, 0.345))
    add_arrow(ax, (0.66, 0.345), (0.72, 0.23))
    add_arrow(ax, (0.86, 0.23), (0.88, 0.23))

    add_arrow(ax, (0.275, 0.68), (0.275, 0.56))
    add_arrow(ax, (0.455, 0.68), (0.455, 0.56))

    add_arrow(ax, (0.51, 0.52), (0.70, 0.52))
    add_arrow(ax, (0.33, 0.52), (0.54, 0.345))

    add_arrow(ax, (0.22, 0.48), (0.25, 0.32))
    add_arrow(ax, (0.22, 0.22), (0.18, 0.16))
    add_arrow(ax, (0.27, 0.11), (0.31, 0.11))
    add_arrow(ax, (0.49, 0.11), (0.53, 0.11))
    add_arrow(ax, (0.71, 0.11), (0.75, 0.11))

    add_box(ax, 0.03, 0.80, 0.12, 0.05, "Local encoder", colors["local"], fontsize=10, lw=1.2, rounding=0.02)
    add_box(ax, 0.03, 0.73, 0.12, 0.05, "Skip feature", colors["skip"], fontsize=10, lw=1.2, rounding=0.02)
    add_box(ax, 0.03, 0.66, 0.12, 0.05, "Global context", colors["global"], fontsize=10, lw=1.2, rounding=0.02)
    add_box(ax, 0.03, 0.59, 0.12, 0.05, "Boundary branch", colors["aux"], fontsize=10, lw=1.2, rounding=0.02)

    ax.text(0.56, 0.83, "Original PMFSNet trunk", fontsize=12, color="#1d4ed8", fontweight="bold")
    ax.text(0.18, 0.36, "Added shallow boundary supervision", fontsize=12, color="#be185d", fontweight="bold")

    png_path = OUTPUT_DIR / "PMFSNetEdgeAux_architecture.png"
    svg_path = OUTPUT_DIR / "PMFSNetEdgeAux_architecture.svg"
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {png_path}")
    print(f"Saved {svg_path}")


if __name__ == "__main__":
    main()
