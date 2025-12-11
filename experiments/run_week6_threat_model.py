# experiments/run_week6_threat_model.py

"""
Week-6: Threat model diagram for OrbiSec-FL-HE.

Generates a static figure showing:
- LEO satellites (FL clients)
- Ground station (FL server)
- Network adversary
- Three security modes: none, mask, ckks

Output:
- results/week6_threat_model.png
"""

import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch


def draw_box(ax, xy, width, height, label, color="#ddeeff"):
    rect = Rectangle(
        xy, width, height,
        linewidth=1.5,
        edgecolor="black",
        facecolor=color
    )
    ax.add_patch(rect)
    ax.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        label,
        ha="center",
        va="center",
        fontsize=10,
        weight="bold",
        )


def draw_arrow(ax, xy_from, xy_to, text=None):
    arrow = FancyArrowPatch(
        xy_from, xy_to,
        arrowstyle="->",
        linewidth=1.5,
        mutation_scale=15,
        color="black",
    )
    ax.add_patch(arrow)
    if text is not None:
        mid = ((xy_from[0] + xy_to[0]) / 2, (xy_from[1] + xy_to[1]) / 2)
        ax.text(
            mid[0],
            mid[1] + 0.1,
            text,
            ha="center",
            va="bottom",
            fontsize=8,
            )


def main():
    project_root = Path(__file__).resolve().parents[1]
    results_dir = project_root / "results"
    results_dir.mkdir(exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 5))

    # --- Layout coordinates (in arbitrary units) ---
    # Left: satellites stacked vertically
    sat_w, sat_h = 1.6, 0.6
    sat_x = 0.5
    sat_y_base = 1.0
    sat_delta = 0.9

    satellites = [
        (sat_x, sat_y_base + i * sat_delta)
        for i in range(3)
    ]

    for i, (x, y) in enumerate(satellites):
        draw_box(
            ax,
            (x, y),
            sat_w,
            sat_h,
            label=f"Satellite {i+1}\n(FL client)",
            color="#e3f2fd",
        )

    # Right: ground station (server)
    server_x, server_y = 6.0, 1.6
    server_w, server_h = 2.2, 1.6
    draw_box(
        ax,
        (server_x, server_y),
        server_w,
        server_h,
        label="Ground Station\n(FL server)",
        color="#e8f5e9",
    )

    # Middle: adversary on the link
    adv_x, adv_y = 3.6, 1.8
    adv_w, adv_h = 1.4, 1.0
    draw_box(
        ax,
        (adv_x, adv_y),
        adv_w,
        adv_h,
        label="Network\nAdversary",
        color="#ffebee",
    )

    # Arrows from satellites -> adversary -> server
    for x, y in satellites:
        draw_arrow(
            ax,
            (x + sat_w, y + sat_h / 2),
            (adv_x, adv_y + adv_h / 2),
        )
    draw_arrow(
        ax,
        (adv_x + adv_w, adv_y + adv_h / 2),
        (server_x, server_y + server_h / 2),
    )

    # Text panel at bottom: what adversary sees in each mode
    text_y = 0.1
    ax.text(
        4.5,
        text_y,
        (
            "Security modes (what the adversary observes on the channel):\n"
            "  • none  : raw model updates (gradients/weights) → high leakage (DLG works).\n"
            "  • mask  : masked updates = update + Gaussian noise → reduced leakage.\n"
            "  • ckks  : CKKS ciphertexts → only encrypted aggregates, no raw updates."
        ),
        ha="center",
        va="bottom",
        fontsize=9,
    )

    ax.set_xlim(0, 9)
    ax.set_ylim(0, 4.2)
    ax.axis("off")
    ax.set_title("Week-6 Threat Model for OrbiSec-FL-HE", fontsize=13, weight="bold")

    out_path = results_dir / "week6_threat_model.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)

    print(f"[Week-6] Threat model diagram saved to: {out_path}")


if __name__ == "__main__":
    main()
