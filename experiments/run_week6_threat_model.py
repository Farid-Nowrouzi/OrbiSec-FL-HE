# experiments/run_week6_threat_model.py

"""
Week-6: Threat model diagram for OrbiSec-FL-HE .

Generates a static figure showing:
- LEO satellites (FL clients)
- Ground station (FL server / aggregator)
- Passive network adversary (eavesdropper)
- Three security modes: none, mask, ckks_like

Output:
- results/week6_threat_model.png
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


# -----------------------------
# Drawing helpers
# -----------------------------
def rounded_box(ax, x, y, w, h, text, facecolor, fontsize=11, bold=True):
    box = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.06",
        linewidth=1.6,
        edgecolor="black",
        facecolor=facecolor,
    )
    ax.add_patch(box)

    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        weight="bold" if bold else "normal",
        )
    return box


def arrow(ax, x1, y1, x2, y2, text=None, text_dy=0.10):
    a = FancyArrowPatch(
        (x1, y1),
        (x2, y2),
        arrowstyle="->",
        linewidth=1.8,
        mutation_scale=16,
        color="black",
    )
    ax.add_patch(a)

    if text:
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(mx, my + text_dy, text, ha="center", va="bottom", fontsize=9)


def add_panel(ax, x, y, w, h, title, lines, facecolor="#f7f7f7"):
    panel = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.03,rounding_size=0.06",
        linewidth=1.2,
        edgecolor="black",
        facecolor=facecolor,
    )
    ax.add_patch(panel)

    ax.text(
        x + 0.03,
        y + h - 0.08,
        title,
        ha="left",
        va="top",
        fontsize=10,
        weight="bold",
        )

    yy = y + h - 0.18
    for line in lines:
        ax.text(
            x + 0.06,
            yy,
            f"• {line}",
            ha="left",
            va="top",
            fontsize=9,
            )
        yy -= 0.11

    return panel


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    results_dir = project_root / "results"
    results_dir.mkdir(exist_ok=True)

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")

    # Title
    ax.text(
        5,
        5.75,
        "Week-6 Threat Model for OrbiSec-FL-HE",
        ha="center",
        va="center",
        fontsize=16,
        weight="bold",
    )
    ax.text(
        5,
        5.47,
        "Federated Learning over LEO link with a passive network eavesdropper",
        ha="center",
        va="center",
        fontsize=10,
    )

    # --- Entities ---
    # Satellites (clients)
    sat_w, sat_h = 2.2, 0.8
    sat_x = 0.8
    sat_ys = [3.9, 2.7, 1.5]

    for i, y in enumerate(sat_ys, start=1):
        rounded_box(
            ax,
            sat_x,
            y,
            sat_w,
            sat_h,
            text=f"Satellite {i}\n(FL client)",
            facecolor="#e3f2fd",
            fontsize=11,
        )

    # Adversary
    adv_x, adv_y = 4.35, 2.45
    adv_w, adv_h = 1.85, 1.2
    rounded_box(
        ax,
        adv_x,
        adv_y,
        adv_w,
        adv_h,
        text="Network Adversary\n(passive eavesdropper)",
        facecolor="#ffebee",
        fontsize=10,
    )

    # Ground station
    srv_x, srv_y = 7.05, 2.15
    srv_w, srv_h = 2.5, 1.8
    rounded_box(
        ax,
        srv_x,
        srv_y,
        srv_w,
        srv_h,
        text="Ground Station\n(FL server)",
        facecolor="#e8f5e9",
        fontsize=11,
    )

    # --- Links ---
    # Clients -> adversary
    for y in sat_ys:
        arrow(
            ax,
            sat_x + sat_w,
            y + sat_h / 2,
            adv_x,
            adv_y + adv_h / 2,
            )

    # Adversary -> server
    arrow(
        ax,
        adv_x + adv_w,
        adv_y + adv_h / 2,
        srv_x,
        srv_y + srv_h / 2,
        text="uplink / downlink traffic",
        text_dy=0.12,
        )

    # --- Panels ---
    add_panel(
        ax,
        x=0.7,
        y=0.15,
        w=6.35,
        h=1.25,
        title="Security modes (what the adversary can observe on the channel)",
        lines=[
            "none: raw model updates (gradients/weights) -> highest leakage (DLG & inference attacks possible).",
            "mask: masked / perturbed updates (random mask or noise-like perturbation) -> reduced leakage.",
            "ckks_like: ciphertexts on the channel -> adversary cannot read raw updates (sees encrypted values only).",
        ],
        facecolor="#f7f7f7",
    )

    add_panel(
        ax,
        x=7.15,
        y=0.15,
        w=2.75,
        h=1.25,
        title="Attacker assumptions",
        lines=[
            "Passive eavesdropper (records messages).",
            "Sees timing + traffic volume metadata.",
            "Does not compromise clients/server.",
        ],
        facecolor="#f7f7f7",
    )

    out_path = results_dir / "week6_threat_model.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)

    print(f"[Week-6] Threat model diagram saved to: {out_path}")


if __name__ == "__main__":
    main()
