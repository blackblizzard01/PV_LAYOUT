"""
Render the PV site layout as a matplotlib figure.

Draws: site boundary rectangle, panel row footprints (to scale), DC cable
runs (block centroid → assigned inverter, thin orange), AC cable runs
(inverter → transformer, thick red), inverter markers (×), transformer
marker (filled square).

Uses matplotlib (PSF/BSD-compatible licence).
"""

from __future__ import annotations

import math

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

from .placement import PlacementResult


def render_layout(
    site_width: float,
    site_height: float,
    row_centroids: np.ndarray,
    module_length: float,
    module_width: float,
    tilt_deg: float,
    placement: PlacementResult,
    output_path: str | None = None,
    title: str = "PV Site Layout",
) -> plt.Figure:
    """
    Render the site layout and return the matplotlib Figure.

    Parameters
    ----------
    site_width, site_height : site rectangle dimensions, metres
    row_centroids           : (n_rows, 2) array from geometry.place_panels
    module_length           : tilted module length (along slope), metres
    module_width            : module east-west width, metres
    tilt_deg                : panel tilt from horizontal, degrees
    placement               : result of placement.place_equipment
    output_path             : if given, save the figure to this path (PNG)
    title                   : figure title

    Returns the Figure object (caller can display or further customise it).
    """
    fig, ax = plt.subplots(figsize=(13, 9))

    # ── Site boundary ──────────────────────────────────────────────────────
    ax.add_patch(mpatches.Rectangle(
        (0, 0), site_width, site_height,
        fill=False, edgecolor="black", linewidth=2,
    ))

    # ── Panel row footprints ───────────────────────────────────────────────
    footprint_ns = module_length * math.cos(math.radians(tilt_deg))
    for cx, cy in row_centroids:
        y_south = cy - footprint_ns / 2.0
        ax.add_patch(mpatches.Rectangle(
            (0, y_south), site_width, footprint_ns,
            facecolor="royalblue", edgecolor="steelblue",
            alpha=0.35, linewidth=0.3,
        ))

    # ── DC cable runs: block centroid → assigned inverter ──────────────────
    for i, (bx, by) in enumerate(placement.block_centroids):
        inv_idx = placement.block_to_inverter[i]
        ix, iy = placement.inverter_positions[inv_idx]
        ax.plot([bx, ix], [by, iy],
                color="darkorange", linewidth=0.9, alpha=0.65, zorder=3)

    # ── AC cable runs: inverter → transformer ──────────────────────────────
    tx = placement.transformer_positions[0]
    for ix, iy in placement.inverter_positions:
        ax.plot([ix, tx[0]], [iy, tx[1]],
                color="crimson", linewidth=2.0, alpha=0.85, zorder=4)

    # ── Inverter markers (×) ───────────────────────────────────────────────
    ax.scatter(
        placement.inverter_positions[:, 0],
        placement.inverter_positions[:, 1],
        marker="x", s=90, color="darkorange", linewidths=2.2, zorder=5,
    )

    # ── Transformer marker (■) ─────────────────────────────────────────────
    ax.scatter(
        [tx[0]], [tx[1]],
        marker="s", s=180, color="darkred", zorder=6,
    )

    # ── Legend ─────────────────────────────────────────────────────────────
    legend_handles = [
        mpatches.Patch(facecolor="royalblue", alpha=0.35, label="Panel rows"),
        Line2D([0], [0], color="darkorange", linewidth=1.2, label="DC cable"),
        Line2D([0], [0], color="crimson", linewidth=2.0, label="AC cable"),
        plt.scatter([], [], marker="x", s=70, color="darkorange",
                    linewidths=2, label=f"Inverters (n={placement.n_inverters})"),
        plt.scatter([], [], marker="s", s=100, color="darkred",
                    label="Transformer"),
        mpatches.Patch(fill=False, edgecolor="black", linewidth=2,
                       label="Site boundary"),
    ]
    ax.legend(handles=legend_handles, loc="upper right", fontsize=9,
              framealpha=0.9)

    ax.set_xlim(-8, site_width + 8)
    ax.set_ylim(-8, site_height + 8)
    ax.set_xlabel("East – West (m)", fontsize=11)
    ax.set_ylabel("South – North (m)", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.18)
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")

    return fig
