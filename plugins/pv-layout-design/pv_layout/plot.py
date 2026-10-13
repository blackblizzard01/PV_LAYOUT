"""
Render the PV site layout as a matplotlib figure.

Draws: site boundary (rectangle or shapely polygon), panel row footprints
colored by terrain status (blue=accepted, amber=earthworks, red=excluded),
DC cable runs (block centroid → assigned inverter, thin orange), AC cable
runs (inverter → transformer, thick red), inverter markers (×), transformer
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


# Row colour palette: (facecolor, edgecolor, alpha)
_ROW_ACCEPTED   = ("#2980b9", "#1a6fa0", 0.45)
_ROW_EARTHWORKS = ("#f39c12", "#d68910", 0.55)
_ROW_EXCLUDED   = ("#e74c3c", "#c0392b", 0.40)
_ROW_DEFAULT    = ("royalblue", "steelblue", 0.35)


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
    boundary=None,
    terrain=None,
) -> plt.Figure:
    """
    Render the site layout and return the matplotlib Figure.

    Parameters
    ----------
    site_width, site_height : site rectangle bounding-box dimensions, metres
    row_centroids           : (n_rows, 2) array — all rows (incl. excluded)
    module_length           : tilted module length (along slope), metres
    module_width            : module east-west width, metres
    tilt_deg                : panel tilt from horizontal, degrees
    placement               : result of placement.place_equipment
    output_path             : if given, save the figure to this path (PNG)
    title                   : figure title
    boundary                : optional shapely Polygon; its exterior outline is
                              drawn and row strips are clipped to it
    terrain                 : optional TerrainResult; rows are coloured by
                              status (accepted / earthworks / excluded)
    """
    fig, ax = plt.subplots(figsize=(14, 9))

    # ── Site boundary ──────────────────────────────────────────────────────
    if boundary is not None:
        xs, ys = boundary.exterior.xy
        ax.plot(xs, ys, color="black", linewidth=2, zorder=10)
    else:
        ax.add_patch(mpatches.Rectangle(
            (0, 0), site_width, site_height,
            fill=False, edgecolor="black", linewidth=2,
        ))

    # ── Panel row footprints ───────────────────────────────────────────────
    footprint_ns = module_length * math.cos(math.radians(tilt_deg))

    for i, (cx, cy) in enumerate(row_centroids):
        if terrain is not None:
            if not terrain.row_mask[i]:
                fc, ec, alpha = _ROW_EXCLUDED
            elif terrain.earthworks_mask[i]:
                fc, ec, alpha = _ROW_EARTHWORKS
            else:
                fc, ec, alpha = _ROW_ACCEPTED
        else:
            fc, ec, alpha = _ROW_DEFAULT

        y_south = cy - footprint_ns / 2.0

        if boundary is not None:
            _draw_clipped_row(ax, boundary, y_south, footprint_ns, fc, ec, alpha)
        else:
            ax.add_patch(mpatches.Rectangle(
                (0, y_south), site_width, footprint_ns,
                facecolor=fc, edgecolor=ec, alpha=alpha, linewidth=0.3,
            ))

    # ── DC cable runs: block centroid → assigned inverter ──────────────────
    for i, (bx, by) in enumerate(placement.block_centroids):
        inv_idx = placement.block_to_inverter[i]
        ix, iy = placement.inverter_positions[inv_idx]
        ax.plot([bx, ix], [by, iy],
                color="darkorange", linewidth=0.9, alpha=0.65, zorder=3)

    # ── AC cable runs: each inverter → its assigned transformer ───────────
    for inv_idx, (ix, iy) in enumerate(placement.inverter_positions):
        tx_idx = placement.inverter_to_transformer[inv_idx]
        tx = placement.transformer_positions[tx_idx]
        ax.plot([ix, tx[0]], [iy, tx[1]],
                color="crimson", linewidth=2.0, alpha=0.85, zorder=4)

    # ── Inverter markers (×) ───────────────────────────────────────────────
    ax.scatter(
        placement.inverter_positions[:, 0],
        placement.inverter_positions[:, 1],
        marker="x", s=90, color="darkorange", linewidths=2.2, zorder=5,
    )

    # ── Transformer markers (■) ────────────────────────────────────────────
    for tx in placement.transformer_positions:
        ax.scatter([tx[0]], [tx[1]], marker="s", s=180, color="darkred", zorder=6)

    # ── Legend ─────────────────────────────────────────────────────────────
    legend_handles = []

    if terrain is not None:
        legend_handles.append(mpatches.Patch(
            facecolor=_ROW_ACCEPTED[0], alpha=_ROW_ACCEPTED[2],
            label=f"Accepted rows ({terrain.n_kept})",
        ))
        if terrain.n_earthworks > 0:
            legend_handles.append(mpatches.Patch(
                facecolor=_ROW_EARTHWORKS[0], alpha=_ROW_EARTHWORKS[2],
                label=f"Earthworks rows ({terrain.n_earthworks})",
            ))
        if terrain.n_removed > 0:
            legend_handles.append(mpatches.Patch(
                facecolor=_ROW_EXCLUDED[0], alpha=_ROW_EXCLUDED[2],
                label=f"Excluded rows ({terrain.n_removed})",
            ))
    else:
        legend_handles.append(mpatches.Patch(
            facecolor=_ROW_DEFAULT[0], alpha=_ROW_DEFAULT[2], label="Panel rows",
        ))

    legend_handles += [
        Line2D([0], [0], color="darkorange", linewidth=1.2, label="DC cable"),
        Line2D([0], [0], color="crimson",    linewidth=2.0, label="AC cable"),
        plt.scatter([], [], marker="x", s=70, color="darkorange", linewidths=2,
                    label=f"Inverters (n={placement.n_inverters})"),
        plt.scatter([], [], marker="s", s=100, color="darkred",
                    label=f"Transformer(s) (n={placement.n_transformers})"),
        mpatches.Patch(fill=False, edgecolor="black", linewidth=2,
                       label="Site boundary" if boundary is None else "Site polygon"),
    ]

    ax.legend(handles=legend_handles, loc="upper right", fontsize=9, framealpha=0.92)

    # ── Terrain stats annotation ───────────────────────────────────────────
    if terrain is not None:
        stats = (
            f"Terrain:  {terrain.n_kept} accepted  ·  "
            f"{terrain.n_earthworks} earthworks  ·  "
            f"{terrain.n_removed} excluded  ·  "
            f"Cut/fill: {terrain.cut_fill_volume_m3:.0f} m³"
        )
        ax.annotate(stats, xy=(0.01, 0.01), xycoords="axes fraction",
                    fontsize=8.5, color="dimgray",
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.75))

    # ── Axis limits ────────────────────────────────────────────────────────
    pad = 8
    if boundary is not None:
        bds = boundary.bounds
        ax.set_xlim(bds[0] - pad, bds[2] + pad)
        ax.set_ylim(bds[1] - pad, bds[3] + pad)
    else:
        ax.set_xlim(-pad, site_width + pad)
        ax.set_ylim(-pad, site_height + pad)

    ax.set_xlabel("East – West (m)", fontsize=11)
    ax.set_ylabel("South – North (m)", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.18)
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")

    return fig


def _draw_clipped_row(ax, boundary, y_south, height, fc, ec, alpha):
    """Clip a full-width row strip to the site polygon and draw it."""
    try:
        from shapely.geometry import box as shapely_box, MultiPolygon
        bds = boundary.bounds
        strip = shapely_box(bds[0], y_south, bds[2], y_south + height)
        clipped = strip.intersection(boundary)
        if clipped.is_empty:
            return
        geoms = list(clipped.geoms) if isinstance(clipped, MultiPolygon) else [clipped]
        for geom in geoms:
            xs, ys = geom.exterior.xy
            ax.fill(xs, ys, facecolor=fc, edgecolor=ec, alpha=alpha, linewidth=0.3)
    except Exception:
        bds = boundary.bounds
        ax.add_patch(mpatches.Rectangle(
            (bds[0], y_south), bds[2] - bds[0], height,
            facecolor=fc, edgecolor=ec, alpha=alpha, linewidth=0.3,
        ))
