"""Static PNG renderers (matplotlib): site plan, slope map, energy/loss KPIs."""

from __future__ import annotations

import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.collections import PolyCollection  # noqa: E402
from matplotlib.colors import BoundaryNorm, ListedColormap  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch, Polygon as MplPolygon  # noqa: E402

from ..terrain import EARTHWORKS, EXCLUDED  # noqa: E402

COLORS = {
    "available": "#ecebe4", "available_edge": "#4d4d4d",
    "restricted": "#f5b7b1", "restricted_edge": "#c0392b",
    "road": "#b9b9b9", "usable_edge": "#7f8c8d",
    "accepted": "#2f6db5", "earthworks": "#f0a020", "excluded": "#d62728",
    "mv": "#c0392b", "lv": "#e67e22", "ps": "#6c1d1d", "inverter": "#e67e22",
    "combiner": "#34495e", "substation": "#111111",
}
BLOCK_TINTS = ["#d6eaf8", "#d5f5e3", "#fcf3cf", "#f5eef8", "#fdebd0", "#e8f8f5"]


def _poly_patches(ax, geom, **kw):
    for p in getattr(geom, "geoms", [geom]):
        if p.is_empty:
            continue
        ax.add_patch(MplPolygon(np.asarray(p.exterior.coords), closed=True, **kw))
        for hole in p.interiors:
            ax.add_patch(MplPolygon(np.asarray(hole.coords), closed=True,
                                    facecolor="white", edgecolor=kw.get("edgecolor", "none"),
                                    linewidth=kw.get("linewidth", 0.5), zorder=kw.get("zorder", 1)))


def _table_verts(centers, w, d):
    x, y = centers[:, 0], centers[:, 1]
    return np.stack([
        np.column_stack([x - w / 2, y - d / 2]), np.column_stack([x + w / 2, y - d / 2]),
        np.column_stack([x + w / 2, y + d / 2]), np.column_stack([x - w / 2, y + d / 2]),
    ], axis=1)


def _north_arrow_and_scale(ax, bounds):
    x0, y0, x1, y1 = bounds
    span = max(x1 - x0, y1 - y0)
    ax.annotate("N", xy=(0.97, 0.93), xytext=(0.97, 0.83), xycoords="axes fraction",
                ha="center", va="center", fontsize=13, fontweight="bold",
                arrowprops=dict(arrowstyle="-|>", lw=2, color="black"))
    nice = [10, 20, 50, 100, 200, 250, 500, 1000, 2000, 5000]
    L = max(n for n in nice if n <= span / 5) if span / 5 >= 10 else 10
    sx, sy = x0, y0 - 0.04 * span
    ax.plot([sx, sx + L], [sy, sy], color="black", lw=3, solid_capstyle="butt")
    ax.text(sx + L / 2, sy - 0.015 * span, f"{L} m", ha="center", va="top", fontsize=9)


def render_layout_png(result, path, dpi: int = 150) -> str:
    site, lay, elec = result.site, result.tables, result.elec
    fig = plt.figure(figsize=(17, 10))
    ax = fig.add_axes([0.03, 0.06, 0.70, 0.86])
    side = fig.add_axes([0.75, 0.06, 0.23, 0.86])
    side.axis("off")

    for name, p in site.available:
        _poly_patches(ax, p, facecolor=COLORS["available"], edgecolor=COLORS["available_edge"],
                      linewidth=1.6, zorder=1)
        rp = p.representative_point()
        ax.text(rp.x, rp.y, name, fontsize=8, color="#555", alpha=0.6, ha="center", zorder=2)
    roads = site.roads(result.config.site.road_spacing_m, result.config.site.road_width_m)
    if not roads.is_empty:
        _poly_patches(ax, roads, facecolor=COLORS["road"], edgecolor="none", zorder=2)
    for name, p in site.restricted:
        _poly_patches(ax, p, facecolor=COLORS["restricted"], edgecolor=COLORS["restricted_edge"],
                      hatch="///", linewidth=1.0, zorder=3)
        rp = p.representative_point()
        ax.text(rp.x, rp.y, name, fontsize=9, color=COLORS["restricted_edge"], ha="center",
                fontweight="bold", zorder=9)
    for p in result.usable.geoms:
        ax.plot(*p.exterior.xy, color=COLORS["usable_edge"], lw=0.8, ls="--", zorder=4)
        for h in p.interiors:
            ax.plot(*h.xy, color=COLORS["usable_edge"], lw=0.8, ls="--", zorder=4)

    # block tint (inverter areas)
    verts = _table_verts(lay.centers, lay.width, lay.depth)
    for b in range(elec.n_inverters):
        m = elec.table_block == b
        if not m.any():
            continue
        pts = verts[m].reshape(-1, 2)
        from shapely.geometry import MultiPoint
        hull = MultiPoint(pts).convex_hull.buffer(1.0)
        ax.add_patch(MplPolygon(np.asarray(hull.exterior.coords),
                                facecolor=BLOCK_TINTS[b % len(BLOCK_TINTS)],
                                edgecolor="#95a5a6", lw=0.5, zorder=4.5))

    # tables
    colors = np.where(result.status == EARTHWORKS, COLORS["earthworks"], COLORS["accepted"])
    ax.add_collection(PolyCollection(verts, facecolors=colors, edgecolors="none", zorder=5))
    ex = result.terrain_all.status == EXCLUDED
    if ex.any():
        ev = _table_verts(result.tables_all.centers[ex], lay.width, lay.depth)
        ax.add_collection(PolyCollection(ev, facecolors=COLORS["excluded"], alpha=0.55,
                                         edgecolors="none", zorder=5))

    # cables
    for r in elec.routes_of("dc_main") + elec.routes_of("lv_ac"):
        ax.plot(r.points[:, 0], r.points[:, 1], color=COLORS["lv"], lw=0.9, alpha=0.8, zorder=6)
    for r in elec.routes_of("mv"):
        ax.plot(r.points[:, 0], r.points[:, 1], color=COLORS["mv"], lw=2.0, zorder=7)

    # equipment
    if len(elec.combiner_xy):
        ax.scatter(*elec.combiner_xy.T, marker="s", s=10, color=COLORS["combiner"], zorder=8)
    if elec.spec.inverter_type == "string":
        ax.scatter(*elec.inverter_xy.T, marker="^", s=28, color=COLORS["inverter"],
                   edgecolor="black", linewidth=0.4, zorder=8)
    for i, poly in enumerate(elec.ps_polys):
        ax.add_patch(MplPolygon(np.asarray(poly.exterior.coords), facecolor=COLORS["ps"],
                                edgecolor="black", lw=0.6, zorder=9))
        c = poly.centroid
        ax.text(c.x, c.y + 4, f"PS{i + 1}", fontsize=7, ha="center", va="bottom",
                color=COLORS["ps"], fontweight="bold", zorder=10)
    sx, sy = elec.substation_xy
    ax.scatter([sx], [sy], marker="D", s=120, color=COLORS["substation"], zorder=10)
    ax.text(sx, sy + 6, "Substation", fontsize=8, ha="center", fontweight="bold", zorder=10)

    b = site.bounds
    pad = 0.06 * max(b[2] - b[0], b[3] - b[1])
    ax.set_xlim(b[0] - pad, b[2] + pad)
    ax.set_ylim(b[1] - 1.6 * pad, b[3] + pad)
    ax.set_aspect("equal")
    ax.set_xticks([]), ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color("#999")
    _north_arrow_and_scale(ax, b)

    k = result.kpis
    ax.set_title(f"{site.name} — {k['dc_kwp'] / 1000:.2f} MWp DC / {k['ac_kw'] / 1000:.2f} MW AC",
                 fontsize=15, fontweight="bold", loc="left")

    handles = [
        Patch(facecolor=COLORS["available"], edgecolor=COLORS["available_edge"], label="Available area"),
        Patch(facecolor=COLORS["restricted"], edgecolor=COLORS["restricted_edge"], hatch="///",
              label="Restricted area"),
        Line2D([], [], color=COLORS["usable_edge"], ls="--", label="Setback line"),
        Patch(color=COLORS["accepted"], label=f"Structures ({int((result.status != EARTHWORKS).sum())})"),
        Patch(color=COLORS["earthworks"], label=f"With earthworks ({k['tables_earthworks']})"),
        Patch(color=COLORS["excluded"], alpha=0.55, label=f"Excluded by slope ({k['tables_excluded_terrain']})"),
        Patch(facecolor=BLOCK_TINTS[0], edgecolor="#95a5a6", label=f"Inverter blocks ({elec.n_inverters})"),
        Patch(facecolor=COLORS["ps"], label=f"Power stations ({elec.n_power_stations})"),
        Line2D([], [], color=COLORS["mv"], lw=2, label="MV cable"),
        Line2D([], [], color=COLORS["lv"], lw=1, label="DC main / LV AC cable"),
    ]
    if not roads.is_empty:
        handles.insert(2, Patch(color=COLORS["road"], label="Internal roads"))
    if len(elec.combiner_xy):
        handles.append(Line2D([], [], marker="s", ls="", color=COLORS["combiner"], label="Combiner boxes"))
    if elec.spec.inverter_type == "string":
        handles.append(Line2D([], [], marker="^", ls="", color=COLORS["inverter"], label="String inverters"))
    handles.append(Line2D([], [], marker="D", ls="", color=COLORS["substation"], label="Substation"))
    side.legend(handles=handles, loc="upper left", fontsize=9, frameon=False, title="Layers",
                title_fontproperties={"weight": "bold"})

    rows = [
        ("Location", f"{k['lat']:.4f}, {k['lon']:.4f}"),
        ("Available / usable", f"{k['available_ha']:.1f} / {k['usable_ha']:.1f} ha"),
        ("Density", f"{k['mwp_per_ha']:.3f} MWp/ha"),
        ("Tilt / azimuth", f"{k['tilt_deg']:.1f}° / {k['azimuth_deg']:.0f}°"),
        ("Pitch / GCR", f"{k['pitch_m']:.2f} m / {k['gcr']:.3f}"),
        ("Table", f"{result.structure.modules_high}×{result.structure.modules_along} "
                  f"{result.structure.orientation[0].upper()}  ({k['table_w_m']:.1f}×{k['table_depth_m']:.1f} m)"),
        ("String", f"{k['modules_per_string']} modules"),
        ("Tables / modules", f"{k['n_tables']:,} / {k['n_modules']:,}"),
        ("DC/AC ratio", f"{k['dc_ac_ratio']:.3f}"),
        ("Inverter blocks", f"{k['inverter_block_kwp_min']:.0f}–{k['inverter_block_kwp_max']:.0f} kWp"),
    ]
    if "annual_mwh" in k:
        rows += [("Energy", f"{k['annual_mwh']:,.0f} MWh/yr"),
                 ("Specific yield", f"{k['specific_yield_kwh_kwp']:.0f} kWh/kWp"),
                 ("PR", f"{100 * k['pr_poa']:.1f} %")]
    y = 0.42
    side.text(0, y + 0.03, "Key figures", fontsize=11, fontweight="bold", transform=side.transAxes)
    for label, val in rows:
        side.text(0, y, label, fontsize=9, color="#555", transform=side.transAxes)
        side.text(1, y, val, fontsize=9, ha="right", transform=side.transAxes)
        y -= 0.034

    fig.savefig(path, dpi=dpi, facecolor="white")
    plt.close(fig)
    return str(path)


def render_slope_png(result, path, dpi: int = 150) -> str:
    """N-S and E-W slope maps, banded per Design_Methodology §5.2.2 (<5, 5–10, 10–15, >15 %)."""
    dem = result.dem
    ns, ew = dem.slope_grids_pct()
    xs, ys = dem.x_coords, dem.y_coords
    cmap = ListedColormap(["#27ae60", "#f39c12", "#e74c3c", "#8e44ad"])
    norm = BoundaryNorm([0, 5, 10, 15, 1e9], cmap.N)
    fig, axes = plt.subplots(1, 2, figsize=(16, 7.5), constrained_layout=True)
    lay = result.tables
    verts = _table_verts(lay.centers, lay.width, lay.depth)
    for ax, grid, title in [(axes[0], ns, "North-South slope"), (axes[1], ew, "East-West slope")]:
        ax.pcolormesh(xs, ys, np.abs(grid), cmap=cmap, norm=norm, shading="auto")
        cs = ax.contour(xs, ys, dem.elevations, levels=12, colors="black", linewidths=0.4, alpha=0.5)
        ax.clabel(cs, fontsize=6, fmt="%.0f m")
        for _, p in result.site.available:
            ax.plot(*p.exterior.xy, color="black", lw=1.4)
        for _, p in result.site.restricted:
            ax.plot(*p.exterior.xy, color="#c0392b", lw=1.2)
        ax.add_collection(PolyCollection(verts, facecolors="none", edgecolors="white",
                                         linewidths=0.3, alpha=0.7))
        b = result.site.bounds
        ax.set_xlim(b[0] - 20, b[2] + 20)
        ax.set_ylim(b[1] - 20, b[3] + 20)
        ax.set_aspect("equal")
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xticks([]), ax.set_yticks([])
    handles = [Patch(color=c, label=l) for c, l in
               zip(cmap.colors, ["< 5 %", "5 – 10 %", "10 – 15 %", "> 15 %"])]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=10, frameon=False,
               bbox_to_anchor=(0.5, -0.04))
    z = dem.elevations
    fig.suptitle(f"{result.site.name} — terrain slope (DEM {result.config.terrain.source}, "
                 f"elevation {z.min():.0f}–{z.max():.0f} m)", fontsize=14, fontweight="bold")
    fig.savefig(path, dpi=dpi, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    return str(path)


def loss_waterfall_steps(energy) -> list[tuple[str, float]]:
    """(label, kWh/kWp after this step) from POA irradiation down to final specific yield."""
    e = energy
    hz = float(e.horizon_loss_frac)
    open_poa = e.poa_annual_kwh_m2 / (1 - hz) if hz < 1 else e.poa_annual_kwh_m2
    steps = [("POA (open sky)", open_poa)] if hz > 0 else []
    steps.append(("POA irradiation" if hz == 0 else "Horizon", e.poa_annual_kwh_m2))
    chain = [
        ("IAM", e.iam_loss_frac), ("Row shading", e.shading_loss_frac),
        ("Temp. & low light", e.temperature_loss_frac), ("Soiling", e.soiling_loss_frac),
        ("Mismatch", e.mismatch_loss_frac), ("LID", e.lid_loss_frac),
        ("DC cables", e.dc_cable_loss_frac), ("Inverter", e.inverter_loss_frac),
        ("Clipping", e.clipping_loss_frac), ("AC/MV cables", e.ac_cable_loss_frac),
        ("Transformers", e.transformer_loss_frac), ("Availability", e.availability_loss_frac),
    ]
    v = e.poa_annual_kwh_m2
    for label, f in chain:
        v *= 1 - float(f)
        steps.append((label, v))
    return steps


def render_kpis_png(result, path, dpi: int = 150) -> str:
    e = result.energy
    steps = loss_waterfall_steps(e)
    fig, ax = plt.subplots(figsize=(14, 6.5), constrained_layout=True)
    labels = [s[0] for s in steps] + ["Specific yield"]
    prev = steps[0][1]
    ax.bar(0, prev, color="#2f6db5")
    ax.text(0, prev, f"{prev:.0f}", ha="center", va="bottom", fontsize=9)
    for i, (label, v) in enumerate(steps[1:], start=1):
        drop = prev - v
        ax.bar(i, -drop, bottom=prev, color="#e74c3c" if drop >= 0 else "#27ae60")
        ax.text(i, prev, f"−{100 * drop / prev:.1f}%" if prev else "", ha="center", va="bottom", fontsize=8)
        prev = v
    final = e.specific_yield_kwh_kwp
    ax.bar(len(steps), final, color="#27ae60")
    ax.text(len(steps), final, f"{final:.0f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylim(0.7 * min(final, steps[-1][1]), 1.06 * steps[0][1])
    ax.set_ylabel("kWh/kWp per year (axis truncated)")
    ax.set_title(f"{result.site.name} — energy loss chain  |  {e.annual_kwh / 1000:,.0f} MWh/yr, "
                 f"PR {100 * e.pr_poa:.1f} %, GHI {e.ghi_annual_kwh_m2:.0f} kWh/m²",
                 fontsize=13, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    fig.savefig(path, dpi=dpi, facecolor="white")
    plt.close(fig)
    return str(path)
