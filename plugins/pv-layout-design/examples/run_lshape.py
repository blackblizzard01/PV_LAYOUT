"""
L-shape site example with terrain analysis — Ahmedabad 23N.

Site: 200 m × 150 m minus a 100 m × 90 m notch from the top-right corner.
Polygon vertices (metres): (0,0)→(200,0)→(200,60)→(100,60)→(100,150)→(0,150)
Setback: 5 m from every edge.
Terrain: 12 % N-S slope, 3 % E-W slope, 0.8 m undulation.

Run from repo root:
    python plugins/pv-layout-design/examples/run_lshape.py
"""

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from shapely.geometry import Polygon
from pv_layout import (
    apply_setback, place_panels_polygon,
    ModuleSpec,
    place_equipment, InverterSpec, TransformerSpec,
    compute_all_losses, CableSpec,
    render_layout,
    recommend_tilt,
    synthetic_dem, compute_slopes, apply_terrain_limits,
    bill_of_quantities,
)

# ── Site ──────────────────────────────────────────────────────────────────────
LAT, LON = 23.0, 72.6
boundary_raw = Polygon([
    (0, 0), (200, 0), (200, 60), (100, 60), (100, 150), (0, 150),
])
boundary = apply_setback(boundary_raw, setback_m=5.0)

# ── Specs ─────────────────────────────────────────────────────────────────────
tilt_deg  = recommend_tilt(LAT)
module    = ModuleSpec(length=2.0, width=1.0, power_wp=400.0)
inverter  = InverterSpec(rated_kw=100.0)
xfmr      = TransformerSpec(rated_kva=500.0, p_no_load_kw=0.5, p_load_rated_kw=5.0)
dc_cable  = CableSpec(material="copper", area_mm2=35.0)
ac_cable  = CableSpec(material="copper", area_mm2=35.0)

DC_V, AC_V, AC_PF = 1000.0, 400.0, 0.98

# ── Geometry ──────────────────────────────────────────────────────────────────
layout = place_panels_polygon(boundary, LAT, LON, module, tilt_deg, target_kwp=700.0)

print(f"Tilt (recommended)  : {tilt_deg} deg")
print(f"Row pitch           : {layout.pitch:.3f} m")
print(f"GCR                 : {layout.gcr:.4f}")
print(f"Rows placed         : {layout.n_rows}")
print(f"Total DC capacity   : {layout.total_kwp:.1f} kWp")

# ── Terrain analysis ──────────────────────────────────────────────────────────
bds = boundary_raw.bounds
dem = synthetic_dem(
    bds[0], bds[1], bds[2] - bds[0], bds[3] - bds[1],
    ns_slope_pct=12.0, ew_slope_pct=3.0,
    undulation_m=0.8, cell_size=5.0,
)
slopes  = compute_slopes(dem, layout.row_centroids, layout.pitch, module.length, tilt_deg)
terrain = apply_terrain_limits(
    layout.row_centroids, slopes, dem=dem,
    max_ns_slope_pct=20.0, earthworks_ns_min_pct=8.0, max_ew_slope_pct=10.0,
)
print(f"\nTerrain (NS 12%, EW 3%)")
print(f"  Accepted            : {terrain.n_kept}")
print(f"  Earthworks          : {terrain.n_earthworks}")
print(f"  Excluded            : {terrain.n_removed}")
print(f"  Cut/fill volume     : {terrain.cut_fill_volume_m3:.1f} m³")

# ── Equipment placement ───────────────────────────────────────────────────────
placement = place_equipment(
    layout.row_centroids, layout.n_per_row, layout.total_kwp,
    module.power_wp, inverter, xfmr,
)
print(f"\nInverters           : {placement.n_inverters}")

# ── Losses ────────────────────────────────────────────────────────────────────
losses = compute_all_losses(
    placement, layout.total_kwp,
    dc_cable, ac_cable,
    DC_V, AC_V, xfmr, ac_power_factor=AC_PF,
)
print(f"Total loss          : {losses.total_loss_pct:.3f} %")

# ── Bill of Quantities ────────────────────────────────────────────────────────
boq = bill_of_quantities(layout.total_modules, layout.total_kwp, placement, dc_cable, ac_cable)
print(f"\nModules             : {boq.n_modules}")
print(f"DC cable            : {boq.dc_cable_total_m:.0f} m  ({boq.dc_cable_mass_kg:.0f} kg Cu)")
print(f"AC cable            : {boq.ac_cable_total_m:.0f} m  ({boq.ac_cable_mass_kg:.0f} kg Cu)")
print(f"Trench volume       : {boq.trench_volume_m3:.1f} m³")

# ── Plot ──────────────────────────────────────────────────────────────────────
out_path = pathlib.Path(__file__).parent / "layout_lshape.png"
site_w = bds[2] - bds[0]
site_h = bds[3] - bds[1]

render_layout(
    site_w, site_h,
    layout.row_centroids, module.length, module.width, tilt_deg,
    placement,
    output_path=str(out_path),
    title="L-Shape Site Layout — Ahmedabad 23N  (terrain analysis)",
    boundary=boundary,
    terrain=terrain,
)
print(f"\nSaved: {out_path}")
