"""
L-shape site example — Ahmedabad 23N.

Site: 200 m × 150 m minus a 100 m × 90 m notch from the top-right corner.
Polygon vertices (metres): (0,0)→(200,0)→(200,60)→(100,60)→(100,150)→(0,150)
Setback: 5 m from every edge.

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
)

# ── Site ──────────────────────────────────────────────────────────────────────
LAT, LON = 23.0, 72.6
boundary_raw = Polygon([
    (0, 0), (200, 0), (200, 60), (100, 60), (100, 150), (0, 150),
])
boundary = apply_setback(boundary_raw, setback_m=5.0)

# ── Specs ─────────────────────────────────────────────────────────────────────
tilt_deg  = recommend_tilt(LAT)          # → 20° for lat 23°
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

# ── Equipment placement ───────────────────────────────────────────────────────
placement = place_equipment(
    layout.row_centroids, layout.n_per_row, layout.total_kwp,
    module.power_wp, inverter, xfmr,
)
print(f"Inverters           : {placement.n_inverters}")

# ── Losses ────────────────────────────────────────────────────────────────────
losses = compute_all_losses(
    placement, layout.total_kwp,
    dc_cable, ac_cable,
    DC_V, AC_V, xfmr, ac_power_factor=AC_PF,
)
print(f"Total loss          : {losses.total_loss_pct:.3f} %")

# ── Plot (with real polygon boundary) ────────────────────────────────────────
out_path = pathlib.Path(__file__).parent / "layout_lshape.png"
bbox = boundary_raw.bounds          # (minx, miny, maxx, maxy)
site_w, site_h = bbox[2] - bbox[0], bbox[3] - bbox[1]

render_layout(
    site_w, site_h,
    layout.row_centroids, module.length, module.width, tilt_deg,
    placement,
    output_path=str(out_path),
    title="L-Shape Site Layout — Ahmedabad 23N",
    boundary=boundary,              # ← draws actual polygon outline
)
print(f"Saved: {out_path}")
