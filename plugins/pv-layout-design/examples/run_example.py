"""
End-to-end worked example: ~1 MWp PV site near Ahmedabad, India.

This script is the single canonical worked example used throughout the
project — its numbers must match the pytest fixtures in tests/.

Fixture
-------
Site        : 200 m × 150 m rectangle, lat 23.0°N, lon 72.6°E
Module      : 400 Wp, 2.0 m × 1.0 m (portrait), 30° tilt (fixed)
Inverter    : 100 kW rated AC
Transformer : 500 kVA, P_no_load = 0.5 kW, P_load_rated = 5.0 kW
Cable       : copper, 35 mm² cross-section (DC and AC sides)
DC bus      : 1 000 V
AC bus      : 400 V (3-phase), power factor 0.98
Shading     : no inter-row shading ±4 h from solar noon on winter solstice

Run
---
    python examples/run_example.py

Saves layout.png in the same directory as this script.
"""

import sys
import pathlib

# Make the package importable when run directly from the examples/ directory
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from pv_layout import (
    place_panels, ModuleSpec, SiteSpec,
    place_equipment, InverterSpec, TransformerSpec,
    compute_all_losses, CableSpec,
    render_layout,
)

# ── Specs ──────────────────────────────────────────────────────────────────────
site = SiteSpec(width=200.0, height=150.0, lat=23.0, lon=72.6)
module = ModuleSpec(length=2.0, width=1.0, power_wp=400.0)
inverter = InverterSpec(rated_kw=100.0)
transformer = TransformerSpec(rated_kva=500.0, p_no_load_kw=0.5, p_load_rated_kw=5.0)
dc_cable = CableSpec(material="copper", area_mm2=35.0)
ac_cable = CableSpec(material="copper", area_mm2=35.0)

TARGET_KWP = 1000.0
TILT_DEG = 30.0
DC_VOLTAGE_V = 1000.0
AC_VOLTAGE_V = 400.0
AC_PF = 0.98

# ── Stage 1: geometry ──────────────────────────────────────────────────────────
layout = place_panels(site, module, TILT_DEG, TARGET_KWP)

print("=" * 60)
print("GEOMETRY")
print("=" * 60)
print(f"  Critical sun elevation (beta): {layout.beta_deg:.2f} deg")
print(f"  Row pitch                   : {layout.pitch:.3f} m")
print(f"  GCR (module_length / pitch) : {layout.gcr:.4f}")
print(f"  Rows placed                 : {layout.n_rows}")
print(f"  Modules per row             : {layout.n_per_row}")
print(f"  Total modules               : {layout.total_modules}")
print(f"  Total DC capacity           : {layout.total_kwp:.1f} kWp")
if layout.target_fits:
    print(f"  Target {TARGET_KWP:.0f} kWp             : FITS")
else:
    print(f"  Target {TARGET_KWP:.0f} kWp             : DOES NOT FIT")
    print(f"    Only {layout.total_kwp:.1f} kWp can be placed.")
    print(f"    Approx. additional land needed: see pitch x rows shortfall.")
print()

# ── Stages 2–4: placement + cable lengths ──────────────────────────────────────
placement = place_equipment(
    layout.row_centroids, layout.n_per_row, layout.total_kwp,
    module.power_wp, inverter, transformer,
)

print("=" * 60)
print("PLACEMENT")
print("=" * 60)
print(f"  Inverters                   : {placement.n_inverters}")
print(f"  Transformer(s)              : {placement.n_transformers}")
print(f"  Total DC cable length       : {placement.total_dc_cable_m:.1f} m")
print(f"  Total AC cable length       : {placement.total_ac_cable_m:.1f} m")
print()

# ── Stage 5: losses ────────────────────────────────────────────────────────────
losses = compute_all_losses(
    placement, layout.total_kwp,
    dc_cable, ac_cable,
    DC_VOLTAGE_V, AC_VOLTAGE_V,
    transformer,
    ac_power_factor=AC_PF,
)

print("=" * 60)
print("LOSSES")
print("=" * 60)
print(f"  DC cable loss               : {losses.dc_cable_loss_kw:.4f} kW"
      f"  ({losses.dc_cable_loss_pct:.4f} % of DC input)")
print(f"  AC cable loss               : {losses.ac_cable_loss_kw:.4f} kW"
      f"  ({losses.ac_cable_loss_pct:.4f} % of AC input)")
print(f"  Transformer loss            : {losses.transformer_loss_kw:.4f} kW"
      f"  ({losses.transformer_loss_pct:.4f} % of TX input)")
print(f"  -- Total loss               : {losses.total_loss_kw:.4f} kW"
      f"  ({losses.total_loss_pct:.4f} % of gross DC output)")
print()
print("  Hand-check DC cable loss (per-segment sum):")
import math as _math
rho = 1.68e-8
A_m2 = 35e-6
per_block = []
for i, (L, P) in enumerate(zip(placement.dc_cable_lengths_m, placement.block_kwp)):
    R = rho * L / A_m2
    I = P * 1000.0 / DC_VOLTAGE_V
    per_block.append(2 * I * I * R)   # + and − conductors
total_dc_w = sum(per_block)
print(f"    sum_i(2 * I_i^2 * R_i) = {total_dc_w:.2f} W = {total_dc_w/1000:.4f} kW"
      f"  (should match {losses.dc_cable_loss_kw:.4f} kW above)")
print()

# ── Stage 6: plot ──────────────────────────────────────────────────────────────
out_path = pathlib.Path(__file__).parent / "layout.png"
render_layout(
    site.width, site.height,
    layout.row_centroids, module.length, module.width, TILT_DEG,
    placement,
    output_path=str(out_path),
    title=f"PV Layout - Ahmedabad 23N - {layout.total_kwp:.0f} kWp",
)
print(f"Layout saved to: {out_path}")

# ══════════════════════════════════════════════════════════════════════════════
# Layer 2 / 3 additions
# ══════════════════════════════════════════════════════════════════════════════

from pv_layout import (
    recommend_tilt, select_conductor_area, bill_of_quantities,
    synthetic_dem, compute_slopes, apply_terrain_limits,
)

print()
print("=" * 60)
print("LAYER 2 ADDITIONS")
print("=" * 60)

# F3: Tilt recommendation
tilt_rec = recommend_tilt(site.lat)
print(f"  Recommended tilt for {site.lat}N : {tilt_rec} deg")
print(f"  (Used in example    : {TILT_DEG} deg)")

# F4: GCR input
from pv_layout import place_panels, ModuleSpec, SiteSpec
layout_gcr = place_panels(site, module, TILT_DEG, 1000.0, gcr=0.40)
print(f"  GCR=0.40 forced pitch: {layout_gcr.pitch:.3f} m"
      f"  (vs sun-angle pitch: {layout.pitch:.3f} m)")

# F5: Conductor sizing — size the WORST-CASE single segment
# DC: largest single block (approx avg block power, avg cable length)
avg_block_kw = layout.total_kwp / len(placement.block_centroids)
avg_dc_len = placement.total_dc_cable_m / len(placement.block_centroids) if len(placement.block_centroids) > 0 else 1.0
dc_area = select_conductor_area(avg_block_kw, DC_VOLTAGE_V,
                                 max(avg_dc_len, 1.0),
                                 material="copper", max_vdrop_pct=0.5)
# AC: one inverter → transformer (each carries 1/n_inverters of total power)
avg_inv_kw = layout.total_kwp / placement.n_inverters
avg_ac_len = placement.total_ac_cable_m / placement.n_inverters
ac_area = select_conductor_area(avg_inv_kw, AC_VOLTAGE_V,
                                 max(avg_ac_len, 1.0),
                                 material="copper", max_vdrop_pct=1.0,
                                 three_phase=True, power_factor=AC_PF)
print(f"  Recommended DC conductor : {dc_area} mm2 copper  (avg segment: {avg_block_kw:.1f} kW / {avg_dc_len:.1f} m)")
print(f"  Recommended AC conductor : {ac_area} mm2 copper  (avg segment: {avg_inv_kw:.1f} kW / {avg_ac_len:.1f} m)")
print(f"  (Example uses fixed      : 35 mm2 for both)")

# F8: BoQ
boq = bill_of_quantities(layout.total_modules, layout.total_kwp,
                          placement, dc_cable, ac_cable)
print()
print("  --- Bill of Quantities ---")
print(f"  Modules             : {boq.n_modules}")
print(f"  Inverters           : {boq.n_inverters}")
print(f"  Transformers        : {boq.n_transformers}")
print(f"  DC cable total      : {boq.dc_cable_total_m:.1f} m"
      f"  ({boq.dc_cable_mass_kg:.1f} kg Cu)")
print(f"  AC cable total      : {boq.ac_cable_total_m:.1f} m"
      f"  ({boq.ac_cable_mass_kg:.1f} kg Cu)")
print(f"  Trench volume       : {boq.trench_volume_m3:.2f} m3"
      f"  ({boq.trench_depth_m} m deep x {boq.trench_width_m} m wide)")
print(f"  Control building    : {boq.control_building_m2:.0f} m2")

print()
print("=" * 60)
print("LAYER 3 ADDITIONS")
print("=" * 60)

# F10: Terrain
dem = synthetic_dem(0, 0, site.width, site.height,
                    ns_slope_pct=8.0, ew_slope_pct=2.0,
                    undulation_m=0.5, cell_size=5.0)
slopes = compute_slopes(dem, layout.row_centroids, layout.pitch,
                        module.length, TILT_DEG)
terrain = apply_terrain_limits(
    layout.row_centroids, slopes, dem=dem,
    max_ns_slope_pct=25.0, earthworks_ns_min_pct=10.0, max_ew_slope_pct=10.0
)
print(f"  Synthetic DEM N-S slope : 8 %  E-W: 2 %")
print(f"  Rows kept               : {terrain.n_kept} / {layout.n_rows}")
print(f"  Rows needing earthworks : {terrain.n_earthworks}")
print(f"  Rows removed (too steep): {terrain.n_removed}")
print(f"  Earthworks volume est.  : {terrain.cut_fill_volume_m3:.1f} m3")
print()

# F9: Energy yield (offline synthetic TMY)
try:
    import numpy as np
    import pandas as pd
    import pvlib
    times = pd.date_range("2023-01-01", periods=8760, freq="h", tz="UTC")
    loc = pvlib.location.Location(site.lat, site.lon, tz="UTC")
    cs = loc.get_clearsky(times)
    tmy = pd.DataFrame(index=times)
    tmy["ghi"] = cs["ghi"].values
    tmy["dhi"] = cs["dhi"].values
    tmy["dni"] = cs["dni"].values
    tmy["temp_air"] = 25.0 + 5.0 * np.sin(np.linspace(0, 2 * np.pi, 8760))
    tmy["wind_speed"] = 3.0
    from pv_layout import annual_energy_yield
    energy = annual_energy_yield(
        lat=site.lat, lon=site.lon,
        total_kwp=layout.total_kwp, tilt_deg=TILT_DEG,
        dc_cable_loss_pct=losses.dc_cable_loss_pct,
        ac_cable_loss_pct=losses.ac_cable_loss_pct,
        transformer_loss_pct=losses.transformer_loss_pct,
        tmy_data=tmy,
    )
    print(f"  Annual energy (clear-sky): {energy.annual_kwh:.0f} kWh/yr")
    print(f"  Specific yield           : {energy.specific_yield_kwh_kwp:.0f} kWh/kWp/yr")
    print(f"  Performance Ratio (PR)   : {energy.performance_ratio:.3f}")
    print(f"  POA irradiation          : {energy.poa_annual_kwh_m2:.0f} kWh/m2/yr")
except Exception as exc:
    print(f"  Energy yield skipped: {exc}")
