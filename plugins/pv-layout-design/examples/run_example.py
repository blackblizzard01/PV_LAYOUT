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
    per_block.append(I * I * R)
total_dc_w = sum(per_block)
print(f"    sum_i(I_i^2 * R_i) = {total_dc_w:.2f} W = {total_dc_w/1000:.4f} kW"
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
