# pv-layout-design

Standalone Python tool for PV site geometric and electrical design.

Given a site boundary (rectangle or arbitrary polygon), a location (lat/lon),
a target DC capacity, and hardware specs, it computes:

- Panel row layout with inter-row shading constraint
- Inverter and transformer placement (k-means facility location)
- DC and AC cable losses (I²R per-segment)
- Transformer losses (IEC two-term nameplate model)
- Conductor cross-section sizing (IEC voltage-drop formula)
- String sizing (Voc/Vmp temperature-corrected)
- Bill of Quantities (cable mass, trench volume, control building)
- Annual energy yield (pvlib Perez transposition + 10-term loss chain)
- Terrain slope analysis and earthworks cut/fill estimate
- A rendered site plan (matplotlib)

## Quick start

```bash
cd plugins/pv-layout-design
pip install -e ".[dev]"
python examples/run_example.py   # runs the Ahmedabad 1 MWp worked example
pytest tests/ -v                 # run all 98 tests
```

## Package layout

```
pv_layout/
  geometry.py    row spacing, panel placement, polygon site, setbacks, GCR, tilt
  placement.py   inverter/transformer facility-location (k-means)
  losses.py      cable I²R + transformer IEC 60076 loss model
  electrical.py  conductor sizing (F5), string sizing (F6)
  boq.py         Bill of Quantities (F8)
  energy.py      annual energy yield via pvlib (F9)
  terrain.py     DEM slope analysis + earthworks (F10)
  plot.py        matplotlib site-plan renderer (rectangle or polygon boundary)
examples/
  run_example.py   end-to-end 1 MWp Ahmedabad example (saves layout.png)
  layout.png       rendered rectangle-site plan
  layout_lshape.png  rendered L-shape polygon plan
tests/
  test_geometry.py
  test_placement.py
  test_losses.py
  test_electrical.py
  test_boq.py
  test_energy.py
  test_terrain.py
```

## Feature reference

### Layer 1 — Base

| Feature | API | Description |
|---|---|---|
| Row layout | `place_panels(site, module, tilt_deg, target_kwp)` | Rectangular site, sun-angle pitch, k-modules per row |
| GCR | `LayoutResult.gcr` | Computed from pitch; or supply `gcr=` to force pitch |
| Equipment placement | `place_equipment(row_centroids, ...)` | k-means inverter + transformer locations |
| Cable losses | `compute_all_losses(placement, ...)` | Per-segment I²R DC + AC + IEC 60076 transformer |
| Plot | `render_layout(site_width, site_height, ..., boundary=None)` | Pass shapely `boundary` to draw polygon outline |

### Layer 2 — Electrical sizing

| Feature | API | Description |
|---|---|---|
| **F3** Tilt recommendation | `recommend_tilt(lat)` | RatedPower table: <5°→5°, 5–10°→lat, 10–20°→0.9×lat, >20°→0.85×lat |
| **F4** GCR input | `place_panels(..., gcr=0.40)` | Forces pitch = module_length / gcr; tilt returns NaN |
| **F5** Conductor sizing | `select_conductor_area(power_kw, voltage_v, length_m, material, max_vdrop_pct, three_phase, power_factor)` | IEC voltage-drop formula; returns smallest standard section ≥ S_min |
| **F6** String sizing | `string_sizing(module_spec, inverter_spec, temp_min_c, temp_max_c)` | Cold/hot Voc/Vmp from temperature coefficients; returns N_min, N_max, N_recommended |
| **F7** Multiple transformers | `place_equipment(..., n_transformers=2)` | Second k-means over inverter positions for transformer placement |
| **F8** Bill of Quantities | `bill_of_quantities(n_modules, total_kwp, placement, dc_cable, ac_cable)` | Cable mass (kg Cu/Al), trench volume (m³), control building area (m²) |

### Layer 3 — Site + yield

| Feature | API | Description |
|---|---|---|
| **F1** Polygon site | `place_panels_polygon(boundary, module, tilt_deg, target_kwp)` | shapely Polygon site; row strips clipped to boundary |
| **F2** Setbacks | `apply_setback(boundary, setback_m)` | Returns `boundary.buffer(-setback_m)` |
| **F9** Annual energy yield | `annual_energy_yield(lat, lon, total_kwp, tilt_deg, ..., tmy_data)` | pvlib Perez transposition + Faiman cell temperature + 10-term multiplicative loss chain; returns PR, specific yield (kWh/kWp/yr) |
| **F10** Terrain analysis | `compute_slopes(dem, row_centroids, ...)` + `apply_terrain_limits(...)` | Slope exclusion, earthworks cut/fill via least-squares plane fit |

## Polygon site example

```python
from shapely.geometry import Polygon
from pv_layout import (
    apply_setback, place_panels_polygon,
    ModuleSpec, place_equipment, InverterSpec, TransformerSpec,
    render_layout,
)

# L-shaped site boundary (metres)
boundary = Polygon([(0,0),(100,0),(100,60),(60,60),(60,100),(0,100)])
boundary_inner = apply_setback(boundary, setback_m=5.0)

module = ModuleSpec(length=2.0, width=1.0, power_wp=400.0)
layout = place_panels_polygon(boundary_inner, module, tilt_deg=20.0, target_kwp=500.0)
placement = place_equipment(
    layout.row_centroids, layout.n_per_row, layout.total_kwp,
    module.power_wp, InverterSpec(rated_kw=100.0),
    TransformerSpec(rated_kva=500.0, p_no_load_kw=0.5, p_load_rated_kw=5.0),
)

# Pass boundary to render_layout to draw the polygon outline
fig = render_layout(
    100, 100, layout.row_centroids, module.length, module.width, 20.0,
    placement, output_path="layout_lshape.png", boundary=boundary_inner,
)
```

## Conductor sizing example

```python
from pv_layout import select_conductor_area, string_sizing, ModuleSpec

# Size a DC string cable: 50 kW block, 1000 V DC, 80 m run, ≤0.5% vdrop
area_mm2 = select_conductor_area(50.0, 1000.0, 80.0,
                                  material="copper", max_vdrop_pct=0.5)
print(f"DC conductor: {area_mm2} mm²")  # → 35 mm²

# Size an AC feeder: 100 kW inverter, 400 V 3-phase, 120 m run, ≤1% vdrop
area_mm2 = select_conductor_area(100.0, 400.0, 120.0,
                                  material="copper", max_vdrop_pct=1.0,
                                  three_phase=True, power_factor=0.98)
print(f"AC conductor: {area_mm2} mm²")
```

## Energy yield example

```python
import pandas as pd
import pvlib
from pv_layout import annual_energy_yield

# Build a TMY DataFrame (or load from PVGIS)
times = pd.date_range("2023-01-01", periods=8760, freq="h", tz="UTC")
loc = pvlib.location.Location(23.0, 72.6, tz="UTC")
cs  = loc.get_clearsky(times)
tmy = pd.DataFrame({
    "ghi": cs["ghi"].values, "dhi": cs["dhi"].values, "dni": cs["dni"].values,
    "temp_air": 25.0, "wind_speed": 3.0,
}, index=times)

energy = annual_energy_yield(
    lat=23.0, lon=72.6,
    total_kwp=1000.0, tilt_deg=23.0,
    dc_cable_loss_pct=0.3, ac_cable_loss_pct=0.2, transformer_loss_pct=0.5,
    tmy_data=tmy,
)
print(f"Annual yield : {energy.annual_kwh:,.0f} kWh/yr")
print(f"Specific yield: {energy.specific_yield_kwh_kwp:.0f} kWh/kWp/yr")
print(f"PR           : {energy.performance_ratio:.3f}")
```

## Worked example — Ahmedabad, India (23.0°N, 72.6°E)

| Parameter | Value |
|---|---|
| Site | 200 m × 150 m |
| Module | 400 Wp, 2.0 × 1.0 m, portrait, 30° tilt |
| Inverter | 100 kW |
| Transformer | 500 kVA, P_no_load=0.5 kW, P_load_rated=5.0 kW |
| Cable | copper, 35 mm² (DC + AC) |
| DC bus | 1 000 V |
| AC bus | 400 V 3-phase, PF=0.98 |
| Shading criterion | no shading ±4 h from solar noon on winter solstice |

## Citations

| Formula | Source |
|---|---|
| Sun elevation angle (winter solstice) | pvlib `solarposition.get_solarposition` (BSD-3) |
| Row pitch: `L·cos(tilt) + L·sin(tilt)/tan(β)` | Duffie & Beckman, *Solar Engineering of Thermal Processes*, 4th ed. (2013), §1.7 |
| ±4 h from solar noon shading criterion | Marion & Dobos, NREL/TP-6A20-62641 (2013) |
| Tilt recommendation table | RatedPower methodology §5.3.3.4 |
| Inverter/transformer facility location | weighted k-means via sklearn (BSD-3) |
| Cable I²R loss | IEC 60228:2004 per-segment |
| Conductor sizing: `S_min = 2ρLI/(δV·V)` DC, `S_min = ρLI·cosφ·√3/(δV·V_LL)` AC | IEC 60228:2004, Electrical Methodology §3 |
| String sizing cold/hot Voc/Vmp | Electrical Methodology §1 Eq 1.1–1.7 |
| Transformer two-term loss model | IEC 60076-1:2011, §5.11 |
| Energy yield loss chain (10-term) | Energy Yield Methodology §10, Tables 10.1–10.4 |
| PR: `E_annual / (total_kWp × GHI_annual / 1000)` | Energy Yield Methodology Eq 10.2 |
| Cell temperature (Faiman simplified) | Energy Yield Methodology §6 Eq 6.15 |
| Polygon row clipping | shapely `intersection` |
| Slope: `ΔZ / sqrt(Δx²+Δy²)` | Topography Analysis Methodology Eq 2.4 |
| Earthworks cut/fill (least-squares plane fit) | Topography Analysis Methodology Eq 3.3–3.4 |
