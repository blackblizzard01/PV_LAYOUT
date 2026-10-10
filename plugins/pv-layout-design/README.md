# pv-layout-design

Standalone Python tool for PV site geometric and electrical design.

Given a rectangular site, a location (lat/lon), a target DC capacity, and
hardware specs, it computes:

- Panel row layout with inter-row shading constraint
- Inverter and transformer placement (k-means facility location)
- DC and AC cable losses (I²R)
- Transformer losses (IEC two-term nameplate model)
- A rendered site plan (matplotlib)

## Quick start

```bash
cd plugins/pv-layout-design
pip install -e ".[dev]"
python examples/run_example.py   # runs the Ahmedabad 1 MWp worked example
pytest tests/ -v                 # run all tests
```

## Package layout

```
pv_layout/
  geometry.py    row spacing, panel placement, GCR
  placement.py   inverter/transformer facility-location (k-means)
  losses.py      cable I²R + transformer IEC 60076 loss model
  plot.py        matplotlib site-plan renderer
examples/
  run_example.py end-to-end 1 MWp Ahmedabad example (saves layout.png)
tests/
  test_geometry.py
  test_placement.py
  test_losses.py
```

## Citations

| Formula | Source |
|---|---|
| Sun elevation angle (winter solstice) | pvlib `solarposition.get_solarposition` (BSD-3); pvlib docs at pvlib-python.readthedocs.io |
| Row pitch: `L·cos(tilt) + L·sin(tilt)/tan(β)` | Duffie & Beckman, *Solar Engineering of Thermal Processes*, 4th ed. (2013), §1.7 |
| ±4 h from solar noon shading criterion | Marion & Dobos, NREL/TP-6A20-62641 (2013) |
| Inverter/transformer facility location | Drezner & Hamacher, *Facility Location: Applications and Theory* (2002), Ch. 2; weighted k-means via sklearn (BSD-3) |
| Cable I²R loss | Chapman, *Electric Machinery Fundamentals*, 5th ed. (2011), §2.3 |
| Conductor resistivity (Cu 1.68×10⁻⁸ Ω·m, Al 2.82×10⁻⁸ Ω·m) | IEC 60228:2004, Table 1 |
| Transformer two-term loss model | IEC 60076-1:2011, §5.11 |

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
