# PV Layout Design — Improvement Plan (Layer 2 / Layer 3)

## Layers

- **Layer 1** (tag `v1-base`, branch `master`) — base tool: rectangle site, fixed tilt, k-means placement, I²R + transformer losses. Never broken.
- **Layer 2** — improvements to existing features (branch `improvements`).
- **Layer 3** — new capabilities from the original out-of-scope list.

## Method decisions (conflicts vs RatedPower reference docs)

| Conflict | Decision | Source |
|---|---|---|
| Tilt default | Add `recommend_tilt(lat)` helper; `tilt_deg` stays a required param — no default change | Design_Methodology §5.3.3.4 |
| Transformer placement | Extend to N transformers (default=1, Layer 1 unchanged) | Design_Methodology §5.3.4 |
| Terrain approach | Flat layout + slope-exclusion + earthworks — NO pitch adjustment | Design §5.2.2: *"RatedPower considers the terrain to be flat when creating the final layout"* |
| Energy yield | pvlib + PVGIS/NASA TMY; base cable+TX % feed in as itemized terms | Energy_Yield_Methodology §10; Design §5.2.3 |

## Feature table

| # | Feature | Layer | File |
|---|---|---|---|
| F3 | Latitude-based tilt recommendation | 2 | geometry.py |
| F4 | GCR or pitch as direct input | 2 | geometry.py |
| F5 | Conductor sizing (ampacity + voltage-drop) | 2 | electrical.py |
| F6 | String sizing (modules/string from MPPT window & T) | 2 | electrical.py |
| F1 | Irregular polygon site boundary | 3 | geometry.py |
| F2 | Setbacks + restricted areas | 3 | geometry.py |
| F7 | Multiple transformers / power stations | 2 | placement.py |
| F8 | Bill of Quantities output | 2 | boq.py |
| F9 | Annual energy yield (PR, specific yield) | 3 | energy.py |
| F10 | Terrain slope-exclusion + earthworks | 3 | terrain.py |

## Regression gate

Every commit: `pytest plugins/pv-layout-design/tests/ -v` — the original 33 Layer-1 tests must stay green.
