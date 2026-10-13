# PV Layout Design — Handoff Document

## Resuming (read this first)

Fresh session: read this file, run `git log --oneline -5` to confirm branch state,
then `pytest plugins/pv-layout-design/tests/ -v` to confirm 98/98 pass.
Continue from **Next steps** below.

---

## Project goal

Standalone Python package that takes a site polygon, module spec, and electrical
specs and returns: geometry (rows, GCR, pitch), equipment placement, cable losses,
bill of quantities, annual energy yield (PR, specific yield), and terrain
earthworks estimate. Outputs feed OptimaRE's MILP solver as fixed parameters
(loss %, area utilisation).

---

## Branches

| Branch | Tag | Contains |
|---|---|---|
| `master` | `v1-base` | Layer 1 base: rectangle site, fixed tilt, k-means inverter placement, I²R + transformer losses, matplotlib plot. 33 tests. |
| `improvements` | — | Layer 2/3 additions: all 10 features below. 98 tests total. |

---

## Stages

| # | Feature | File | Status | What was built |
|---|---|---|---|---|
| F3 | Latitude-based tilt recommendation | `geometry.py` | **DONE** | `recommend_tilt(lat)` — RatedPower table: <5°→5°, 5–10°→lat, 10–20°→lat×0.9, >20°→lat×0.85 |
| F4 | GCR as direct input | `geometry.py` | **DONE** | `gcr=` kwarg in `place_panels`; if given, `pitch=length/gcr`, `beta_deg=NaN` |
| F5 | Conductor sizing | `electrical.py` | **DONE** | `select_conductor_area(...)` — IEC voltage-drop formula, returns smallest standard section ≥ S_min |
| F6 | String sizing | `electrical.py` | **DONE** | `string_sizing(...)` — cold/hot Voc/Vmp from temperature coefficients, N_min/N_max/N_recommended |
| F1 | Polygon site | `geometry.py` | **DONE** | `place_panels_polygon(boundary, ...)` — shapely row-clipping; handles L/U/irregular shapes |
| F2 | Setbacks | `geometry.py` | **DONE** | `apply_setback(boundary, setback_m)` — `boundary.buffer(-setback_m)` |
| F7 | Multiple transformers | `placement.py` | **DONE** | `n_transformers=` kwarg in `place_equipment`; second k-means on inverter positions |
| F8 | Bill of Quantities | `boq.py` | **DONE** | `bill_of_quantities(...)` → `BoQResult`: cable mass, trench volume, control building area |
| F9 | Annual energy yield | `energy.py` | **DONE** | `annual_energy_yield(...)` → `EnergyResult`: pvlib Perez transposition + Faiman cell temp + 10-term loss chain; PR, specific yield |
| F10 | Terrain slope analysis | `terrain.py` | **DONE** | `synthetic_dem`, `compute_slopes`, `apply_terrain_limits` — flat layout, slope-exclusion, earthworks cut/fill via least-squares plane |
| F11 | Roads + routed cabling | — | **NOT STARTED** | Deprioritised (see Cut-list) |

---

## How to run

```bash
# End-to-end example (from plugins/pv-layout-design/)
python examples/run_example.py          # prints all stages, saves layout.png

# Tests (from repo root)
pytest plugins/pv-layout-design/tests/ -v
# Expected: 98 passed
```

---

## Key decisions and formula sources

| Decision | Formula / rule | Source |
|---|---|---|
| Pitch from sun angle | `pitch = L·cos(tilt) + L·sin(tilt)/tan(β)`, β = min elevation at ±4 h winter solstice | Design_Methodology §5.4.3 |
| GCR | `GCR = module_length / pitch` | Design_Methodology §5.4.3 |
| Tilt recommendation | <5°→5°; 5–10°→lat; 10–20°→lat×0.9; >20°→lat×0.85 | Design_Methodology §5.3.3.4 |
| DC cable loss | `Σ I_i² × R_i` per segment (NOT one aggregated segment) | IEC 60228; Electrical_Methodology |
| AC cable loss | Same per-segment sum, 3-phase | IEC 60228 |
| Transformer loss | `P_no_load + P_load_rated × (S/S_rated)²` | IEC 60076-1; Design_Methodology §5.3.4.2 |
| Conductor sizing | `S_min = 2ρLI/(δV·V)` DC; `S_min = ρLI·cosφ·√3/(δV·V_LL)` AC | Electrical_Methodology §3; T_D_UndergroundLine §2 |
| String sizing | `N_max = min(floor(V_sys/Voc_cold), floor(V_mppt_max/Vmp_ref))` | Electrical_Methodology §1 Eq 1.1–1.7 |
| Terrain: flat layout | No pitch adjustment for slope — rows stay flat, steep rows excluded | Design_Methodology §5.2.2 |
| Slope calc | `ΔZ / sqrt(Δx²+Δy²)` | Topography_Analysis_Methodology Eq 2.4 |
| Earthworks cut/fill | Least-squares plane fit over affected rows | Topography_Analysis_Methodology Eq 3.3–3.4 |
| Energy yield loss chain | Multiplicative: soiling→mismatch→LID→T-derate→DC cable→inverter→AC cable→TX→availability | Energy_Yield_Methodology §10 Tables 10.1–10.4 |
| PR definition | `PR = E_annual / (total_kwp × GHI_annual / 1000)` | Energy_Yield_Methodology Eq 10.2 |
| Cell temperature | `T_cell = T_amb + G × (NOCT−20) / 800` (Faiman simplified) | Energy_Yield_Methodology §6 Eq 6.15 |

---

## Known issues and limitations

1. **L-shape polygon plot** — `render_layout` in `plot.py` draws the bounding box,
   not the shapely polygon outline. The L-shaped layout PNG shows rows correctly
   but the boundary is invisible.
2. **Straight-line cable lengths** — all cable lengths are Euclidean (no road
   routing). Underestimates real cable lengths, especially for complex sites.
3. **Conductor sizing is advisory** — `select_conductor_area` returns the minimum
   section by voltage drop; ampacity derating (buried depth, grouping) is not applied.
4. **Energy yield uses clear-sky synthetic TMY** in the example script — a real
   TMY fetch from PVGIS requires network access (`tmy_data=None`).
5. **`layout_lshape.png` is untracked** — exists in `examples/` but not committed.
6. **Single tilt only** — no E-W bifacial or single-axis tracker layout.

---

## Next steps (priority order)

1. ~~**Commit `layout_lshape.png`**~~ — **DONE** (commit `44994a9`)
2. ~~**Fix polygon boundary outline in `plot.py`**~~ — **DONE** (commit `7ec1ffb`);
   `render_layout` now accepts `boundary=<shapely Polygon>` and draws
   `ax.plot(*boundary.exterior.xy)` instead of the bounding-box rectangle.
3. ~~**Update README.md**~~ — **DONE** (commit `7ec1ffb`); full feature
   reference (F1–F10), three usage examples, updated citations.
4. **Push to GitHub** — user must run manually:
   ```
   git push -u origin improvements
   ```
5. **F11 roads** (optional, low priority) — spine + perimeter roads; route cable
   along roads instead of straight-line.

---

## Cut-list (drop first if time runs short)

1. F11 roads + routed cabling (highest effort, lowest academic credit)
2. Ampacity derating in conductor sizing (minor refinement to F5)
3. Polygon boundary outline in plot (cosmetic only)

---

## Regression gate

Every commit: `pytest plugins/pv-layout-design/tests/ -v` — all 98 tests must pass.
