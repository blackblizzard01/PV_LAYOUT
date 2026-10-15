"""
End-to-end validation on the five real-location example sites.

Generic invariants are checked on every site; each site also has checks for
the feature it exists to exercise.  Energy is compared with PVGIS PVcalc
(stored in expected.json), corrected for the TMY year being sunnier or duller
than the 2005–2020 mean.  Everything runs offline from the cached files.
"""

import json
import math
import pathlib

import numpy as np
import pytest
import shapely

from pv_layout.design import DesignConfig, run_design
from pv_layout.electrical import string_sizing
from pv_layout.terrain import EARTHWORKS, EXCLUDED

EXAMPLES = pathlib.Path(__file__).parent.parent / "examples" / "sites"

SITES = ["bhadla", "sanand", "fukushima", "darlington_point", "cambridgeshire"]
_cache = {}


@pytest.fixture(params=SITES)
def case(request):
    name = request.param
    if name not in _cache:
        cfg = DesignConfig.from_toml(EXAMPLES / name / "config.toml")
        exp = json.loads((EXAMPLES / name / "expected.json").read_text(encoding="utf-8"))
        _cache[name] = (cfg, run_design(cfg), exp)
    return name, *_cache[name]


def _result(name):
    if name not in _cache:
        cfg = DesignConfig.from_toml(EXAMPLES / name / "config.toml")
        exp = json.loads((EXAMPLES / name / "expected.json").read_text(encoding="utf-8"))
        _cache[name] = (cfg, run_design(cfg), exp)
    return _cache[name][1]


# ── Generic invariants ────────────────────────────────────────────────────────

def test_tables_inside_usable_area(case):
    _, _, r, _ = case
    assert shapely.covered_by(r.tables.boxes(), r.usable.buffer(1e-6)).all()


def test_tables_clear_of_restricted_areas(case):
    _, cfg, r, _ = case
    for name, poly in r.site.restricted:
        zone = poly.buffer(cfg.site.restricted_setback_m - 0.01)
        assert not shapely.intersects(r.tables.boxes(), zone).any(), name


def test_tables_do_not_overlap(case):
    _, _, r, _ = case
    b = r.tables.boxes()
    i, j = shapely.STRtree(b).query(b, predicate="intersects")
    m = i < j
    if m.any():
        assert shapely.area(shapely.intersection(b[i[m]], b[j[m]])).max() < 1e-6


def test_string_and_voltage_limits(case):
    _, cfg, r, _ = case
    m, e = cfg.module, cfg.electrical
    n = r.modules_per_string
    voc_cold = m.voc_v * (1 + m.voc_tc_pct / 100 * (e.t_min_c - 25))
    t_hot = e.t_max_c + (m.noct_c - 20) * 1000 / 800
    vmp_hot = m.vmp_v * (1 + m.vmp_tc_pct / 100 * (t_hot - 25))
    assert n * voc_cold <= e.system_voltage_v
    assert n * m.vmp_v <= e.mppt_max_v
    assert n * vmp_hot >= e.mppt_min_v
    assert r.tables.n_modules == r.tables.n_tables * r.elec.strings_per_table * n
    ss = string_sizing(m.voc_v, m.vmp_v, m.voc_tc_pct, m.vmp_tc_pct, e.system_voltage_v,
                       e.mppt_min_v, e.mppt_max_v, e.t_min_c, e.t_max_c, m.noct_c)
    assert n == ss.n_recommended


def test_inverter_blocks_within_dc_ac_cap(case):
    _, cfg, r, _ = case
    e = cfg.electrical
    assert r.elec.inverter_dc_kwp.max() <= e.inverter_ac_kw * e.dc_ac_ratio + 1e-6
    assert r.elec.inverter_dc_kwp.sum() == pytest.approx(r.kpis["dc_kwp"], abs=0.1)


def test_power_station_capacity(case):
    _, cfg, r, _ = case
    e = cfg.electrical
    assert (r.elec.ps_ac_kw() <= e.ps_kva * e.power_factor + 1e-6).all() or e.inverter_type == "central"
    for poly in r.elec.ps_polys:
        assert not shapely.intersects(r.tables.boxes(), poly).any()


def test_routes_are_manhattan_and_connected(case):
    _, _, r, _ = case
    for route in r.elec.routes:
        p = route.points
        assert route.length_m == pytest.approx(np.abs(np.diff(p, axis=0)).sum())
        assert route.length_m + 1e-9 >= np.hypot(*(p[-1] - p[0]))
    assert len(r.elec.routes_of("mv")) == r.elec.n_power_stations


def test_energy_matches_pvgis(case):
    name, _, r, exp = case
    p = exp["pvgis"]
    ref = p["E_y_kwh_kwp"] * p["tmy_over_longterm"]
    ratio = r.energy.specific_yield_kwh_kwp / ref
    assert abs(ratio - 1) < exp["tolerance"]["yield_vs_pvgis_rel"], f"{name}: {ratio:.3f}"
    assert 0.75 < r.energy.pr_poa < 0.92


def test_regression_snapshot(case):
    name, _, r, exp = case
    tol = exp["tolerance"]["snapshot_rel"]
    for key, want in exp["snapshot"].items():
        got = r.kpis[key]
        if want == 0:
            assert got == 0, f"{name}.{key}"
        else:
            assert got == pytest.approx(want, rel=tol, abs=1e-9), f"{name}.{key}: {got} vs {want}"


def test_density_and_losses_plausible(case):
    _, _, r, _ = case
    assert 0.25 < r.kpis["mwp_per_ha"] < 1.2
    assert 0 < r.losses["dc_cable_loss_pct"] < 2.5
    assert 0 <= r.losses["ac_cable_loss_pct"] < 2.0
    assert 0.3 < r.losses["transformer_loss_pct"] < 2.0
    assert max(r.losses["max_vdrop_pct"].values()) < 3.0


def test_no_unexpected_warnings(case):
    _, _, r, _ = case
    assert r.warnings == []


# ── Site-specific checks ──────────────────────────────────────────────────────

def test_bhadla_central_topology():
    r = _result("bhadla")
    assert r.elec.spec.inverter_type == "central"
    assert len(r.elec.combiner_xy) > 0
    assert r.elec.n_power_stations == r.elec.n_inverters
    assert not r.elec.routes_of("lv_ac") and r.elec.routes_of("dc_main")
    cb_strings = np.bincount(r.elec.table_combiner) * r.elec.strings_per_table
    assert cb_strings.max() <= r.config.electrical.combiner_max_strings


def test_sanand_kml_restrictions_split_site():
    r = _result("sanand")
    names = {n for n, _ in r.site.restricted}
    assert {"RA-Rural road", "RA-Sanand-Moraiya Road", "RA-Brick kiln", "RA-Farmhouse",
            "RA-Orchard grove"} <= names
    assert len(r.usable.geoms) >= 3                       # two roads cut the parcel
    assert r.kpis["gcr"] == pytest.approx(0.40)
    lon, lat = r.site.frame.to_lonlat(*r.elec.substation_xy)
    assert (float(lon), float(lat)) == pytest.approx((72.40156, 22.96708), abs=1e-6)
    assert r.kpis["tables_earthworks"] == 0 and r.kpis["tables_excluded_terrain"] == 0


def test_fukushima_terrain_classes_and_earthworks():
    r = _result("fukushima")
    st = r.terrain_all.status
    assert (st == EXCLUDED).sum() > 0 and (st == EARTHWORKS).sum() > 0 and (st == 0).sum() > 0
    lim = r.config.terrain
    ok = st != EXCLUDED
    assert (np.abs(r.terrain_all.ns_slope_pct[ok]) <= lim.ns_earthworks_pct + 1e-9).all()
    assert (np.abs(r.terrain_all.ew_slope_pct[ok]) <= lim.ew_earthworks_pct + 1e-9).all()
    assert r.boq["earthworks_cut_m3"] > 0 and r.boq["earthworks_fill_m3"] > 0
    assert r.dem.elevations.max() - r.dem.elevations.min() > 60       # real hillside relief
    assert r.losses["dc_voltage_v"] > 0 and r.config.electrical.mv_voltage_v == 6600


def test_darlington_southern_hemisphere_and_target():
    r = _result("darlington_point")
    assert r.site.lat < 0 and r.kpis["azimuth_deg"] == 0 and r.energy.azimuth_deg == 0
    table_kwp = r.tables.modules_per_table * 0.55
    removed = r.kpis["tables_removed_for_ps"]
    assert 45000 - (removed + 1) * table_kwp <= r.kpis["dc_kwp"] <= 45000 + table_kwp
    assert r.tables_all.n_tables * table_kwp > 50000                # trim actually happened
    assert r.tables.beta_deg > 0 and r.site.lat < 0                  # June-solstice sun angle used
    assert r.energy.clipping_loss_frac >= 0


def test_cambridgeshire_multi_area_roads_shading():
    r = _result("cambridgeshire")
    assert len(r.site.available) == 3
    roads = r.site.roads(r.config.site.road_spacing_m, r.config.site.road_width_m)
    assert roads.area > 0 and not shapely.intersects(r.tables.boxes(), roads).any()
    from pv_layout.tables import component_ids
    from shapely.geometry import MultiPolygon
    comp = component_ids(r.tables, MultiPolygon([p for _, p in r.site.available]))
    for b in range(r.elec.n_inverters):
        assert len(np.unique(comp[r.elec.table_block == b])) == 1   # block stays in one field
    assert r.energy.shading_loss_frac > 0
    beta = r.tables.beta_deg
    assert 8 < beta < 12                                            # ±2 h winter sun at 52°N
    assert r.kpis["gcr"] == pytest.approx(r.tables.slope_length / r.tables.pitch, rel=1e-3)
    assert math.isfinite(r.energy.annual_kwh)
