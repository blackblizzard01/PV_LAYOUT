import numpy as np
import pytest
from shapely.geometry import MultiPolygon, box

from pv_layout.blocks import (
    ElectricalSpec, bisect_partition, build_electrical, manhattan_route, partition_by_capacity,
)
from pv_layout.geometry import ModuleSpec
from pv_layout.tables import StructureSpec, layout_tables

MOD = ModuleSpec(length=2.278, width=1.134, power_wp=550.0)


def test_bisect_balanced_counts():
    rng = np.random.default_rng(0)
    pts = rng.uniform(0, 100, (101, 2))
    lab = bisect_partition(pts, np.ones(101), 7)
    sizes = np.bincount(lab)
    assert len(sizes) == 7 and sizes.max() - sizes.min() <= 2


def test_partition_respects_cap_with_indivisible_items():
    pts = np.column_stack([np.arange(100.0), np.zeros(100)])
    lab = partition_by_capacity(pts, np.full(100, 29.7), 472.5)
    sums = np.bincount(lab, weights=np.full(100, 29.7))
    assert sums.max() <= 472.5 + 1e-9
    assert len(sums) <= int(np.ceil(100 / 15)) + 1        # close to the 15-per-block optimum


def test_partition_never_spans_components():
    pts = np.vstack([np.zeros((10, 2)), np.full((10, 2), 1000.0)])
    comps = np.array([0] * 10 + [1] * 10)
    lab = partition_by_capacity(pts, np.ones(20), 30, comps)
    for g in np.unique(lab):
        assert len(np.unique(comps[lab == g])) == 1


def test_manhattan_route_length():
    pts, L = manhattan_route((0, 0), (10, 5), 2.0)
    assert L == pytest.approx(2 + 10 + 3)
    assert L >= np.hypot(10, 5)


def _layout(w=600, h=300):
    st = StructureSpec(modules_high=2, modules_along=27)
    return layout_tables(MultiPolygon([box(0, 0, w, h)]), MOD, st, 20.0, 23.0, 72.6)


@pytest.mark.parametrize("kind", ["string", "central"])
def test_build_electrical_invariants(kind):
    lay = _layout()
    spec = ElectricalSpec(inverter_type=kind, inverter_ac_kw=300 if kind == "string" else 2500,
                          dc_ac_ratio=1.25, ps_kva=3300 if kind == "string" else 2750)
    el = build_electrical(lay, 550.0, 27, spec)
    table_kwp = 54 * 0.55
    assert el.inverter_dc_kwp.max() <= spec.inverter_ac_kw * spec.dc_ac_ratio + 1e-6
    assert el.inverter_dc_kwp.sum() == pytest.approx(el.layout.n_tables * table_kwp)
    # every table in exactly one block; PS footprints clear of tables
    assert len(el.table_block) == el.layout.n_tables
    for poly in el.ps_polys:
        assert not any(b.intersects(poly) for b in el.layout.boxes())
    for r in el.routes:
        assert r.length_m + 1e-9 >= np.hypot(*(r.points[-1] - r.points[0]))
    if kind == "central":
        assert len(el.combiner_xy) > 0 and not el.routes_of("lv_ac")
        assert el.n_power_stations == el.n_inverters
    else:
        inv_per_ps = int(spec.ps_kva * spec.power_factor // spec.inverter_ac_kw)
        assert np.bincount(el.inverter_ps).max() <= inv_per_ps


def test_table_larger_than_block_rejected():
    lay = _layout()
    with pytest.raises(ValueError, match="exceeds an inverter block"):
        build_electrical(lay, 550.0, 27, ElectricalSpec(inverter_ac_kw=10, ps_kva=100))


def test_strings_must_tile_table():
    lay = _layout()
    with pytest.raises(ValueError, match="whole number"):
        build_electrical(lay, 550.0, 25, ElectricalSpec())
