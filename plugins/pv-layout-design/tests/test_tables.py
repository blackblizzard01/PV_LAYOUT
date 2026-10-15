import math

import numpy as np
import pytest
import shapely
from shapely.geometry import MultiPolygon, Polygon, box

from pv_layout.geometry import ModuleSpec
from pv_layout.tables import (
    StructureSpec, compute_table_pitch, layout_tables, trim_mask,
)

MOD = ModuleSpec(length=2.278, width=1.134, power_wp=550.0)
ST = StructureSpec(modules_high=2, modules_along=28)


def test_structure_dims_portrait_and_landscape():
    w, L = ST.dims(MOD)
    assert w == pytest.approx(28 * 1.134 + 27 * 0.02)
    assert L == pytest.approx(2 * 2.278 + 0.02)
    w2, L2 = StructureSpec(modules_high=4, modules_along=27, orientation="landscape").dims(MOD)
    assert w2 == pytest.approx(27 * 2.278 + 26 * 0.02)
    assert L2 == pytest.approx(4 * 1.134 + 3 * 0.02)


def test_bad_structure_rejected():
    with pytest.raises(ValueError):
        StructureSpec(orientation="diagonal")


def test_table_pitch_formula_and_gcr_input():
    L = 4.576
    pitch, gcr, beta = compute_table_pitch(L, 20.0, 23.0, 72.6)
    assert pitch == pytest.approx(L * (math.cos(math.radians(20)) + math.sin(math.radians(20)) / math.tan(math.radians(beta))))
    assert gcr == pytest.approx(L / pitch)
    p2, g2, b2 = compute_table_pitch(L, 20.0, 23.0, 72.6, gcr=0.4)
    assert p2 == pytest.approx(L / 0.4) and math.isnan(b2)
    with pytest.raises(ValueError):
        compute_table_pitch(L, 20.0, 23.0, 72.6, gcr=1.2)


def test_tables_fully_inside_slanted_polygon():
    # Trapezoid with slanted east edge: centre-line clipping would overhang it.
    area = MultiPolygon([Polygon([(0, 0), (300, 0), (220, 200), (0, 200)])])
    lay = layout_tables(area, MOD, ST, 20.0, 23.0, 72.6)
    assert lay.n_tables > 0
    assert shapely.covered_by(lay.boxes(), area).all()
    # no two tables overlap
    b = lay.boxes()
    tree = shapely.STRtree(b)
    a_idx, b_idx = tree.query(b, predicate="intersects")
    pairs = a_idx != b_idx
    if pairs.any():
        inter = shapely.area(shapely.intersection(b[a_idx[pairs]], b[b_idx[pairs]]))
        assert inter.max() < 1e-6


def test_rows_spaced_by_pitch():
    area = MultiPolygon([box(0, 0, 200, 150)])
    lay = layout_tables(area, MOD, ST, 20.0, 23.0, 72.6)
    ys = np.unique(np.round(lay.centers[:, 1], 9))
    np.testing.assert_allclose(np.diff(ys), lay.pitch, rtol=1e-8)


def test_tiny_area_reports_diagnostics():
    with pytest.raises(ValueError, match="No table fits"):
        layout_tables(MultiPolygon([box(0, 0, 20, 3)]), MOD, ST, 20.0, 23.0, 72.6)


def test_trim_keeps_central_tables():
    area = MultiPolygon([box(0, 0, 400, 300)])
    lay = layout_tables(area, MOD, ST, 20.0, 23.0, 72.6)
    kwp_table = ST.modules_per_table * 0.55
    m = trim_mask(lay, 10 * kwp_table, 550.0)
    assert m.sum() == 10
    c = lay.centers.mean(axis=0)
    d = np.linalg.norm(lay.centers - c, axis=1)
    assert d[m].max() <= d[~m].min() + 1e-9
    assert trim_mask(lay, 1e9, 550.0).all()
