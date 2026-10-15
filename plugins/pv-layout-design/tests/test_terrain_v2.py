import numpy as np
import pytest

from pv_layout.terrain import (
    ACCEPTED, EARTHWORKS, EXCLUDED, DEMGrid, SlopeLimits, classify_tables, smooth_dem, synthetic_dem,
)


def test_vectorised_matches_scalar():
    dem = synthetic_dem(0, 0, 200, 200, cell_size=5, ns_slope_pct=7, ew_slope_pct=3, undulation_m=1.0)
    rng = np.random.default_rng(1)
    xs, ys = rng.uniform(-10, 210, 50), rng.uniform(-10, 210, 50)
    np.testing.assert_allclose(dem.elevations_at(xs, ys), [dem.elevation_at(x, y) for x, y in zip(xs, ys)])


def test_plane_slopes_recovered_exactly():
    dem = synthetic_dem(-50, -50, 400, 400, cell_size=5, ns_slope_pct=8.0, ew_slope_pct=-3.0)
    centers = np.array([[100.0, 100.0], [150.0, 220.0]])
    t = classify_tables(dem, centers, 30.0, 4.3)
    np.testing.assert_allclose(t.ns_slope_pct, 8.0, atol=1e-9)
    np.testing.assert_allclose(t.ew_slope_pct, -3.0, atol=1e-9)


@pytest.mark.parametrize("ns,expected", [(5, ACCEPTED), (12, EARTHWORKS), (18, EXCLUDED)])
def test_classification_thresholds(ns, expected):
    dem = synthetic_dem(-50, -50, 300, 300, cell_size=5, ns_slope_pct=ns)
    t = classify_tables(dem, np.array([[100.0, 100.0]]), 30.0, 4.3, SlopeLimits(10, 15, 5, 10))
    assert t.status[0] == expected


def test_earthworks_volume_balanced_on_plane():
    dem = synthetic_dem(-50, -50, 300, 300, cell_size=5, ns_slope_pct=12)
    t = classify_tables(dem, np.array([[100.0, 100.0]]), 30.0, 4.3, SlopeLimits(10, 15, 5, 10))
    # grading a 12 % plane to 10 % about the centre: cut = fill = area × Δslope × depth/4 (approx)
    assert t.cut_m3[0] > 0
    assert t.cut_m3[0] == pytest.approx(t.fill_m3[0], rel=1e-9)
    assert t.cut_m3[0] == pytest.approx(30 * 4.3 * 0.02 * 4.3 / 4, rel=0.35)


def test_flat_tables_no_earthworks():
    dem = synthetic_dem(-50, -50, 300, 300, cell_size=5)
    t = classify_tables(dem, np.array([[100.0, 100.0]]), 30.0, 4.3)
    assert t.status[0] == ACCEPTED and t.cut_m3[0] == 0 and t.fill_m3[0] == 0


def test_nan_cells_treated_as_accepted():
    Z = np.zeros((40, 40))
    Z[:, :] = np.nan
    t = classify_tables(DEMGrid(Z, 0, 0, 5), np.array([[50.0, 50.0]]), 30, 4)
    assert t.nan_tables == 1 and t.status[0] == ACCEPTED


def test_limits_validation():
    with pytest.raises(ValueError):
        SlopeLimits(ns_install_pct=20, ns_earthworks_pct=10)


def test_smoothing_removes_quantisation_steps():
    xs = np.arange(0, 300, 10.0)
    Z = np.floor(np.tile(xs * 0.03, (30, 1)))           # 3 % slope stored as integer metres
    dem = smooth_dem(DEMGrid(Z, 0, 0, 10), 30)
    ns, ew = dem.slope_grids_pct()
    inner = ew[5:-5, 5:-5]
    assert np.abs(inner - 3.0).max() < 1.0
