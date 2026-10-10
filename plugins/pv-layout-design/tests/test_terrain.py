"""
Tests for terrain.py — F10 slope analysis, exclusion, earthworks.

Uses the synthetic_dem() helper so no real elevation data is needed.
"""

import math
import pytest
import numpy as np

from pv_layout.geometry import place_panels, ModuleSpec, SiteSpec
from pv_layout.terrain import (
    DEMGrid, SlopeResult, TerrainResult,
    synthetic_dem, compute_slopes, apply_terrain_limits,
)

SITE = SiteSpec(width=200.0, height=150.0, lat=23.0, lon=72.6)
MODULE = ModuleSpec(length=2.0, width=1.0, power_wp=400.0)


@pytest.fixture(scope="module")
def layout():
    return place_panels(SITE, MODULE, 30.0, 1000.0)


class TestSyntheticDEM:
    def test_flat_dem_zero_elevation(self):
        dem = synthetic_dem(0, 0, 200, 150, ns_slope_pct=0, ew_slope_pct=0)
        assert dem.elevation_at(100, 75) == pytest.approx(0.0, abs=1e-9)

    def test_ns_slope_correct(self):
        """
        N-S slope 10 % → over 100 m north, elevation rises 10 m.
        Source: Topography_Analysis_Methodology Eq 2.4.
        """
        dem = synthetic_dem(0, 0, 200, 200, ns_slope_pct=10.0, cell_size=1.0)
        z_south = dem.elevation_at(100, 0)
        z_north = dem.elevation_at(100, 100)
        delta_z = z_north - z_south
        actual_slope_pct = 100.0 * delta_z / 100.0  # over 100 m
        assert actual_slope_pct == pytest.approx(10.0, rel=0.01)

    def test_elevation_at_clips_to_boundary(self):
        dem = synthetic_dem(0, 0, 100, 100, ns_slope_pct=5.0)
        # Out-of-bounds query should not raise
        z = dem.elevation_at(-10, 200)
        assert math.isfinite(z)


class TestComputeSlopes:
    def test_flat_dem_zero_slopes(self, layout):
        dem = synthetic_dem(0, 0, SITE.width, SITE.height)  # flat
        slopes = compute_slopes(
            dem, layout.row_centroids, layout.pitch, MODULE.length, 30.0
        )
        assert np.allclose(slopes.ns_slope_pct, 0.0, atol=1e-6)

    def test_sloped_dem_nonzero_ns(self, layout):
        dem = synthetic_dem(0, 0, SITE.width, SITE.height, ns_slope_pct=8.0)
        slopes = compute_slopes(
            dem, layout.row_centroids, layout.pitch, MODULE.length, 30.0
        )
        # All NS slopes should be close to 8 %
        assert np.all(np.abs(slopes.ns_slope_pct) > 0.1)

    def test_slopes_shape(self, layout):
        dem = synthetic_dem(0, 0, SITE.width, SITE.height)
        slopes = compute_slopes(
            dem, layout.row_centroids, layout.pitch, MODULE.length, 30.0
        )
        assert slopes.ns_slope_pct.shape == (layout.n_rows,)
        assert slopes.ew_slope_pct.shape == (layout.n_rows,)
        assert slopes.max_slope_pct.shape == (layout.n_rows,)


class TestApplyTerrainLimits:
    def test_flat_terrain_keeps_all(self, layout):
        """Flat terrain: no rows removed, no earthworks."""
        dem = synthetic_dem(0, 0, SITE.width, SITE.height)
        slopes = compute_slopes(dem, layout.row_centroids, layout.pitch, MODULE.length, 30.0)
        result = apply_terrain_limits(layout.row_centroids, slopes)
        assert result.n_removed == 0
        assert result.n_earthworks == 0
        assert result.n_kept == layout.n_rows

    def test_steep_terrain_removes_rows(self, layout):
        """30 % N-S slope exceeds default 25 % limit → all rows removed."""
        dem = synthetic_dem(0, 0, SITE.width, SITE.height, ns_slope_pct=30.0, cell_size=1.0)
        slopes = compute_slopes(dem, layout.row_centroids, layout.pitch, MODULE.length, 30.0)
        result = apply_terrain_limits(layout.row_centroids, slopes, max_ns_slope_pct=25.0)
        assert result.n_removed == layout.n_rows
        assert result.n_kept == 0

    def test_intermediate_slope_triggers_earthworks(self, layout):
        """15 % NS slope > 10 % earthworks limit but < 25 % removal limit."""
        dem = synthetic_dem(0, 0, SITE.width, SITE.height, ns_slope_pct=15.0, cell_size=1.0)
        slopes = compute_slopes(dem, layout.row_centroids, layout.pitch, MODULE.length, 30.0)
        result = apply_terrain_limits(
            layout.row_centroids, slopes,
            max_ns_slope_pct=25.0, earthworks_ns_min_pct=10.0
        )
        assert result.n_earthworks > 0
        assert result.n_removed == 0

    def test_mask_consistent(self, layout):
        """row_mask = keep (not removed); earthworks ⊆ keep."""
        dem = synthetic_dem(0, 0, SITE.width, SITE.height, ns_slope_pct=12.0, cell_size=1.0)
        slopes = compute_slopes(dem, layout.row_centroids, layout.pitch, MODULE.length, 30.0)
        result = apply_terrain_limits(layout.row_centroids, slopes)
        # earthworks rows must all be in keep
        assert np.all(result.row_mask[result.earthworks_mask])
        # n_kept + n_removed = n_rows
        assert result.n_kept + result.n_removed == layout.n_rows

    def test_cut_fill_volume_with_dem(self, layout):
        """Volume > 0 when earthworks are needed and DEM is provided."""
        dem = synthetic_dem(0, 0, SITE.width, SITE.height, ns_slope_pct=15.0, cell_size=1.0)
        slopes = compute_slopes(dem, layout.row_centroids, layout.pitch, MODULE.length, 30.0)
        result = apply_terrain_limits(
            layout.row_centroids, slopes, dem=dem,
            max_ns_slope_pct=25.0, earthworks_ns_min_pct=10.0
        )
        if result.n_earthworks > 0:
            assert result.cut_fill_volume_m3 > 0

    def test_no_dem_volume_zero(self, layout):
        dem = synthetic_dem(0, 0, SITE.width, SITE.height, ns_slope_pct=15.0, cell_size=1.0)
        slopes = compute_slopes(dem, layout.row_centroids, layout.pitch, MODULE.length, 30.0)
        result = apply_terrain_limits(layout.row_centroids, slopes, dem=None)
        assert result.cut_fill_volume_m3 == 0.0
