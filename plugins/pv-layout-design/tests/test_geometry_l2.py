"""
Layer 2/3 tests for geometry.py: F3 tilt recommendation, F4 GCR input,
F1 polygon site, F2 setbacks.

Same Ahmedabad fixture as Layer 1 tests for cross-check.
"""

import math
import pytest
import numpy as np

from pv_layout.geometry import (
    recommend_tilt, place_panels, place_panels_polygon,
    apply_setback, ModuleSpec, SiteSpec,
)

SITE = SiteSpec(width=200.0, height=150.0, lat=23.0, lon=72.6)
MODULE = ModuleSpec(length=2.0, width=1.0, power_wp=400.0)
TILT_DEG = 30.0
TARGET_KWP = 1000.0


# ── F3: Latitude-based tilt recommendation ────────────────────────────────────

class TestRecommendTilt:
    """
    Hand-verify the RatedPower tilt table (Design_Methodology §5.3.3.4).
    """

    def test_below_5_deg(self):
        assert recommend_tilt(3.0) == 5.0
        assert recommend_tilt(0.0) == 5.0

    def test_5_to_10_deg(self):
        assert recommend_tilt(7.5) == 7.5
        assert recommend_tilt(5.0) == 5.0
        assert recommend_tilt(10.0) == 10.0

    def test_10_to_20_deg(self):
        # lat × 0.9, rounded to 1 dp
        assert recommend_tilt(15.0) == pytest.approx(13.5, rel=1e-9)
        assert recommend_tilt(20.0) == pytest.approx(18.0, rel=1e-9)

    def test_above_20_deg(self):
        # Ahmedabad 23° → 23 × 0.85 = 19.55 → 19.6
        assert recommend_tilt(23.0) == pytest.approx(19.6, abs=0.05)
        # London 51.5° → 51.5 × 0.85 = 43.775 → 43.8
        assert recommend_tilt(51.5) == pytest.approx(43.8, abs=0.05)

    def test_southern_hemisphere(self):
        # abs(lat) is used
        assert recommend_tilt(-23.0) == recommend_tilt(23.0)


# ── F4: Direct GCR input ──────────────────────────────────────────────────────

class TestGCRInput:
    """
    If gcr is given, pitch = module_length / gcr and beta_deg is NaN.
    Layer 1 tests must still pass (gcr=None default).
    """

    def test_gcr_sets_pitch(self):
        layout = place_panels(SITE, MODULE, TILT_DEG, TARGET_KWP, gcr=0.40)
        expected_pitch = MODULE.length / 0.40
        assert layout.pitch == pytest.approx(expected_pitch, rel=1e-9)

    def test_gcr_sets_gcr_field(self):
        layout = place_panels(SITE, MODULE, TILT_DEG, TARGET_KWP, gcr=0.40)
        assert layout.gcr == pytest.approx(0.40, rel=1e-9)

    def test_gcr_beta_is_nan(self):
        layout = place_panels(SITE, MODULE, TILT_DEG, TARGET_KWP, gcr=0.40)
        assert math.isnan(layout.beta_deg)

    def test_gcr_rows_fit(self):
        layout = place_panels(SITE, MODULE, TILT_DEG, TARGET_KWP, gcr=0.40)
        assert layout.n_rows > 0
        assert layout.total_kwp >= TARGET_KWP

    def test_gcr_invalid_raises(self):
        with pytest.raises(ValueError):
            place_panels(SITE, MODULE, TILT_DEG, TARGET_KWP, gcr=1.5)
        with pytest.raises(ValueError):
            place_panels(SITE, MODULE, TILT_DEG, TARGET_KWP, gcr=0.0)

    def test_none_gcr_unchanged(self):
        """With gcr=None (default) the Layer 1 path runs unchanged."""
        layout = place_panels(SITE, MODULE, TILT_DEG, TARGET_KWP)
        assert not math.isnan(layout.beta_deg)
        assert layout.beta_deg > 0


# ── F1: Polygon site ──────────────────────────────────────────────────────────

try:
    from shapely.geometry import box as shapely_box, Polygon
    SHAPELY_OK = True
except ImportError:
    SHAPELY_OK = False

pytestmark_shapely = pytest.mark.skipif(
    not SHAPELY_OK, reason="shapely not installed"
)


@pytest.mark.skipif(not SHAPELY_OK, reason="shapely not installed")
class TestPolygonSite:
    """
    Rectangle polygon must reproduce the same layout as place_panels.
    L-shaped polygon must give fewer rows than full rectangle.
    """

    def test_rectangle_matches_place_panels(self):
        rect = shapely_box(0, 0, SITE.width, SITE.height)
        poly_layout = place_panels_polygon(
            rect, SITE.lat, SITE.lon, MODULE, TILT_DEG, TARGET_KWP
        )
        rect_layout = place_panels(SITE, MODULE, TILT_DEG, TARGET_KWP)
        # Same pitch (same sun angle, same lat/lon)
        assert poly_layout.pitch == pytest.approx(rect_layout.pitch, rel=1e-6)
        # Same total capacity (within one row's kWp)
        assert abs(poly_layout.total_kwp - rect_layout.total_kwp) <= (
            rect_layout.n_per_row * MODULE.power_wp / 1000.0
        )

    def test_l_shape_fewer_modules_than_full_rectangle(self):
        """
        For an impossibly large target (target_fits=False for both) no
        trimming occurs, so the L-shape places fewer modules than the full
        rectangle because it has less area.
        """
        l_shape = Polygon([
            (0, 0), (200, 0), (200, 75), (100, 75), (100, 150), (0, 150), (0, 0)
        ])
        # 9999 kWp cannot fit in either site → no trimming, all rows placed
        poly_layout = place_panels_polygon(
            l_shape, SITE.lat, SITE.lon, MODULE, TILT_DEG, 9999.0
        )
        rect_layout = place_panels(SITE, MODULE, TILT_DEG, 9999.0)
        # L-shape loses ~25 % area → fewer total modules
        assert poly_layout.total_modules < rect_layout.total_modules

    def test_polygon_centroids_inside_boundary(self):
        rect = shapely_box(0, 0, SITE.width, SITE.height)
        layout = place_panels_polygon(
            rect, SITE.lat, SITE.lon, MODULE, TILT_DEG, TARGET_KWP
        )
        for cx, cy in layout.row_centroids:
            assert 0 <= cx <= SITE.width
            assert 0 <= cy <= SITE.height

    def test_n_per_row_array_populated(self):
        rect = shapely_box(0, 0, SITE.width, SITE.height)
        layout = place_panels_polygon(
            rect, SITE.lat, SITE.lon, MODULE, TILT_DEG, TARGET_KWP
        )
        assert layout.n_per_row_array is not None
        assert len(layout.n_per_row_array) == layout.n_rows

    def test_gcr_kwarg_works_with_polygon(self):
        rect = shapely_box(0, 0, SITE.width, SITE.height)
        layout = place_panels_polygon(
            rect, SITE.lat, SITE.lon, MODULE, TILT_DEG, TARGET_KWP, gcr=0.40
        )
        assert layout.gcr == pytest.approx(0.40, rel=1e-9)


@pytest.mark.skipif(not SHAPELY_OK, reason="shapely not installed")
class TestSetbacks:
    """F2: setback_m shrinks the usable area, reducing panel count."""

    def test_setback_reduces_row_width(self):
        """
        A 5 m setback on a 200 m wide site narrows each row from 200 to
        190 modules.  The total module count may be slightly higher (more
        rows needed to meet the same kWp target), but each individual row
        is shorter.  The meaningful check is n_per_row.
        """
        rect = shapely_box(0, 0, SITE.width, SITE.height)
        no_sb = place_panels_polygon(
            rect, SITE.lat, SITE.lon, MODULE, TILT_DEG, TARGET_KWP
        )
        with_sb = place_panels_polygon(
            rect, SITE.lat, SITE.lon, MODULE, TILT_DEG, TARGET_KWP,
            setback_m=5.0
        )
        # Setback shrinks row width: 200 m − 2×5 m = 190 m → 190 modules/row
        assert with_sb.n_per_row < no_sb.n_per_row

    def test_setback_too_large_raises(self):
        small = shapely_box(0, 0, 10, 10)
        with pytest.raises(ValueError):
            apply_setback(small, setback_m=20.0)

    def test_zero_setback_unchanged(self):
        rect = shapely_box(0, 0, SITE.width, SITE.height)
        from shapely.geometry import box as shapely_box2
        inner = apply_setback(rect, 0.0)
        assert inner.equals(rect)
