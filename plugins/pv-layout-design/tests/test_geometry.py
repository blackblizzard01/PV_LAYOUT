"""
Tests for geometry.py.

Fixture: Ahmedabad site (23.0°N, 72.6°E), 200 × 150 m, 400 Wp module
(2.0 × 1.0 m), 30° tilt, 1 000 kWp target.  All expected values are
hand-checkable against the formulas in geometry.py.
"""

import math
import pytest
import numpy as np

from pv_layout.geometry import (
    place_panels, compute_pitch, compute_gcr,
    ModuleSpec, SiteSpec,
)

# ── Fixture constants ──────────────────────────────────────────────────────────
SITE = SiteSpec(width=200.0, height=150.0, lat=23.0, lon=72.6)
MODULE = ModuleSpec(length=2.0, width=1.0, power_wp=400.0)
TILT_DEG = 30.0
TARGET_KWP = 1000.0


@pytest.fixture(scope="module")
def layout():
    return place_panels(SITE, MODULE, TILT_DEG, TARGET_KWP)


# ── GCR and pitch ──────────────────────────────────────────────────────────────

def test_gcr_in_realistic_range(layout):
    """GCR must fall in the utility-scale range 0.30–0.55."""
    assert 0.30 <= layout.gcr <= 0.55, (
        f"GCR = {layout.gcr:.3f} is outside the realistic 0.30–0.55 range"
    )


def test_gcr_formula():
    """compute_gcr(2.0, 5.0) == 0.4 exactly."""
    assert compute_gcr(2.0, 5.0) == pytest.approx(0.4, rel=1e-9)


def test_pitch_formula_hand_check():
    """
    Hand-verify pitch for L=2 m, tilt=30°, β=15°.
    pitch = 2·(cos30° + sin30°/tan15°)
          = 2·(0.86603 + 0.5/0.26795) = 2·(0.86603 + 1.86603) = 5.464 m
    """
    tilt, beta = 30.0, 15.0
    expected = 2.0 * (
        math.cos(math.radians(tilt))
        + math.sin(math.radians(tilt)) / math.tan(math.radians(beta))
    )
    assert compute_pitch(2.0, tilt, beta) == pytest.approx(expected, rel=1e-9)


def test_pitch_positive(layout):
    assert layout.pitch > 0


def test_beta_positive(layout):
    """Sun elevation at ±4 h from noon on winter solstice at 23°N must be positive."""
    assert layout.beta_deg > 0


# ── Panel count and capacity ───────────────────────────────────────────────────

def test_target_fits(layout):
    """1 000 kWp must fit in a 200 × 150 m site at 23°N."""
    assert layout.target_fits


def test_total_kwp_at_least_target(layout):
    assert layout.total_kwp >= TARGET_KWP


def test_total_modules_consistent(layout):
    assert layout.total_modules == layout.n_rows * layout.n_per_row


def test_total_kwp_consistent(layout):
    expected = layout.total_modules * MODULE.power_wp / 1000.0
    assert layout.total_kwp == pytest.approx(expected, rel=1e-9)


# ── Row centroid positions ─────────────────────────────────────────────────────

def test_row_centroids_shape(layout):
    assert layout.row_centroids.shape == (layout.n_rows, 2)


def test_row_centroids_inside_site(layout):
    xs = layout.row_centroids[:, 0]
    ys = layout.row_centroids[:, 1]
    assert np.all(xs >= 0) and np.all(xs <= SITE.width)
    assert np.all(ys >= 0) and np.all(ys <= SITE.height)


def test_row_centroids_x_centred(layout):
    """All row centroids share the same x (east-west centre of the site)."""
    assert np.allclose(layout.row_centroids[:, 0], SITE.width / 2.0)


def test_row_spacing_equals_pitch(layout):
    """Consecutive row centroids are separated by exactly one pitch in y."""
    ys = layout.row_centroids[:, 1]
    diffs = np.diff(ys)
    assert np.allclose(diffs, layout.pitch, rtol=1e-6)
