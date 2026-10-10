"""
Tests for placement.py.

Same Ahmedabad fixture as test_geometry.py.  The key acceptance criterion
from the task brief (§7.3): k-means placement must give a shorter total DC
cable length than the naive baseline of putting every inverter at the
south-west corner (0, 0).
"""

import pytest
import numpy as np

from pv_layout.geometry import place_panels, ModuleSpec, SiteSpec
from pv_layout.placement import place_equipment, InverterSpec, TransformerSpec

SITE = SiteSpec(width=200.0, height=150.0, lat=23.0, lon=72.6)
MODULE = ModuleSpec(length=2.0, width=1.0, power_wp=400.0)
INVERTER = InverterSpec(rated_kw=100.0)
TRANSFORMER = TransformerSpec(rated_kva=500.0, p_no_load_kw=0.5, p_load_rated_kw=5.0)


@pytest.fixture(scope="module")
def layout():
    return place_panels(SITE, MODULE, 30.0, 1000.0)


@pytest.fixture(scope="module")
def placement(layout):
    return place_equipment(
        layout.row_centroids, layout.n_per_row, layout.total_kwp,
        MODULE.power_wp, INVERTER, TRANSFORMER,
    )


# ── Inverters ──────────────────────────────────────────────────────────────────

def test_inverter_count_positive(placement):
    assert placement.n_inverters >= 1


def test_inverter_positions_shape(placement):
    assert placement.inverter_positions.shape == (placement.n_inverters, 2)


def test_all_blocks_assigned(placement):
    assert len(placement.block_to_inverter) == len(placement.block_centroids)


def test_inverter_labels_in_range(placement):
    assert placement.block_to_inverter.min() >= 0
    assert placement.block_to_inverter.max() < placement.n_inverters


# ── Transformer ────────────────────────────────────────────────────────────────

def test_transformer_count(placement):
    assert placement.n_transformers == 1


def test_transformer_positions_shape(placement):
    assert placement.transformer_positions.shape == (1, 2)


# ── Cable lengths ──────────────────────────────────────────────────────────────

def test_dc_cable_lengths_positive(placement):
    assert placement.total_dc_cable_m > 0


def test_ac_cable_lengths_positive(placement):
    assert placement.total_ac_cable_m > 0


def test_total_dc_matches_sum(placement):
    assert placement.total_dc_cable_m == pytest.approx(
        placement.dc_cable_lengths_m.sum(), rel=1e-9
    )


def test_total_ac_matches_sum(placement):
    assert placement.total_ac_cable_m == pytest.approx(
        placement.ac_cable_lengths_m.sum(), rel=1e-9
    )


def test_kmeans_beats_corner_baseline(placement):
    """
    Acceptance criterion (brief §7.3): k-means placement reduces total DC
    cable vs. the naive baseline of placing all inverters at corner (0, 0).
    """
    naive_dc = float(np.linalg.norm(placement.block_centroids, axis=1).sum())
    assert placement.total_dc_cable_m < naive_dc, (
        f"k-means DC cable ({placement.total_dc_cable_m:.1f} m) is not shorter "
        f"than corner-at-origin naive ({naive_dc:.1f} m)"
    )
