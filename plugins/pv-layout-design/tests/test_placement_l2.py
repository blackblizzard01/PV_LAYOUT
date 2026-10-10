"""
Layer 2 tests for placement.py — F7 multi-transformer.
All Layer 1 placement tests must still pass (tested in test_placement.py).
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


class TestMultiTransformer:
    """F7: n_transformers > 1 places multiple transformers via k-means."""

    def test_default_still_one_transformer(self, layout):
        """Layer 1 contract: default n_transformers=1."""
        p = place_equipment(
            layout.row_centroids, layout.n_per_row, layout.total_kwp,
            MODULE.power_wp, INVERTER, TRANSFORMER,
        )
        assert p.n_transformers == 1

    def test_two_transformers_shape(self, layout):
        p = place_equipment(
            layout.row_centroids, layout.n_per_row, layout.total_kwp,
            MODULE.power_wp, INVERTER, TRANSFORMER, n_transformers=2,
        )
        assert p.n_transformers == 2
        assert p.transformer_positions.shape == (2, 2)

    def test_inverter_to_transformer_mapping(self, layout):
        """Each inverter must be assigned to a valid transformer index."""
        p = place_equipment(
            layout.row_centroids, layout.n_per_row, layout.total_kwp,
            MODULE.power_wp, INVERTER, TRANSFORMER, n_transformers=2,
        )
        assert p.inverter_to_transformer.min() >= 0
        assert p.inverter_to_transformer.max() < p.n_transformers

    def test_ac_cable_uses_correct_transformer(self, layout):
        """
        AC cable lengths must equal distance from each inverter to its
        assigned transformer (not necessarily transformer 0).
        """
        p = place_equipment(
            layout.row_centroids, layout.n_per_row, layout.total_kwp,
            MODULE.power_wp, INVERTER, TRANSFORMER, n_transformers=2,
        )
        for i in range(p.n_inverters):
            tx_idx = p.inverter_to_transformer[i]
            expected = float(np.linalg.norm(
                p.inverter_positions[i] - p.transformer_positions[tx_idx]
            ))
            assert p.ac_cable_lengths_m[i] == pytest.approx(expected, rel=1e-9)

    def test_n_transformers_capped_at_n_inverters(self, layout):
        """Requesting more transformers than inverters is silently capped."""
        p1 = place_equipment(
            layout.row_centroids, layout.n_per_row, layout.total_kwp,
            MODULE.power_wp, INVERTER, TRANSFORMER, n_transformers=1,
        )
        p_big = place_equipment(
            layout.row_centroids, layout.n_per_row, layout.total_kwp,
            MODULE.power_wp, INVERTER, TRANSFORMER, n_transformers=999,
        )
        assert p_big.n_transformers <= p1.n_inverters

    def test_two_transformers_shorter_ac_than_one(self, layout):
        """
        Two transformers should give shorter or equal total AC cable than one,
        because each inverter is closer to its nearest transformer.
        """
        p1 = place_equipment(
            layout.row_centroids, layout.n_per_row, layout.total_kwp,
            MODULE.power_wp, INVERTER, TRANSFORMER, n_transformers=1,
        )
        p2 = place_equipment(
            layout.row_centroids, layout.n_per_row, layout.total_kwp,
            MODULE.power_wp, INVERTER, TRANSFORMER, n_transformers=2,
        )
        assert p2.total_ac_cable_m <= p1.total_ac_cable_m + 1.0  # ≤ with small tolerance
