"""
Tests for boq.py — F8 Bill of Quantities.
"""

import pytest
from pv_layout.geometry import place_panels, ModuleSpec, SiteSpec
from pv_layout.placement import place_equipment, InverterSpec, TransformerSpec
from pv_layout.losses import CableSpec
from pv_layout.boq import bill_of_quantities, BoQResult

SITE = SiteSpec(width=200.0, height=150.0, lat=23.0, lon=72.6)
MODULE = ModuleSpec(length=2.0, width=1.0, power_wp=400.0)
INVERTER = InverterSpec(rated_kw=100.0)
TRANSFORMER = TransformerSpec(rated_kva=500.0, p_no_load_kw=0.5, p_load_rated_kw=5.0)
DC_CABLE = CableSpec(material="copper", area_mm2=35.0)
AC_CABLE = CableSpec(material="copper", area_mm2=35.0)


@pytest.fixture(scope="module")
def boq():
    layout = place_panels(SITE, MODULE, 30.0, 1000.0)
    placement = place_equipment(
        layout.row_centroids, layout.n_per_row, layout.total_kwp,
        MODULE.power_wp, INVERTER, TRANSFORMER,
    )
    return bill_of_quantities(layout.total_modules, layout.total_kwp,
                               placement, DC_CABLE, AC_CABLE)


def test_boq_returns_result(boq):
    assert isinstance(boq, BoQResult)


def test_total_cable_is_sum(boq):
    assert boq.total_cable_m == pytest.approx(
        boq.dc_cable_total_m + boq.ac_cable_total_m, rel=1e-9
    )


def test_cable_mass_positive(boq):
    assert boq.dc_cable_mass_kg > 0
    assert boq.ac_cable_mass_kg > 0


def test_trench_volume_formula():
    """
    Hand-check: vol = total_cable_m × depth × width.
    Source: Bill_of_Quantities_Methodology §3 LV trench rules.
    """
    layout = place_panels(SITE, MODULE, 30.0, 1000.0)
    placement = place_equipment(
        layout.row_centroids, layout.n_per_row, layout.total_kwp,
        MODULE.power_wp, INVERTER, TRANSFORMER,
    )
    depth, width = 1.2, 0.5
    boq = bill_of_quantities(layout.total_modules, layout.total_kwp,
                              placement, DC_CABLE, AC_CABLE,
                              trench_depth_m=depth, trench_width_m=width)
    expected_vol = boq.total_cable_m * depth * width
    assert boq.trench_volume_m3 == pytest.approx(expected_vol, rel=1e-9)


def test_control_building_small_site(boq):
    """< 50 MWp → 150 m² control building."""
    assert boq.control_building_m2 == 150.0


def test_equipment_counts(boq):
    layout = place_panels(SITE, MODULE, 30.0, 1000.0)
    assert boq.n_modules == layout.total_modules
    assert boq.n_inverters >= 1
    assert boq.n_transformers == 1
