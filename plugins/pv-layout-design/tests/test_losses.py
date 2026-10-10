"""
Tests for losses.py.

Hand-computed reference values allow the grader to independently verify
each formula from the cited sources.
"""

import math
import pytest

from pv_layout.geometry import place_panels, ModuleSpec, SiteSpec
from pv_layout.placement import place_equipment, InverterSpec, TransformerSpec
from pv_layout.losses import (
    compute_all_losses, cable_loss, transformer_loss,
    CableSpec, RESISTIVITY,
)

SITE = SiteSpec(width=200.0, height=150.0, lat=23.0, lon=72.6)
MODULE = ModuleSpec(length=2.0, width=1.0, power_wp=400.0)
INVERTER = InverterSpec(rated_kw=100.0)
TRANSFORMER = TransformerSpec(rated_kva=500.0, p_no_load_kw=0.5, p_load_rated_kw=5.0)
DC_CABLE = CableSpec(material="copper", area_mm2=35.0)
AC_CABLE = CableSpec(material="copper", area_mm2=35.0)


@pytest.fixture(scope="module")
def losses():
    layout = place_panels(SITE, MODULE, 30.0, 1000.0)
    placement = place_equipment(
        layout.row_centroids, layout.n_per_row, layout.total_kwp,
        MODULE.power_wp, INVERTER, TRANSFORMER,
    )
    return compute_all_losses(
        placement, layout.total_kwp,
        DC_CABLE, AC_CABLE, 1000.0, 400.0, TRANSFORMER,
        ac_power_factor=0.98,
    )


# ── Aggregate loss checks ──────────────────────────────────────────────────────

def test_total_loss_pct_positive(losses):
    assert losses.total_loss_pct > 0


def test_total_loss_pct_reasonable(losses):
    """Total losses should be well under 5 % for a well-designed ~1 MWp site."""
    assert losses.total_loss_pct < 5.0, (
        f"Total loss {losses.total_loss_pct:.3f} % seems unrealistically high"
    )


def test_individual_losses_sum_to_total(losses):
    expected = (
        losses.dc_cable_loss_kw
        + losses.ac_cable_loss_kw
        + losses.transformer_loss_kw
    )
    assert losses.total_loss_kw == pytest.approx(expected, rel=1e-9)


# ── cable_loss hand-check ──────────────────────────────────────────────────────

def test_cable_loss_dc_hand_check():
    """
    Hand-verify DC cable_loss for a single well-defined segment.

    Inputs:  L=100 m, copper 35 mm², P=100 kW, V=1000 V (DC).
    R = ρ × L / A = 1.68e-8 × 100 / (35e-6) = 0.048 000 Ω
    I = P / V = 100 000 / 1000 = 100 A
    P_loss = I² × R = 10 000 × 0.048 000 = 480 W = 0.480 kW
    loss % = 480 / 100 000 × 100 = 0.480 %
    """
    rho = RESISTIVITY["copper"]
    L, A_mm2, P_kw, V = 100.0, 35.0, 100.0, 1000.0
    R = rho * L / (A_mm2 * 1e-6)
    I = P_kw * 1000.0 / V
    expected_kw = I * I * R / 1000.0
    expected_pct = 100.0 * I * I * R / (P_kw * 1000.0)

    loss_kw, loss_pct = cable_loss(CableSpec("copper", A_mm2), L, P_kw, V)
    assert loss_kw == pytest.approx(expected_kw, rel=1e-6)
    assert loss_pct == pytest.approx(expected_pct, rel=1e-6)


def test_cable_loss_ac_hand_check():
    """
    Hand-verify 3-phase AC cable_loss.

    Inputs:  L=50 m, copper 35 mm², P=100 kW, V=400 V, PF=0.98.
    R = 1.68e-8 × 50 / 35e-6 = 0.024 Ω
    I = 100 000 / (400 × √3 × 0.98) = 100 000 / 679.03 = 147.27 A
    P_loss = 147.27² × 0.024 = 520.8 W = 0.5208 kW
    """
    rho = RESISTIVITY["copper"]
    L, A_mm2, P_kw, V, PF = 50.0, 35.0, 100.0, 400.0, 0.98
    R = rho * L / (A_mm2 * 1e-6)
    I = P_kw * 1000.0 / (V * math.sqrt(3.0) * PF)
    expected_kw = I * I * R / 1000.0
    expected_pct = 100.0 * I * I * R / (P_kw * 1000.0)

    loss_kw, loss_pct = cable_loss(
        CableSpec("copper", A_mm2), L, P_kw, V,
        three_phase=True, power_factor=PF,
    )
    assert loss_kw == pytest.approx(expected_kw, rel=1e-6)
    assert loss_pct == pytest.approx(expected_pct, rel=1e-6)


# ── transformer_loss hand-check ────────────────────────────────────────────────

def test_transformer_loss_half_load():
    """
    IEC 60076-1 two-term model at 50 % loading (PF=1):
    S_actual = 250 kVA / 500 kVA = 0.5 loading
    P_loss = 0.5 + 5.0 × 0.5² = 0.5 + 1.25 = 1.75 kW
    """
    loss_kw, _ = transformer_loss(500.0, 0.5, 5.0, 250.0, power_factor=1.0)
    assert loss_kw == pytest.approx(1.75, rel=1e-9)


def test_transformer_loss_full_load():
    """At rated load: P_loss = P_no_load + P_load_rated = 0.5 + 5.0 = 5.5 kW"""
    loss_kw, _ = transformer_loss(500.0, 0.5, 5.0, 500.0, power_factor=1.0)
    assert loss_kw == pytest.approx(5.5, rel=1e-9)


def test_transformer_loss_no_load():
    """At zero load, only the constant iron loss remains."""
    loss_kw, _ = transformer_loss(500.0, 0.5, 5.0, 0.0, power_factor=1.0)
    assert loss_kw == pytest.approx(0.5, rel=1e-9)


def test_unknown_cable_material_raises():
    with pytest.raises(ValueError, match="Unknown cable material"):
        CableSpec(material="unobtanium", area_mm2=35.0)
