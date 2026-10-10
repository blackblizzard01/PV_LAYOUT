"""
Tests for energy.py — F9 annual energy yield.

Network access is NOT required: tests use synthetic TMY data generated
by pvlib's location object, so they always run offline.
"""

import pytest
import pandas as pd
import numpy as np
import pvlib

from pv_layout.energy import annual_energy_yield, EnergyResult


def _make_synthetic_tmy(lat: float = 23.0, lon: float = 72.6) -> pd.DataFrame:
    """
    Generate a synthetic hourly TMY for one year using pvlib's clear-sky model.
    Returns a DataFrame with ghi, dhi, dni, temp_air, wind_speed.
    This avoids any network call and satisfies the energy module's input contract.
    """
    times = pd.date_range("2023-01-01", periods=8760, freq="h", tz="UTC")
    loc = pvlib.location.Location(lat, lon, tz="UTC")
    cs = loc.get_clearsky(times, model="ineichen")
    df = pd.DataFrame(index=times)
    df["ghi"] = cs["ghi"].values
    df["dhi"] = cs["dhi"].values
    df["dni"] = cs["dni"].values
    df["temp_air"] = 25.0 + 5.0 * np.sin(np.linspace(0, 2 * np.pi, 8760))
    df["wind_speed"] = 3.0
    return df


@pytest.fixture(scope="module")
def result():
    tmy = _make_synthetic_tmy()
    return annual_energy_yield(
        lat=23.0, lon=72.6, total_kwp=1040.0, tilt_deg=30.0,
        dc_cable_loss_pct=0.003, ac_cable_loss_pct=0.22,
        transformer_loss_pct=2.21,
        tmy_data=tmy,
    )


def test_returns_energy_result(result):
    assert isinstance(result, EnergyResult)


def test_annual_kwh_positive(result):
    assert result.annual_kwh > 0


def test_specific_yield_realistic(result):
    """
    Ahmedabad (23°N) clear-sky: specific yield should be within 700–2500 kWh/kWp/yr.
    (Wide range because clear-sky is optimistic; real TMY would be ~1400–1600.)
    """
    assert 700.0 < result.specific_yield_kwh_kwp < 2500.0


def test_pr_between_0_and_1(result):
    """PR < 1 always for a system with losses."""
    assert 0.0 < result.performance_ratio < 1.0


def test_loss_fractions_between_0_and_1(result):
    for name in [
        "soiling_loss_frac", "mismatch_loss_frac", "lid_loss_frac",
        "temperature_loss_frac", "dc_cable_loss_frac", "inverter_loss_frac",
        "ac_cable_loss_frac", "transformer_loss_frac", "availability_loss_frac",
    ]:
        val = getattr(result, name)
        assert 0.0 <= val <= 1.0, f"{name} = {val}"


def test_cable_losses_match_input(result):
    """The dc/ac/transformer loss fractions must reflect the input percentages."""
    assert result.dc_cable_loss_frac == pytest.approx(0.003 / 100, rel=1e-9)
    assert result.ac_cable_loss_frac == pytest.approx(0.22 / 100, rel=1e-9)
    assert result.transformer_loss_frac == pytest.approx(2.21 / 100, rel=1e-9)


def test_poa_positive(result):
    assert result.poa_annual_kwh_m2 > 0


def test_zero_kwp_raises_no_error():
    """Edge case: zero capacity gives zero energy."""
    tmy = _make_synthetic_tmy()
    r = annual_energy_yield(lat=23.0, lon=72.6, total_kwp=0.0, tilt_deg=30.0, tmy_data=tmy)
    assert r.annual_kwh == 0.0
