import numpy as np
import pandas as pd
import pvlib
import pytest

from pv_layout.energy import annual_energy_yield, row_shade_fraction


def _clearsky(lat, lon):
    t = pd.date_range("2023-01-01", periods=8760, freq="h", tz="UTC")
    cs = pvlib.location.Location(lat, lon).get_clearsky(t)
    return pd.DataFrame({"ghi": cs["ghi"], "dni": cs["dni"], "dhi": cs["dhi"],
                         "temp_air": 20.0, "wind_speed": 2.0}, index=t)


def test_shade_fraction_geometry():
    # sun at zenith → never shaded; sun behind the array → 0 (no beam on front)
    assert row_shade_fraction(90, 180, 180, 25, 0.5) == 0
    assert row_shade_fraction(20, 0, 180, 25, 0.5) == 0
    # profile angle α with tan α = tan(10°): fs = 1 − sinα / (GCR · sin(α+β))
    fs = row_shade_fraction(10, 180, 180, 25, 0.5)
    a, b = np.radians(10), np.radians(25)
    assert fs == pytest.approx(1 - np.sin(a) / (0.5 * np.sin(a + b)))


def test_auto_azimuth_faces_equator():
    sh = annual_energy_yield(-34.6, 146.0, 1000, 30, azimuth_deg=None, tmy_data=_clearsky(-34.6, 146.0))
    wrong = annual_energy_yield(-34.6, 146.0, 1000, 30, azimuth_deg=180, tmy_data=_clearsky(-34.6, 146.0))
    assert sh.azimuth_deg == 0.0
    assert sh.annual_kwh > 1.3 * wrong.annual_kwh


def test_detailed_losses_reduce_yield_monotonically():
    tmy = _clearsky(52.3, 0.1)
    base = annual_energy_yield(52.3, 0.1, 1000, 25, azimuth_deg=None, tmy_data=tmy, pv_model="huld")
    iam = annual_energy_yield(52.3, 0.1, 1000, 25, azimuth_deg=None, tmy_data=tmy, pv_model="huld", iam_ar=0.16)
    shade = annual_energy_yield(52.3, 0.1, 1000, 25, azimuth_deg=None, tmy_data=tmy, pv_model="huld",
                                iam_ar=0.16, gcr=0.6, n_rows=20)
    assert base.annual_kwh > iam.annual_kwh > shade.annual_kwh
    assert 0.02 < iam.iam_loss_frac < 0.06
    assert shade.shading_loss_frac > 0.01


def test_clipping_and_transformer_hourly_chain():
    tmy = _clearsky(27.5, 71.9)
    kw = dict(azimuth_deg=None, tmy_data=tmy, pv_model="huld", ohmic_at_stc=True,
              dc_cable_loss_pct=1.0, ac_cable_loss_pct=0.5)
    loose = annual_energy_yield(27.5, 71.9, 1000, 23, ac_capacity_kw=1000, **kw)
    tight = annual_energy_yield(27.5, 71.9, 1000, 23, ac_capacity_kw=600, **kw)
    assert tight.clipping_loss_frac > 0.02 > loose.clipping_loss_frac
    # I²R scaled by load: annual DC loss below the 1 % STC value
    assert 0 < loose.dc_cable_loss_frac < 0.01
    tx = annual_energy_yield(27.5, 71.9, 1000, 23, transformer_losses=(1.0, 10.0, 1100), **kw)
    assert 0.002 < tx.transformer_loss_frac < 0.02


def test_horizon_blocks_beam():
    tmy = _clearsky(37.4, 140.2)
    hz = pd.Series(np.full(48, 15.0), index=np.arange(0, 360, 7.5))
    open_sky = annual_energy_yield(37.4, 140.2, 1000, 15, azimuth_deg=None, tmy_data=tmy)
    blocked = annual_energy_yield(37.4, 140.2, 1000, 15, azimuth_deg=None, tmy_data=tmy, horizon=hz)
    assert blocked.horizon_loss_frac > 0.03
    assert blocked.annual_kwh < open_sky.annual_kwh


def test_unknown_pv_model():
    with pytest.raises(ValueError, match="pv_model"):
        annual_energy_yield(10, 10, 100, 10, tmy_data=_clearsky(10, 10), pv_model="magic")
