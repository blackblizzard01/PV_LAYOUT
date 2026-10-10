"""
Annual energy yield: Performance Ratio (PR) and specific yield (kWh/kWp/yr).

The base tool's cable and transformer losses feed into this module as
itemized loss-chain terms, replacing the flat derating_factor that OptimaRE
currently uses.

Sources
-------
- Full loss chain and PR definition: RatedPower Energy_Yield_Methodology §10
  (Table 10.1–10.4), reference/RatedPower_Energy_Yield_Methodology.txt.
- Irradiance transposition: pvlib.irradiance.get_total_irradiance,
  using the Perez diffuse model.
  Source: Energy_Yield_Methodology §3, Eq 3.1–3.3.
- Cell temperature (Faiman model):
  T_cell = T_amb + G × (NOCT − 20) / 800
  Source: Energy_Yield_Methodology §6, Eq 6.15 (simplified single-term).
  Faiman (1997), Solar Energy 59(1-3):19-25.
- Module output temperature correction:
  P_rel(G, T) = (G / G_stc) × [1 + γ_pmax × (T_cell − 25)]
  Source: Energy_Yield_Methodology §6 (one-diode model; simplified here to
  linear temperature correction only).
- PVGIS TMY data: pvlib.iotools.get_pvgis_tmy (BSD-3).
  RatedPower uses NASA-POWER, NASA-SSE, PVGIS-5.2; Design_Methodology §5.2.3.
- Default loss-factor values: Energy_Yield_Methodology Table 10.1–10.4.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import pvlib


@dataclass
class EnergyResult:
    """
    Annual energy yield results.

    All percentages are fractional losses (positive = loss).
    Loss chain follows Energy_Yield_Methodology Table 10.1–10.4 order.
    """
    annual_kwh: float               # AC energy delivered to grid, kWh/year
    specific_yield_kwh_kwp: float   # annual_kwh / total_kwp
    performance_ratio: float        # PR = annual_kwh / (total_kwp × GHI/1000)

    # Itemized loss fractions (fraction of gross DC, 0–1)
    soiling_loss_frac: float
    mismatch_loss_frac: float
    lid_loss_frac: float
    temperature_loss_frac: float
    dc_cable_loss_frac: float       # from base losses.py result
    inverter_loss_frac: float
    ac_cable_loss_frac: float       # from base losses.py result
    transformer_loss_frac: float    # from base losses.py result
    availability_loss_frac: float

    total_loss_frac: float          # 1 − PR (approximately)
    poa_annual_kwh_m2: float        # annual plane-of-array irradiation, kWh/m²
    total_kwp: float


def _fetch_pvgis_tmy(lat: float, lon: float) -> pd.DataFrame:
    """
    Fetch TMY hourly data from PVGIS-5.2 via pvlib.iotools.get_pvgis_tmy.
    Returns DataFrame with columns: ghi, dhi, dni, temp_air, wind_speed.
    Source: RatedPower Design_Methodology §5.2.3 (PVGIS listed as meteo source).
    """
    data, _, _, _ = pvlib.iotools.get_pvgis_tmy(
        lat, lon, outputformat="json", map_variables=True
    )
    return data


def annual_energy_yield(
    lat: float,
    lon: float,
    total_kwp: float,
    tilt_deg: float,
    azimuth_deg: float = 180.0,
    dc_cable_loss_pct: float = 0.0,
    ac_cable_loss_pct: float = 0.0,
    transformer_loss_pct: float = 0.0,
    soiling_pct: float = 2.0,
    mismatch_pct: float = 2.0,
    lid_pct: float = 0.5,
    availability_pct: float = 1.0,
    inverter_efficiency_pct: float = 97.0,
    gamma_pmax_per_c: float = -0.004,
    noct_c: float = 45.0,
    tmy_data: pd.DataFrame | None = None,
) -> EnergyResult:
    """
    Compute annual energy yield, PR, and specific yield for a PV plant.

    Loss chain (Energy_Yield_Methodology Table 10.1–10.4 order):
      1. Irradiance → POA (transposition, pvlib Perez model)
      2. Soiling (fixed %)
      3. Mismatch (fixed %)
      4. LID — Light-Induced Degradation (fixed %)
      5. Temperature derating (Faiman model)
      6. DC cable I²R (from base losses.py, passed in as dc_cable_loss_pct)
      7. Inverter efficiency (fixed %)
      8. AC cable I²R (from base losses.py)
      9. Transformer losses (from base losses.py)
      10. Availability (fixed %)

    Parameters
    ----------
    lat, lon          : Site location, decimal degrees.
    total_kwp         : Installed DC capacity, kWp.
    tilt_deg          : Panel tilt from horizontal, degrees.
    azimuth_deg       : Panel azimuth; 180 = south-facing (pvlib convention:
                        0 = north, 90 = east, 180 = south, 270 = west).
    dc_cable_loss_pct : DC cable loss % from losses.compute_all_losses.
    ac_cable_loss_pct : AC cable loss % from losses.compute_all_losses.
    transformer_loss_pct : Transformer loss % from losses.compute_all_losses.
    soiling_pct       : Soiling loss %. Default 2 % (Energy_Yield_Methodology
                        Table 10.1 typical range 2–5 %).
    mismatch_pct      : Mismatch loss %. Default 2 % (Table 10.2, typical 1.5–2.5 %).
    lid_pct           : LID loss %. Default 0.5 % (Table 10.2 typical ~0.5 %).
    availability_pct  : Plant availability loss %. Default 1 % (Table 10.4).
    inverter_efficiency_pct : Inverter efficiency. Default 97 % (Table 10.3).
    gamma_pmax_per_c  : Module Pmax temperature coefficient, 1/°C. Typical −0.004.
    noct_c            : Nominal Operating Cell Temperature, °C. Typical 45 °C.
    tmy_data          : Pre-loaded hourly TMY DataFrame (skip PVGIS fetch).
                        Must have columns: ghi, dhi, dni, temp_air, wind_speed.
                        Index must be a DatetimeIndex with timezone.

    Returns
    -------
    EnergyResult with annual kWh, PR, specific yield, and itemized losses.
    """
    if tmy_data is None:
        tmy_data = _fetch_pvgis_tmy(lat, lon)

    # ── Sun position ──────────────────────────────────────────────────────────
    solpos = pvlib.solarposition.get_solarposition(
        tmy_data.index, lat, lon
    )

    # ── POA irradiance (Perez diffuse model) ──────────────────────────────────
    # Source: Energy_Yield_Methodology §3 Eq 3.1–3.3; pvlib Perez model.
    # Perez requires extraterrestrial DNI (dni_extra); compute from Julian day.
    dni_extra = pvlib.irradiance.get_extra_radiation(tmy_data.index)
    poa = pvlib.irradiance.get_total_irradiance(
        surface_tilt=tilt_deg,
        surface_azimuth=azimuth_deg,
        solar_zenith=solpos["apparent_zenith"],
        solar_azimuth=solpos["azimuth"],
        dni=tmy_data["dni"],
        ghi=tmy_data["ghi"],
        dhi=tmy_data["dhi"],
        model="perez",
        dni_extra=dni_extra,
    )
    G = poa["poa_global"].clip(lower=0.0)  # W/m²

    # ── Cell temperature (Faiman single-term, simplified Eq 6.15) ─────────────
    # T_cell = T_amb + G × (NOCT − 20) / 800
    T_amb = tmy_data["temp_air"]
    T_cell = T_amb + G * (noct_c - 20.0) / 800.0

    # ── Relative module output ────────────────────────────────────────────────
    # P_rel = (G / 1000) × [1 + γ_pmax × (T_cell − 25)]
    P_rel = (G / 1000.0) * (1.0 + gamma_pmax_per_c * (T_cell - 25.0))
    P_rel = P_rel.clip(lower=0.0)

    # ── Temperature loss fraction (relative to G/1000 without T correction) ──
    P_rel_notc = (G / 1000.0).clip(lower=0.0)
    temp_loss_frac = float(
        1.0 - (P_rel.sum() / P_rel_notc.sum())
        if P_rel_notc.sum() > 0 else 0.0
    )

    # ── Gross DC energy [kWh] at STC capacity ─────────────────────────────────
    # E_gross = Σ P_rel(t) × total_kwp × Δt   (Δt = 1 h)
    E_gross_kwh = float((P_rel * total_kwp).sum())  # kWh

    # ── Apply multiplicative loss chain ───────────────────────────────────────
    # Energy_Yield_Methodology Eq 10.5: E_final = E_init × Π(1 − L_i)
    def _apply(energy: float, loss_pct: float) -> float:
        return energy * (1.0 - loss_pct / 100.0)

    E = E_gross_kwh
    E = _apply(E, soiling_pct)
    E = _apply(E, mismatch_pct)
    E = _apply(E, lid_pct)
    # Temperature already baked into P_rel; do not double-count.
    E = _apply(E, dc_cable_loss_pct)
    E = _apply(E, 100.0 - inverter_efficiency_pct)
    E = _apply(E, ac_cable_loss_pct)
    E = _apply(E, transformer_loss_pct)
    E = _apply(E, availability_pct)
    annual_kwh = E

    # ── PR and specific yield ─────────────────────────────────────────────────
    # PR = E_annual / (total_kwp × GHI_annual / G_stc)
    # Energy_Yield_Methodology Eq 10.2.
    ghi_annual_kwh_m2 = float(tmy_data["ghi"].sum()) / 1000.0
    poa_annual_kwh_m2 = float(G.sum()) / 1000.0
    pr_denominator = total_kwp * ghi_annual_kwh_m2
    performance_ratio = annual_kwh / pr_denominator if pr_denominator > 0 else 0.0
    specific_yield = annual_kwh / total_kwp if total_kwp > 0 else 0.0

    total_loss_frac = 1.0 - (annual_kwh / E_gross_kwh) if E_gross_kwh > 0 else 0.0

    return EnergyResult(
        annual_kwh=annual_kwh,
        specific_yield_kwh_kwp=specific_yield,
        performance_ratio=performance_ratio,
        soiling_loss_frac=soiling_pct / 100.0,
        mismatch_loss_frac=mismatch_pct / 100.0,
        lid_loss_frac=lid_pct / 100.0,
        temperature_loss_frac=temp_loss_frac,
        dc_cable_loss_frac=dc_cable_loss_pct / 100.0,
        inverter_loss_frac=(100.0 - inverter_efficiency_pct) / 100.0,
        ac_cable_loss_frac=ac_cable_loss_pct / 100.0,
        transformer_loss_frac=transformer_loss_pct / 100.0,
        availability_loss_frac=availability_pct / 100.0,
        total_loss_frac=total_loss_frac,
        poa_annual_kwh_m2=poa_annual_kwh_m2,
        total_kwp=total_kwp,
    )
