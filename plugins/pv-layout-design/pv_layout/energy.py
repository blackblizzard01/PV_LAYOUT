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

    # Optional detailed-model terms (0 when the corresponding option is off)
    iam_loss_frac: float = 0.0
    shading_loss_frac: float = 0.0
    clipping_loss_frac: float = 0.0
    pr_poa: float = 0.0             # IEC 61724 PR = Yf / (H_poa / 1 kW/m²)
    ghi_annual_kwh_m2: float = 0.0
    azimuth_deg: float = 180.0
    horizon_loss_frac: float = 0.0  # POA lost to far-horizon (terrain) shading


def row_shade_fraction(sun_elev_deg, sun_az_deg, surface_az_deg, tilt_deg, gcr):
    """
    Fraction of a row's slope length shaded by the row in front (infinite rows).
      tan(α) = tan(elev) / cos(γs − γ)          profile angle
      fs     = 1 − sin(α) / (GCR · sin(α + β))  clipped to [0, 1]
    Source: Appelbaum & Bany (1979); same geometry as pvlib.shading.
    """
    elev = np.radians(np.asarray(sun_elev_deg, float))
    rel = np.radians(np.asarray(sun_az_deg, float) - surface_az_deg)
    beta = np.radians(tilt_deg)
    cos_rel = np.cos(rel)
    front = (elev > 0) & (cos_rel > 1e-6)
    with np.errstate(divide="ignore", invalid="ignore"):
        alpha = np.arctan2(np.tan(elev), np.where(front, cos_rel, 1.0))
        fs = 1.0 - np.sin(alpha) / (gcr * np.sin(alpha + beta))
    return np.where(front, np.clip(fs, 0.0, 1.0), 0.0)


def _fetch_pvgis_tmy(lat: float, lon: float) -> pd.DataFrame:
    """
    Fetch TMY hourly data from PVGIS-5.2 via pvlib.iotools.get_pvgis_tmy.
    Returns DataFrame with columns: ghi, dhi, dni, temp_air, wind_speed.
    Source: RatedPower Design_Methodology §5.2.3 (PVGIS listed as meteo source).
    """
    result = pvlib.iotools.get_pvgis_tmy(
        lat, lon, outputformat="json", map_variables=True
    )
    return result[0]  # pvlib ≥0.13 returns (data, meta); older returned 4 items


def annual_energy_yield(
    lat: float,
    lon: float,
    total_kwp: float,
    tilt_deg: float,
    azimuth_deg: float | None = 180.0,
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
    *,
    gcr: float | None = None,
    n_rows: int | None = None,
    iam_ar: float | None = None,
    pv_model: str = "linear",
    horizon: pd.Series | None = None,
    ac_capacity_kw: float | None = None,
    ohmic_at_stc: bool = False,
    transformer_losses: tuple[float, float, float] | None = None,
    power_factor: float = 1.0,
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

    Detailed-model options (keyword-only; all off by default):
    azimuth_deg=None  : face the equator (180° north / 0° south hemisphere).
    gcr, n_rows       : inter-row beam shading (row_shade_fraction); the front
                        row is unshaded, so the loss is scaled by (1 − 1/n_rows).
    iam_ar            : Martin & Ruiz IAM (a_r ≈ 0.16 for glass) applied to beam,
                        sky-diffuse and ground-reflected POA (same model as PVGIS).
    pv_model          : "linear" — NOCT cell temperature + γ_pmax (legacy);
                        "huld"   — Faiman module temperature (u0 26.9, u1 6.2,
                        wind from TMY) + Huld c-Si power model, which includes
                        low-irradiance efficiency (PVGIS-5 coefficients).
    horizon           : Series azimuth° → horizon elevation°; beam is blocked
                        while the sun is below the far horizon (terrain).
    ac_capacity_kw    : hourly inverter clipping at total AC rating.
    ohmic_at_stc      : treat dc/ac cable % as values at STC; scale hourly with
                        load (I²R ∝ P²) instead of a flat derate.
    transformer_losses: (no_load_kw, load_loss_kw_at_rated, rated_kva) summed
                        over all power stations; applied hourly, no-load 24 h.

    Returns
    -------
    EnergyResult with annual kWh, PR, specific yield, and itemized losses.
    """
    if azimuth_deg is None:
        azimuth_deg = 180.0 if lat >= 0 else 0.0
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

    def transpose(dni):
        return pvlib.irradiance.get_total_irradiance(
            surface_tilt=tilt_deg,
            surface_azimuth=azimuth_deg,
            solar_zenith=solpos["apparent_zenith"],
            solar_azimuth=solpos["azimuth"],
            dni=dni,
            ghi=tmy_data["ghi"],
            dhi=tmy_data["dhi"],
            model="perez",
            dni_extra=dni_extra,
        )

    poa = transpose(tmy_data["dni"])
    G_open = poa["poa_global"].fillna(0.0).clip(lower=0.0)
    horizon_loss_frac = 0.0
    if horizon is not None:
        hz = np.interp(solpos["azimuth"].to_numpy() % 360.0, horizon.index.to_numpy(float),
                       horizon.to_numpy(float), period=360.0)
        blocked = solpos["apparent_elevation"].to_numpy() < hz
        poa = transpose(tmy_data["dni"].where(~blocked, 0.0))
    G = poa["poa_global"].fillna(0.0).clip(lower=0.0)  # W/m²; Perez is NaN at some twilight hours
    if horizon is not None and G_open.sum() > 0:
        horizon_loss_frac = 1.0 - float(G.sum()) / float(G_open.sum())

    # ── Optional IAM + inter-row beam shading on the direct component ─────────
    direct = poa["poa_direct"].fillna(0.0).clip(lower=0.0)
    sky = poa["poa_sky_diffuse"].fillna(0.0).clip(lower=0.0)
    ground = poa["poa_ground_diffuse"].fillna(0.0).clip(lower=0.0)
    direct_iam, sky_iam, ground_iam = direct, sky, ground
    if iam_ar is not None:
        aoi = pvlib.irradiance.aoi(tilt_deg, azimuth_deg, solpos["apparent_zenith"], solpos["azimuth"])
        direct_iam = direct * pvlib.iam.martin_ruiz(aoi, a_r=iam_ar).fillna(0.0)
        mrd = pvlib.iam.martin_ruiz_diffuse(tilt_deg, a_r=iam_ar)
        f_sky, f_gnd = (mrd["sky"], mrd["ground"]) if isinstance(mrd, dict) else mrd
        sky_iam, ground_iam = sky * float(f_sky), ground * float(f_gnd)
    direct_eff = direct_iam
    if gcr is not None:
        fs = row_shade_fraction(solpos["apparent_elevation"], solpos["azimuth"],
                                azimuth_deg, tilt_deg, gcr)
        front = 1.0 - 1.0 / n_rows if n_rows and n_rows > 0 else 1.0
        direct_eff = direct_iam * (1.0 - fs * front)
    G_iam = (direct_iam + sky_iam + ground_iam).clip(lower=0.0)
    G_eff = (direct_eff + sky_iam + ground_iam).clip(lower=0.0)
    g_tot = float(G.sum())
    g_iam = float(G_iam.sum())
    iam_loss_frac = 1.0 - g_iam / g_tot if g_tot > 0 and iam_ar is not None else 0.0
    shading_loss_frac = 1.0 - float(G_eff.sum()) / g_iam if g_iam > 0 and gcr is not None else 0.0
    if iam_ar is None and gcr is None:
        G_eff = G

    T_amb = tmy_data["temp_air"]
    if pv_model == "huld":
        # Module temperature: Faiman (PVGIS free-standing coefficients); power: Huld c-Si.
        wind = tmy_data["wind_speed"] if "wind_speed" in tmy_data else 1.0
        T_mod = pvlib.temperature.faiman(G, T_amb, wind, u0=26.9, u1=6.2)
        P_rel = pvlib.pvarray.huld(G_eff, T_mod, pdc0=1.0, cell_type="csi").fillna(0.0).clip(lower=0.0)
        P_rel = P_rel.where(G_eff > 1.0, 0.0)
    elif pv_model == "linear":
        # ── Cell temperature (simplified Eq 6.15): T_cell = T_amb + G·(NOCT − 20)/800
        T_cell = T_amb + G * (noct_c - 20.0) / 800.0
        # ── Relative output: P_rel = (G / 1000) × [1 + γ_pmax × (T_cell − 25)]
        P_rel = ((G_eff / 1000.0) * (1.0 + gamma_pmax_per_c * (T_cell - 25.0))).clip(lower=0.0)
    else:
        raise ValueError(f"pv_model must be 'linear' or 'huld', got {pv_model!r}")

    # ── Temperature (+ low-irradiance for Huld) loss fraction vs ideal G/1000 ──
    P_rel_notc = (G_eff / 1000.0).clip(lower=0.0)
    temp_loss_frac = float(
        1.0 - (P_rel.sum() / P_rel_notc.sum())
        if P_rel_notc.sum() > 0 else 0.0
    )

    detailed = ac_capacity_kw is not None or ohmic_at_stc or transformer_losses is not None
    if detailed:
        return _hourly_chain(
            P_rel=P_rel.to_numpy(), total_kwp=total_kwp, tmy_data=tmy_data, G=G,
            soiling_pct=soiling_pct, mismatch_pct=mismatch_pct, lid_pct=lid_pct,
            dc_cable_loss_pct=dc_cable_loss_pct, ac_cable_loss_pct=ac_cable_loss_pct,
            transformer_loss_pct=transformer_loss_pct, availability_pct=availability_pct,
            inverter_efficiency_pct=inverter_efficiency_pct, ac_capacity_kw=ac_capacity_kw,
            ohmic_at_stc=ohmic_at_stc, transformer_losses=transformer_losses,
            power_factor=power_factor, temp_loss_frac=temp_loss_frac,
            iam_loss_frac=iam_loss_frac, shading_loss_frac=shading_loss_frac,
            azimuth_deg=azimuth_deg, horizon_loss_frac=horizon_loss_frac,
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
        iam_loss_frac=iam_loss_frac,
        shading_loss_frac=shading_loss_frac,
        pr_poa=specific_yield / poa_annual_kwh_m2 if poa_annual_kwh_m2 > 0 else 0.0,
        ghi_annual_kwh_m2=ghi_annual_kwh_m2,
        azimuth_deg=azimuth_deg,
        horizon_loss_frac=horizon_loss_frac,
    )


def _hourly_chain(*, P_rel, total_kwp, tmy_data, G, soiling_pct, mismatch_pct, lid_pct,
                  dc_cable_loss_pct, ac_cable_loss_pct, transformer_loss_pct,
                  availability_pct, inverter_efficiency_pct, ac_capacity_kw, ohmic_at_stc,
                  transformer_losses, power_factor, temp_loss_frac, iam_loss_frac,
                  shading_loss_frac, azimuth_deg, horizon_loss_frac=0.0) -> EnergyResult:
    """Hourly loss chain with clipping, load-scaled I²R and transformer losses."""
    def frac(e_in, e_out):
        return 1.0 - e_out / e_in if e_in > 0 else 0.0

    p = P_rel * total_kwp                                           # kW DC at module
    p = p * (1 - soiling_pct / 100) * (1 - mismatch_pct / 100) * (1 - lid_pct / 100)
    e_mod = p.sum()

    k_dc = dc_cable_loss_pct / 100 * ((p / total_kwp) if ohmic_at_stc and total_kwp > 0 else 1.0)
    p_dc = p * (1 - k_dc)
    e_dc = p_dc.sum()

    eta = inverter_efficiency_pct / 100
    p_inv = p_dc * eta
    e_inv_unclipped = p_inv.sum()
    if ac_capacity_kw is not None:
        p_inv = np.minimum(p_inv, ac_capacity_kw)
    e_inv = p_inv.sum()

    ac_ref = min(total_kwp * eta, ac_capacity_kw) if ac_capacity_kw else total_kwp * eta
    k_ac = ac_cable_loss_pct / 100 * ((p_inv / ac_ref) if ohmic_at_stc and ac_ref > 0 else 1.0)
    p_ac = p_inv * (1 - k_ac)
    e_ac = p_ac.sum()

    if transformer_losses is not None:
        p0, pk, s_rated = transformer_losses
        loss = p0 + pk * (p_ac / power_factor / s_rated) ** 2
        p_tx = p_ac - loss                                         # night: −p0 (import)
    else:
        p_tx = p_ac * (1 - transformer_loss_pct / 100)
    e_tx = p_tx.sum()

    p_out = p_tx * (1 - availability_pct / 100)
    annual_kwh = float(p_out.sum())

    ghi = float(tmy_data["ghi"].sum()) / 1000.0
    poa = float(G.sum()) / 1000.0
    sy = annual_kwh / total_kwp if total_kwp > 0 else 0.0
    e_gross = float((P_rel * total_kwp).sum())
    return EnergyResult(
        annual_kwh=annual_kwh,
        specific_yield_kwh_kwp=sy,
        performance_ratio=annual_kwh / (total_kwp * ghi) if total_kwp * ghi > 0 else 0.0,
        soiling_loss_frac=soiling_pct / 100,
        mismatch_loss_frac=mismatch_pct / 100,
        lid_loss_frac=lid_pct / 100,
        temperature_loss_frac=temp_loss_frac,
        dc_cable_loss_frac=frac(e_mod, e_dc),
        inverter_loss_frac=1 - eta,
        ac_cable_loss_frac=frac(e_inv, e_ac),
        transformer_loss_frac=frac(e_ac, e_tx),
        availability_loss_frac=availability_pct / 100,
        total_loss_frac=frac(e_gross, annual_kwh),
        poa_annual_kwh_m2=poa,
        total_kwp=total_kwp,
        iam_loss_frac=iam_loss_frac,
        shading_loss_frac=shading_loss_frac,
        clipping_loss_frac=float(frac(e_inv_unclipped, e_inv)),
        pr_poa=sy / poa if poa > 0 else 0.0,
        ghi_annual_kwh_m2=ghi,
        azimuth_deg=azimuth_deg,
        horizon_loss_frac=horizon_loss_frac,
    )
