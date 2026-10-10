"""
Conductor sizing and string sizing for PV plant electrical design.

Sources
-------
- Conductor sizing (3 criteria): RatedPower Electrical_Methodology §3 and
  T_D_UndergroundLine_Methodology §2, extracted from reference/*.txt.
  Criteria: (1) current-carrying capacity (ampacity), (2) voltage drop,
  (3) short-circuit withstand.
- Standard cross-sections: IEC 60228 (Table 1, conductor sizes mm²).
- Temperature-corrected resistivity: IEC 60228 / Electrical_Methodology Eq 3.x.
- String sizing from MPPT window: Electrical_Methodology §1, Eq 1.1–1.7.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# IEC 60228 standard conductor cross-sections in mm²
STANDARD_SECTIONS_MM2: tuple[float, ...] = (
    1.5, 2.5, 4.0, 6.0, 10.0, 16.0, 25.0, 35.0,
    50.0, 70.0, 95.0, 120.0, 150.0, 185.0, 240.0,
    300.0, 400.0, 500.0, 630.0,
)

# IEC 60228 resistivity at 20 °C, Ω·m
_RESISTIVITY_20C = {"copper": 1.68e-8, "aluminum": 2.82e-8}

# Temperature coefficient of resistance, 1/°C (IEC 60228)
_ALPHA = {"copper": 0.00393, "aluminum": 0.00403}


@dataclass(frozen=True)
class StringSizingResult:
    n_min: int          # minimum modules per string (MPPT lower limit)
    n_max: int          # maximum modules per string (voltage / MPPT upper limit)
    n_recommended: int  # n_max (or n_min if n_max < n_min — sizing conflict)
    sizing_ok: bool     # False if n_min > n_max (MPPT window too narrow)


def _resistivity_at_t(material: str, temp_c: float) -> float:
    """
    DC resistivity at operating temperature using IEC 60228 linear model.
    ρ(T) = ρ₂₀ × [1 + α × (T − 20)]
    Source: Electrical_Methodology §3.21, T_D_UndergroundLine §2.
    """
    rho20 = _RESISTIVITY_20C[material.lower()]
    alpha = _ALPHA[material.lower()]
    return rho20 * (1.0 + alpha * (temp_c - 20.0))


def select_conductor_area(
    power_kw: float,
    voltage_v: float,
    length_m: float,
    material: str = "copper",
    max_vdrop_pct: float = 0.5,
    three_phase: bool = False,
    power_factor: float = 1.0,
    conductor_temp_c: float = 90.0,
) -> float:
    """
    Return the smallest IEC-standard conductor cross-section (mm²) that
    satisfies the voltage-drop criterion at the given operating conditions.

    Voltage-drop criterion (Electrical_Methodology §3, Eq 3.20 DC / Eq 3.19 AC):

      DC:        S_min = 2 × ρ(T) × L × I / (δ_V × V)
      3-phase AC: S_min ≈ ρ(T) × L × I × cos(φ) / (δ_V × V_phase)
                          (reactance term omitted — conservative for LV cables)

    where δ_V = max_vdrop_pct / 100, and the factor of 2 on DC accounts for
    the return conductor.

    Parameters
    ----------
    power_kw         : power flowing through this segment, kW.
    voltage_v        : bus voltage (DC bus voltage, or line-to-line AC voltage).
    length_m         : one-way cable length, metres.
    material         : "copper" or "aluminum".
    max_vdrop_pct    : maximum allowable voltage drop as % of bus voltage.
                       RatedPower default: 0.5 % (LV), 3 % (MV).
    three_phase      : True for 3-phase AC segments.
    power_factor     : AC power factor (ignored for DC).
    conductor_temp_c : operating temperature for resistivity correction (°C).
                       Typically 90 °C for XLPE, 70 °C for PVC.

    Returns
    -------
    Cross-section in mm² (the smallest standard size meeting the criterion).
    Raises ValueError if even the largest standard section is insufficient.

    Source: Electrical_Methodology.pdf §3, extracted reference/Electrical_Methodology.txt.
    """
    if material.lower() not in _RESISTIVITY_20C:
        raise ValueError(f"Unknown material '{material}'. Choose copper or aluminum.")
    if power_kw <= 0 or length_m <= 0:
        return STANDARD_SECTIONS_MM2[0]

    rho = _resistivity_at_t(material, conductor_temp_c)
    delta_v = max_vdrop_pct / 100.0
    power_w = power_kw * 1000.0

    if three_phase:
        I = power_w / (voltage_v * math.sqrt(3.0) * power_factor)
        # Three-phase voltage-drop: ΔV = √3 × I × R × L / Nc
        # S_min = ρ(T) × L × I × cos(φ) × √3 / (δ_V × V_LL)
        s_min_m2 = rho * length_m * I * power_factor * math.sqrt(3.0) / (delta_v * voltage_v)
    else:
        I = power_w / voltage_v
        # DC: S_min = 2 × ρ(T) × L × I / (δ_V × V)
        s_min_m2 = 2.0 * rho * length_m * I / (delta_v * voltage_v)

    s_min_mm2 = s_min_m2 * 1e6  # convert m² → mm²

    for s in STANDARD_SECTIONS_MM2:
        if s >= s_min_mm2:
            return s

    raise ValueError(
        f"Required conductor area {s_min_mm2:.1f} mm² exceeds the largest "
        f"standard section ({STANDARD_SECTIONS_MM2[-1]} mm²).  "
        "Consider splitting the circuit or raising the voltage."
    )


# ── F6: String sizing ─────────────────────────────────────────────────────────

def string_sizing(
    voc_ref_v: float,
    vmp_ref_v: float,
    voc_tc_pct_per_c: float,
    vmp_tc_pct_per_c: float,
    v_max_system_v: float,
    v_mppt_min_v: float,
    v_mppt_max_v: float,
    t_min_c: float,
    t_max_c: float,
    noct_c: float = 45.0,
    irradiance_max_wm2: float = 1000.0,
) -> StringSizingResult:
    """
    Compute the allowable range of modules per string from the inverter MPPT
    voltage window and module temperature coefficients.

    Algorithm (Electrical_Methodology §1, Eq 1.1–1.7):

    Cold-condition (maximum Voc):
      T_cell_min ≈ T_air_min
      Voc_cold = Voc_ref × [1 + (voc_tc_pct_per_c/100) × (T_cell_min − 25)]
      N_max_sys  = floor(V_max_system / Voc_cold)
      N_max_mppt = floor(V_mppt_max / Vmp_ref)
      N_max = min(N_max_sys, N_max_mppt)

    Hot-condition (minimum Vmp):
      T_cell_max = T_air_max + (NOCT − 20) × G_max / 800
      Vmp_hot = Vmp_ref × [1 + (vmp_tc_pct_per_c/100) × (T_cell_max − 25)]
      N_min = ceil(V_mppt_min / Vmp_hot)

    Parameters
    ----------
    voc_ref_v         : Open-circuit voltage at STC, V.
    vmp_ref_v         : MPP voltage at STC, V.
    voc_tc_pct_per_c  : Voc temperature coefficient, % per °C (typically −0.23 to −0.3).
    vmp_tc_pct_per_c  : Vmp temperature coefficient, % per °C (typically −0.23 to −0.3).
    v_max_system_v    : Maximum DC system voltage, V (e.g. 1000 or 1500).
    v_mppt_min_v      : Minimum inverter MPPT voltage, V.
    v_mppt_max_v      : Maximum inverter MPPT voltage, V.
    t_min_c           : Site minimum historical air temperature, °C.
    t_max_c           : Site maximum historical air temperature, °C.
    noct_c            : Nominal Operating Cell Temperature, °C (typical 45°C).
    irradiance_max_wm2: Maximum expected irradiance, W/m² (typically 1000).

    Returns
    -------
    StringSizingResult with n_min, n_max, n_recommended, sizing_ok.

    Source: Electrical_Methodology.pdf §1 Eq 1.1–1.7,
    reference/Electrical_Methodology.txt.
    """
    # Cold condition — highest voltage (use air temperature directly, Eq 1.3)
    t_cell_min = t_min_c
    voc_cold = voc_ref_v * (1.0 + voc_tc_pct_per_c / 100.0 * (t_cell_min - 25.0))
    n_max_sys = int(v_max_system_v / voc_cold) if voc_cold > 0 else 0
    n_max_mppt = int(v_mppt_max_v / vmp_ref_v) if vmp_ref_v > 0 else 0
    n_max = min(n_max_sys, n_max_mppt)

    # Hot condition — lowest voltage (Eq 1.5/1.6)
    t_cell_max = t_max_c + (noct_c - 20.0) * irradiance_max_wm2 / 800.0
    vmp_hot = vmp_ref_v * (1.0 + vmp_tc_pct_per_c / 100.0 * (t_cell_max - 25.0))
    n_min = math.ceil(v_mppt_min_v / vmp_hot) if vmp_hot > 0 else 1

    sizing_ok = n_min <= n_max
    n_recommended = n_max if sizing_ok else n_min

    return StringSizingResult(
        n_min=n_min,
        n_max=n_max,
        n_recommended=n_recommended,
        sizing_ok=sizing_ok,
    )
