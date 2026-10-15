"""
Cable (I²R) and transformer loss calculations.

Sources
-------
- Resistivity constants: IEC 60228:2004, Table 1.
  Cu ≈ 1.68×10⁻⁸ Ω·m, Al ≈ 2.82×10⁻⁸ Ω·m at 20 °C.
- I²R formula: Chapman, *Electric Machinery Fundamentals*, 5th ed. (2011), §2.3.
  Lengths are one-way route lengths; every current-carrying conductor
  dissipates I²R, so DC (+/−) counts 2 conductors and 3-phase AC counts 3.
- Transformer two-term loss model: IEC 60076-1:2011, §5.11.
  P_loss = P_no_load + P_load_rated × (S / S_rated)²
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .placement import PlacementResult, TransformerSpec


# IEC 60228:2004 Table 1 — resistivity at 20 °C, Ω·m
RESISTIVITY: dict[str, float] = {
    "copper": 1.68e-8,
    "aluminum": 2.82e-8,
}


def n_conductors(three_phase: bool) -> int:
    """Current-carrying conductors per circuit: 2 for DC (+/−), 3 for 3-phase AC."""
    return 3 if three_phase else 2


@dataclass(frozen=True)
class CableSpec:
    material: str    # "copper" or "aluminum"
    area_mm2: float  # conductor cross-sectional area, mm²

    def __post_init__(self) -> None:
        key = self.material.lower()
        if key not in RESISTIVITY:
            raise ValueError(
                f"Unknown cable material '{self.material}'. "
                f"Choose from: {list(RESISTIVITY)}"
            )


@dataclass
class LossResult:
    dc_cable_loss_kw: float
    dc_cable_loss_pct: float    # % of DC power entering the cable
    ac_cable_loss_kw: float
    ac_cable_loss_pct: float    # % of AC power entering the cable
    transformer_loss_kw: float
    transformer_loss_pct: float # % of power entering the transformer
    total_loss_kw: float
    total_loss_pct: float       # % of gross DC site output
    site_output_kw: float       # gross DC capacity (= total_kwp, at STC)


def cable_loss(
    cable: CableSpec,
    total_length_m: float,
    power_kw: float,
    voltage_v: float,
    three_phase: bool = False,
    power_factor: float = 1.0,
) -> tuple[float, float]:
    """
    I²R cable loss for a DC or AC segment.

    R = ρ × L / A                              (Ω, one conductor)
    I = P / V                                  (DC)
      = P / (V × √3 × PF)                     (3-phase AC)
    P_loss = n × I² × R                        (n = 2 DC, 3 three-phase)

    Parameters
    ----------
    cable        : conductor material and cross-section
    total_length_m : one-way route length of this segment, metres
    power_kw     : power flowing through this segment, kW
    voltage_v    : bus voltage, V (DC bus voltage or line-to-line AC voltage)
    three_phase  : True for 3-phase AC segments
    power_factor : AC power factor (ignored for DC)

    Returns (loss_kw, loss_pct_of_input_power).

    Source: Chapman, Electric Machinery Fundamentals, 5th ed. §2.3;
            IEC 60228:2004 resistivities.
    """
    if total_length_m <= 0 or power_kw <= 0:
        return 0.0, 0.0

    rho = RESISTIVITY[cable.material.lower()]
    area_m2 = cable.area_mm2 * 1e-6
    R = rho * total_length_m / area_m2  # Ω

    power_w = power_kw * 1000.0
    if three_phase:
        I = power_w / (voltage_v * math.sqrt(3.0) * power_factor)
    else:
        I = power_w / voltage_v

    p_loss_w = n_conductors(three_phase) * I * I * R
    loss_kw = p_loss_w / 1000.0
    loss_pct = 100.0 * p_loss_w / power_w
    return loss_kw, loss_pct


def transformer_loss(
    rated_kva: float,
    p_no_load_kw: float,
    p_load_rated_kw: float,
    actual_power_kw: float,
    power_factor: float = 1.0,
) -> tuple[float, float]:
    """
    Two-term nameplate transformer loss (IEC 60076-1:2011 §5.11).

    P_loss = P_no_load + P_load_rated × (S / S_rated)²

    P_no_load      : core/iron loss — constant, independent of loading.
    P_load_rated   : copper/winding loss at rated load.
    S              : actual apparent power flowing through the transformer.
    S_rated        : rated apparent power (= rated_kva).

    All three nameplate values (rated_kva, p_no_load_kw, p_load_rated_kw)
    appear on every real transformer datasheet.

    Returns (loss_kw, loss_pct_of_actual_power_input).
    """
    if actual_power_kw <= 0:
        return p_no_load_kw, 0.0

    actual_kva = actual_power_kw / power_factor
    loading = actual_kva / rated_kva
    loss_kw = p_no_load_kw + p_load_rated_kw * (loading ** 2)
    loss_pct = 100.0 * loss_kw / actual_power_kw
    return loss_kw, loss_pct


def _cable_loss_segments(
    cable: CableSpec,
    lengths_m: np.ndarray,
    powers_kw: np.ndarray,
    voltage_v: float,
    three_phase: bool = False,
    power_factor: float = 1.0,
) -> float:
    """
    Sum of I²R losses over N independent parallel cable segments (kW).

    Each segment i carries powers_kw[i] over lengths_m[i].  Because I²R
    scales as the *square* of current, aggregating all segments into one
    "total length / total power" cable is incorrect — this function sums
    the per-segment losses to give the physically correct answer.

    Source: same as cable_loss (Chapman §2.3; IEC 60228 resistivities).
    """
    rho = RESISTIVITY[cable.material.lower()]
    area_m2 = cable.area_mm2 * 1e-6
    R = rho * np.asarray(lengths_m) / area_m2              # Ω per segment
    powers_w = np.asarray(powers_kw) * 1000.0
    if three_phase:
        I = powers_w / (voltage_v * math.sqrt(3.0) * power_factor)
    else:
        I = powers_w / voltage_v
    return n_conductors(three_phase) * float((I * I * R).sum()) / 1000.0  # kW


def compute_all_losses(
    placement: PlacementResult,
    total_kwp: float,
    dc_cable: CableSpec,
    ac_cable: CableSpec,
    dc_voltage_v: float,
    ac_voltage_v: float,
    transformer: TransformerSpec,
    ac_power_factor: float = 0.98,
) -> LossResult:
    """
    Compute DC cable, AC cable, and transformer losses for the full site.

    Uses per-segment I²R computation (see _cable_loss_segments) so that
    each block→inverter and inverter→transformer cable segment is evaluated
    at its own current, not the aggregate site current.

    Losses are applied sequentially: DC loss reduces AC input; AC loss
    reduces transformer input.  Each is reported as kW and as % of the
    power entering that stage, plus an aggregate % of gross DC output.
    """
    # ── DC cable: one segment per block ───────────────────────────────────
    dc_loss_kw = _cable_loss_segments(
        dc_cable,
        placement.dc_cable_lengths_m,
        placement.block_kwp,
        dc_voltage_v,
        three_phase=False,
    )
    dc_loss_pct = 100.0 * dc_loss_kw / total_kwp if total_kwp > 0 else 0.0

    # ── AC cable: one segment per inverter ────────────────────────────────
    # Power delivered by each inverter = DC kWp of its assigned blocks,
    # reduced by the average DC cable efficiency.
    dc_efficiency = 1.0 - dc_loss_pct / 100.0
    inv_dc_kwp = np.array([
        placement.block_kwp[placement.block_to_inverter == j].sum()
        for j in range(placement.n_inverters)
    ])
    inv_ac_kw = inv_dc_kwp * dc_efficiency

    ac_loss_kw = _cable_loss_segments(
        ac_cable,
        placement.ac_cable_lengths_m,
        inv_ac_kw,
        ac_voltage_v,
        three_phase=True,
        power_factor=ac_power_factor,
    )
    total_ac_kw = float(inv_ac_kw.sum())
    ac_loss_pct = 100.0 * ac_loss_kw / total_ac_kw if total_ac_kw > 0 else 0.0

    # ── Transformer ────────────────────────────────────────────────────────
    tx_input_kw = total_ac_kw - ac_loss_kw
    tx_loss_kw, tx_loss_pct = transformer_loss(
        transformer.rated_kva,
        transformer.p_no_load_kw,
        transformer.p_load_rated_kw,
        tx_input_kw,
        power_factor=ac_power_factor,
    )

    total_loss_kw = dc_loss_kw + ac_loss_kw + tx_loss_kw
    total_loss_pct = 100.0 * total_loss_kw / total_kwp if total_kwp > 0 else 0.0

    return LossResult(
        dc_cable_loss_kw=dc_loss_kw,
        dc_cable_loss_pct=dc_loss_pct,
        ac_cable_loss_kw=ac_loss_kw,
        ac_cable_loss_pct=ac_loss_pct,
        transformer_loss_kw=tx_loss_kw,
        transformer_loss_pct=tx_loss_pct,
        total_loss_kw=total_loss_kw,
        total_loss_pct=total_loss_pct,
        site_output_kw=total_kwp,
    )
