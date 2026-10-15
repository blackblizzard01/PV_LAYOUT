"""
Bill of Quantities: summarise cable lengths, trench volumes, and equipment
counts from an existing layout + placement result.

Source: RatedPower Bill_of_Quantities_Methodology.pdf, extracted via
reference/Bill_of_Quantities_Methodology.txt.

Trench rules applied here:
  - LV trench: depth 1.0 m, width 0.4–0.8 m (BQ doc §3).
  - Trench volume = total_cable_length × trench_cross_section.
  - Control building: 150 m² if site < 50 MW, else 300 m²
    (BQ doc substation BoQ rules).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .losses import CableSpec, n_conductors


@dataclass
class BoQResult:
    """
    Bill-of-quantities summary for a PV plant layout.

    All cable lengths are one-way Euclidean (straight-line);
    multiply by a routing factor (typically 1.15–1.30) to estimate
    real trench lengths.
    """
    # DC side
    dc_cable_total_m: float          # sum of block→inverter cable runs, m
    dc_cable_material: str
    dc_cable_area_mm2: float
    dc_cable_mass_kg: float          # estimated copper/aluminium mass

    # AC side
    ac_cable_total_m: float          # sum of inverter→transformer cable runs, m
    ac_cable_material: str
    ac_cable_area_mm2: float
    ac_cable_mass_kg: float

    # Equipment counts
    n_modules: int
    n_inverters: int
    n_transformers: int

    # Civil
    trench_depth_m: float
    trench_width_m: float
    trench_volume_m3: float          # (DC + AC cable length) × depth × width

    # Substation control building
    control_building_m2: float       # 150 m² if < 50 MW, else 300 m²

    # Totals
    total_cable_m: float             # DC + AC
    site_kwp: float


# ── Conductor mass densities kg/m per mm² ────────────────────────────────────
# Mass = area_mm² × length_m × density_kg_per_m_per_mm2
# Cu: ~8900 kg/m³ → 8.9e-3 kg / (m · mm²)
# Al: ~2700 kg/m³ → 2.7e-3 kg / (m · mm²)
_MASS_DENSITY = {"copper": 8.9e-3, "aluminum": 2.7e-3}


def bill_of_quantities(
    total_modules: int,
    total_kwp: float,
    placement: "PlacementResult",
    dc_cable: CableSpec,
    ac_cable: CableSpec,
    trench_depth_m: float = 1.0,
    trench_width_m: float = 0.6,
) -> BoQResult:
    """
    Compute a BoQ from a completed layout + placement.

    Parameters
    ----------
    total_modules   : module count from LayoutResult.
    total_kwp       : DC capacity, kWp.
    placement       : PlacementResult from place_equipment.
    dc_cable        : CableSpec for the DC (block→inverter) segments.
    ac_cable        : CableSpec for the AC (inverter→transformer) segments.
    trench_depth_m  : LV trench depth.  Default 1.0 m (BoQ §3 LV rules).
    trench_width_m  : LV trench width.  Default 0.6 m (mid-range 0.4–0.8 m).

    Cable mass estimated as: mass = n × A_mm2 × L_m × density_kg_per_m_per_mm2,
    with n = 2 conductors for DC and 3 for 3-phase AC.
    Routing overhead (trenches vs straight-line) is NOT applied here;
    include a 15–30 % factor in downstream cost calculations.

    Source: Bill_of_Quantities_Methodology.pdf §3 (LV trench sizing),
    reference/Bill_of_Quantities_Methodology.txt.
    """
    dc_m = placement.total_dc_cable_m
    ac_m = placement.total_ac_cable_m
    total_cable_m = dc_m + ac_m

    # Route length × conductors per circuit (2 DC, 3 three-phase AC).
    dc_mass = (n_conductors(False) * dc_cable.area_mm2 * dc_m
               * _MASS_DENSITY.get(dc_cable.material.lower(), 8.9e-3))
    ac_mass = (n_conductors(True) * ac_cable.area_mm2 * ac_m
               * _MASS_DENSITY.get(ac_cable.material.lower(), 8.9e-3))

    trench_vol = total_cable_m * trench_depth_m * trench_width_m

    # Control building: 150 m² < 50 MW, else 300 m²
    control_building = 150.0 if total_kwp < 50_000 else 300.0

    return BoQResult(
        dc_cable_total_m=dc_m,
        dc_cable_material=dc_cable.material,
        dc_cable_area_mm2=dc_cable.area_mm2,
        dc_cable_mass_kg=dc_mass,
        ac_cable_total_m=ac_m,
        ac_cable_material=ac_cable.material,
        ac_cable_area_mm2=ac_cable.area_mm2,
        ac_cable_mass_kg=ac_mass,
        n_modules=total_modules,
        n_inverters=placement.n_inverters,
        n_transformers=placement.n_transformers,
        trench_depth_m=trench_depth_m,
        trench_width_m=trench_width_m,
        trench_volume_m3=trench_vol,
        control_building_m2=control_building,
        total_cable_m=total_cable_m,
        site_kwp=total_kwp,
    )
