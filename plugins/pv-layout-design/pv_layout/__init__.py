"""
pv_layout — PV site geometric and electrical design tool.

Layer 1 (base): rectangle layout, I²R losses, k-means placement, plot.
Layer 2: tilt recommendation, GCR input, conductor/string sizing,
         multi-transformer, Bill of Quantities.
Layer 3: polygon site + setbacks, annual energy yield, terrain analysis.
"""

# ── Layer 1 (base) ────────────────────────────────────────────────────────────
from .geometry import (
    place_panels, compute_pitch, compute_gcr,
    ModuleSpec, SiteSpec, LayoutResult,
)
from .placement import (
    place_equipment, InverterSpec, TransformerSpec, PlacementResult,
)
from .losses import (
    compute_all_losses, cable_loss, transformer_loss,
    CableSpec, LossResult,
)
from .plot import render_layout

# ── Layer 2 ───────────────────────────────────────────────────────────────────
from .geometry import recommend_tilt                          # F3
# F4 (gcr kwarg) is part of place_panels — no extra export needed
from .electrical import (                                     # F5, F6
    select_conductor_area, string_sizing,
    StringSizingResult, STANDARD_SECTIONS_MM2,
)
from .boq import bill_of_quantities, BoQResult                # F8

# ── Layer 3 ───────────────────────────────────────────────────────────────────
from .geometry import place_panels_polygon, apply_setback     # F1, F2
from .energy import annual_energy_yield, EnergyResult         # F9
from .terrain import (                                        # F10
    compute_slopes, apply_terrain_limits, synthetic_dem,
    DEMGrid, SlopeResult, TerrainResult,
)

# ── Design engine (real sites, tables, terrain, electrical, renders) ─────────
from .site import Site                                        # noqa: E402
from .geo import GeoFrame, read_site                          # noqa: E402
from .tables import StructureSpec, TableLayout, layout_tables  # noqa: E402
from .terrain import SlopeLimits, TableTerrain, classify_tables  # noqa: E402
from .blocks import ElectricalSpec, ElectricalLayout, build_electrical  # noqa: E402
from .design import DesignConfig, DesignResult, run_design, save_summary  # noqa: E402

__all__ = [
    # Design engine
    "Site", "GeoFrame", "read_site", "StructureSpec", "TableLayout", "layout_tables",
    "SlopeLimits", "TableTerrain", "classify_tables", "ElectricalSpec", "ElectricalLayout",
    "build_electrical", "DesignConfig", "DesignResult", "run_design", "save_summary",
    # Layer 1
    "place_panels", "compute_pitch", "compute_gcr",
    "ModuleSpec", "SiteSpec", "LayoutResult",
    "place_equipment", "InverterSpec", "TransformerSpec", "PlacementResult",
    "compute_all_losses", "cable_loss", "transformer_loss",
    "CableSpec", "LossResult",
    "render_layout",
    # Layer 2
    "recommend_tilt",
    "select_conductor_area", "string_sizing",
    "StringSizingResult", "STANDARD_SECTIONS_MM2",
    "bill_of_quantities", "BoQResult",
    # Layer 3
    "place_panels_polygon", "apply_setback",
    "annual_energy_yield", "EnergyResult",
    "compute_slopes", "apply_terrain_limits", "synthetic_dem",
    "DEMGrid", "SlopeResult", "TerrainResult",
]
