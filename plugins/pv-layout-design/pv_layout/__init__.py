"""
pv_layout — PV site geometric and electrical design tool.

Compute panel layout, inverter/transformer placement, cable losses, and
render the site plan. Standalone package; outputs (loss %, area utilisation)
are intended to feed into OptimaRE as parameters in a separate integration step.
"""

from .geometry import place_panels, compute_pitch, compute_gcr, ModuleSpec, SiteSpec, LayoutResult
from .placement import place_equipment, InverterSpec, TransformerSpec, PlacementResult
from .losses import compute_all_losses, cable_loss, transformer_loss, CableSpec, LossResult
from .plot import render_layout

__all__ = [
    "place_panels", "compute_pitch", "compute_gcr", "ModuleSpec", "SiteSpec", "LayoutResult",
    "place_equipment", "InverterSpec", "TransformerSpec", "PlacementResult",
    "compute_all_losses", "cable_loss", "transformer_loss", "CableSpec", "LossResult",
    "render_layout",
]
