"""
Row spacing, panel placement, and Ground Coverage Ratio.

Sources
-------
- Sun position: pvlib.solarposition.get_solarposition (BSD-3 licence).
  pvlib docs: https://pvlib-python.readthedocs.io/en/stable/reference/generated/pvlib.solarposition.get_solarposition.html
- Row-pitch derivation: Duffie & Beckman, *Solar Engineering of Thermal
  Processes*, 4th ed. (2013), §1.7 "Shadow geometry for tilted surfaces".
  pitch = L·cos(tilt) + L·sin(tilt)/tan(β)
  where β is the minimum acceptable sun elevation angle.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
import pandas as pd
import pvlib


@dataclass(frozen=True)
class ModuleSpec:
    """Physical and electrical spec for one PV module."""
    length: float    # metres — dimension along the tilt direction (portrait)
    width: float     # metres — east-west (short) dimension
    power_wp: float  # rated DC power, Wp


@dataclass(frozen=True)
class SiteSpec:
    """Rectangular site geometry and location."""
    width: float   # metres, east-west
    height: float  # metres, north-south
    lat: float     # degrees, positive = north
    lon: float     # degrees, positive = east


class LayoutResult(NamedTuple):
    row_centroids: np.ndarray  # shape (n_rows, 2) — (x, y) per row, metres from SW corner
    n_rows: int
    n_per_row: int             # modules per row (east-west)
    total_modules: int
    total_kwp: float           # DC capacity of placed panels, kWp
    pitch: float               # row pitch, metres (front-edge to front-edge)
    gcr: float                 # Ground Coverage Ratio = module_length / pitch
    beta_deg: float            # critical sun elevation angle used, degrees
    target_fits: bool          # True if placed capacity ≥ target_kwp
    target_kwp: float          # requested target capacity, kWp
    area_used_m2: float        # north-south extent × site width


def _winter_solstice_min_elevation(
    lat: float,
    lon: float,
    hours_from_noon: float = 4.0,
) -> float:
    """
    Minimum sun elevation angle (degrees) at ±hours_from_noon from solar noon
    on the winter solstice.

    Uses pvlib.solarposition.get_solarposition (pvlib BSD-3 licence).
    Winter solstice: 21 Dec in the northern hemisphere, 21 Jun in the southern.

    The ±4 h from solar noon convention is the standard "no shading for the
    central 8 h of a winter day" criterion — see Marion & Dobos (NREL/TP-6A20-
    62641, 2013) for a citable reference.
    """
    year = 2024
    month_day = "12-21" if lat >= 0 else "06-21"

    # Approximate solar noon in UTC: UTC = 12:00 local solar time − lon/15 h
    solar_noon_utc = 12.0 - lon / 15.0

    times = pd.DatetimeIndex([
        pd.Timestamp(f"{year}-{month_day}", tz="UTC")
        + pd.Timedelta(hours=solar_noon_utc - hours_from_noon),
        pd.Timestamp(f"{year}-{month_day}", tz="UTC")
        + pd.Timedelta(hours=solar_noon_utc + hours_from_noon),
    ])

    solpos = pvlib.solarposition.get_solarposition(times, lat, lon)
    return float(solpos["elevation"].min())


def compute_pitch(module_length: float, tilt_deg: float, beta_deg: float) -> float:
    """
    Minimum row pitch (metres, front-edge to front-edge) for no inter-row
    shading at sun elevation beta_deg.

    Derivation (Duffie & Beckman §1.7):
      - Height of panel top edge above its base = L · sin(tilt)
      - Shadow of that top edge extends L·sin(tilt)/tan(β) beyond the panel base
      - Panel ground footprint = L · cos(tilt)
      - pitch = footprint + shadow extension
             = L·cos(tilt) + L·sin(tilt)/tan(β)
    """
    tilt = math.radians(tilt_deg)
    beta = math.radians(beta_deg)
    return module_length * (math.cos(tilt) + math.sin(tilt) / math.tan(beta))


def compute_gcr(module_length: float, pitch: float) -> float:
    """
    Ground Coverage Ratio = module_length / pitch.
    Typical utility-scale range: 0.35–0.45.
    """
    return module_length / pitch


def place_panels(
    site: SiteSpec,
    module: ModuleSpec,
    tilt_deg: float,
    target_kwp: float,
    hours_from_noon: float = 4.0,
) -> LayoutResult:
    """
    Pack the minimum number of complete east-west rows needed to meet
    target_kwp into the rectangular site, respecting the inter-row shading
    constraint at ±hours_from_noon from solar noon on the winter solstice.

    Coordinate system: origin at the south-west corner of the site.
    x = east (+east), y = north (+north).
    Row centroids are at the ground-projected centre of each row's footprint.

    If target_kwp cannot fit, places as many rows as possible and sets
    target_fits = False.
    """
    beta = _winter_solstice_min_elevation(site.lat, site.lon, hours_from_noon)
    if beta <= 0:
        raise ValueError(
            f"Sun elevation ≤ 0° at ±{hours_from_noon} h from solar noon on "
            f"the winter solstice at lat={site.lat}°. "
            "Reduce hours_from_noon or choose a lower-latitude site."
        )

    pitch = compute_pitch(module.length, tilt_deg, beta)
    gcr = compute_gcr(module.length, pitch)

    # North-south ground footprint of one tilted module row
    footprint_ns = module.length * math.cos(math.radians(tilt_deg))

    # Maximum rows and modules that physically fit in the site
    max_rows = int(site.height // pitch)
    n_per_row = int(site.width // module.width)

    if max_rows == 0 or n_per_row == 0:
        raise ValueError(
            "Site is too small to fit even a single row of panels with the "
            "given module dimensions and row pitch."
        )

    # Only place enough complete rows to reach (or exceed) the target
    target_modules = math.ceil(target_kwp * 1000.0 / module.power_wp)
    rows_needed = math.ceil(target_modules / n_per_row)

    if rows_needed <= max_rows:
        n_rows = rows_needed
        target_fits = True
    else:
        n_rows = max_rows
        target_fits = False

    total_modules = n_rows * n_per_row
    total_kwp = total_modules * module.power_wp / 1000.0

    # Row i: front edge at y = i * pitch; centroid at y = i * pitch + footprint_ns / 2
    row_y = np.arange(n_rows) * pitch + footprint_ns / 2.0
    row_x = np.full(n_rows, site.width / 2.0)  # centred east-west
    row_centroids = np.column_stack([row_x, row_y])

    area_used = n_rows * pitch * site.width

    return LayoutResult(
        row_centroids=row_centroids,
        n_rows=n_rows,
        n_per_row=n_per_row,
        total_modules=total_modules,
        total_kwp=total_kwp,
        pitch=pitch,
        gcr=gcr,
        beta_deg=beta,
        target_fits=target_fits,
        target_kwp=target_kwp,
        area_used_m2=area_used,
    )
