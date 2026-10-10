"""
Row spacing, panel placement, and Ground Coverage Ratio.

Sources
-------
- Sun position: pvlib.solarposition.get_solarposition (BSD-3 licence).
- Row-pitch formula: Duffie & Beckman, *Solar Engineering of Thermal
  Processes*, 4th ed. (2013), §1.7 "Shadow geometry for tilted surfaces".
  pitch = L·cos(tilt) + L·sin(tilt)/tan(β)
- Shading-time criterion: Marion & Dobos, NREL/TP-6A20-62641 (2013).
- Tilt recommendation table: RatedPower Design_Methodology §5.3.3.4.
- GCR/pitch relationship: RatedPower Design_Methodology §5.4.3.
- Polygon site + setbacks: RatedPower Design_Methodology §5.2.1, §5.4.6.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
import pandas as pd
import pvlib

try:
    from shapely.geometry import Polygon, LineString, box as shapely_box
    HAS_SHAPELY = True
except ImportError:  # pragma: no cover
    HAS_SHAPELY = False


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
    row_centroids: np.ndarray  # (n_rows, 2) — (x, y) per row, metres from SW corner
    n_rows: int
    n_per_row: int             # modules per row (maximum across all rows for polygon sites)
    total_modules: int
    total_kwp: float
    pitch: float               # row pitch, metres (front-edge to front-edge)
    gcr: float                 # Ground Coverage Ratio = module_length / pitch
    beta_deg: float            # critical sun elevation angle used (NaN if gcr input)
    target_fits: bool
    target_kwp: float
    area_used_m2: float
    # Optional fields added in Layer 2/3 — None means rectangle case (all rows equal n_per_row)
    n_per_row_array: np.ndarray | None = None  # per-row module counts (polygon sites)


# ── Layer 2 — F3: Tilt recommendation ────────────────────────────────────────

def recommend_tilt(lat: float) -> float:
    """
    Latitude-based tilt angle recommendation for fixed-tilt PV structures.

    Table from RatedPower Design_Methodology §5.3.3.4 (Mounting System):
      Latitude < 5°   →  5°
      5°–10°          →  latitude
      10°–20°         →  latitude × 0.9
      >20°            →  latitude × 0.85

    Examples: Ahmedabad 23°N → 19.6°; London 51.5°N → 43.8°.

    Source: RatedPower Design_Methodology.pdf §5.3.3.4, extracted via
    reference/Design_Methodology.txt.
    """
    a = abs(lat)
    if a < 5.0:
        return 5.0
    elif a <= 10.0:
        return float(a)
    elif a <= 20.0:
        return round(a * 0.9, 1)
    else:
        return round(a * 0.85, 1)


# ── Layer 1 helpers (unchanged) ───────────────────────────────────────────────

def _winter_solstice_min_elevation(
    lat: float,
    lon: float,
    hours_from_noon: float = 4.0,
) -> float:
    """
    Minimum sun elevation angle (degrees) at ±hours_from_noon from solar noon
    on the winter solstice.

    Uses pvlib.solarposition.get_solarposition (pvlib BSD-3 licence).
    Winter solstice: 21 Dec (NH), 21 Jun (SH).
    ±4 h criterion: Marion & Dobos, NREL/TP-6A20-62641 (2013).
    """
    year = 2024
    month_day = "12-21" if lat >= 0 else "06-21"
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
    Minimum row pitch (m, front-edge to front-edge) for no inter-row shading
    at sun elevation beta_deg.

    pitch = L·cos(tilt) + L·sin(tilt)/tan(β)
    Source: Duffie & Beckman, Solar Engineering of Thermal Processes (2013), §1.7.
    """
    tilt = math.radians(tilt_deg)
    beta = math.radians(beta_deg)
    return module_length * (math.cos(tilt) + math.sin(tilt) / math.tan(beta))


def compute_gcr(module_length: float, pitch: float) -> float:
    """
    Ground Coverage Ratio = module_length / pitch.
    Typical utility-scale range: 0.35–0.45.
    Source: RatedPower Design_Methodology §5.4.3.
    """
    return module_length / pitch


# ── Layer 1 — rectangle placement (unchanged signature) ──────────────────────

def place_panels(
    site: SiteSpec,
    module: ModuleSpec,
    tilt_deg: float,
    target_kwp: float,
    hours_from_noon: float = 4.0,
    gcr: float | None = None,
) -> LayoutResult:
    """
    Pack the minimum number of complete east-west rows needed to meet
    target_kwp into the rectangular site.

    Layer 1 parameters (unchanged):
      site, module, tilt_deg, target_kwp, hours_from_noon

    Layer 2 (F4) — gcr:
      If given, use pitch = module.length / gcr and skip the sun-angle
      calculation.  Enables direct GCR control without a shading-time target.
      Source: RatedPower Design_Methodology §5.4.3 (GCR/pitch/clearance
      interdependence — the three values are linked; specifying any one fixes
      the others).

    Coordinate system: origin at SW corner; x = east, y = north.
    """
    if gcr is not None:
        # F4: direct GCR/pitch input
        if gcr <= 0 or gcr >= 1:
            raise ValueError(f"gcr must be in (0, 1); got {gcr}")
        pitch = module.length / gcr
        gcr_val = gcr
        beta_used = float("nan")
    else:
        beta = _winter_solstice_min_elevation(site.lat, site.lon, hours_from_noon)
        if beta <= 0:
            raise ValueError(
                f"Sun elevation ≤ 0° at ±{hours_from_noon} h from solar noon on "
                f"the winter solstice at lat={site.lat}°. "
                "Reduce hours_from_noon or choose a lower-latitude site."
            )
        pitch = compute_pitch(module.length, tilt_deg, beta)
        gcr_val = compute_gcr(module.length, pitch)
        beta_used = beta

    footprint_ns = module.length * math.cos(math.radians(tilt_deg))
    max_rows = int(site.height // pitch)
    n_per_row = int(site.width // module.width)

    if max_rows == 0 or n_per_row == 0:
        raise ValueError(
            "Site is too small to fit even a single row of panels with the "
            "given module dimensions and row pitch."
        )

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

    row_y = np.arange(n_rows) * pitch + footprint_ns / 2.0
    row_x = np.full(n_rows, site.width / 2.0)
    row_centroids = np.column_stack([row_x, row_y])

    return LayoutResult(
        row_centroids=row_centroids,
        n_rows=n_rows,
        n_per_row=n_per_row,
        total_modules=total_modules,
        total_kwp=total_kwp,
        pitch=pitch,
        gcr=gcr_val,
        beta_deg=beta_used,
        target_fits=target_fits,
        target_kwp=target_kwp,
        area_used_m2=n_rows * pitch * site.width,
    )


# ── Layer 3 — F1/F2: Polygon site + setbacks ─────────────────────────────────

def _require_shapely() -> None:
    if not HAS_SHAPELY:
        raise ImportError(
            "shapely ≥ 2.0 is required for polygon site support. "
            "Install it with: pip install shapely"
        )


def apply_setback(boundary: "Polygon", setback_m: float) -> "Polygon":
    """
    Shrink a site polygon inward by setback_m (negative buffer).

    Source: RatedPower Design_Methodology §5.4.6 (setbacks):
      distance from area perimeter to fence and to structures.
      Minimum fence-to-structure distance = 2 m.

    Returns the inner polygon.  Raises ValueError if the setback erases the
    entire boundary.
    """
    _require_shapely()
    if setback_m <= 0:
        return boundary
    inner = boundary.buffer(-setback_m, join_style=2)
    if inner.is_empty:
        raise ValueError(
            f"Setback of {setback_m} m erases the entire site polygon.  "
            "Use a smaller setback."
        )
    return inner


def _row_segments_at_y(
    boundary: "Polygon",
    y_centroid: float,
) -> list[tuple[float, float]]:
    """
    Return (x_start, x_end) intervals inside the polygon at y = y_centroid.
    Handles simple and multi-segment boundaries (e.g. L-shaped, U-shaped sites).
    """
    x0, _, x1, _ = boundary.bounds
    scan = LineString([(x0 - 1.0, y_centroid), (x1 + 1.0, y_centroid)])
    inter = boundary.intersection(scan)
    if inter.is_empty:
        return []
    gtype = inter.geom_type
    if gtype == "Point":
        return []
    if gtype == "LineString":
        xs = [c[0] for c in inter.coords]
        return [(min(xs), max(xs))]
    if gtype == "MultiLineString":
        segs = []
        for g in inter.geoms:
            xs = [c[0] for c in g.coords]
            segs.append((min(xs), max(xs)))
        return segs
    return []


def place_panels_polygon(
    boundary: "Polygon",
    lat: float,
    lon: float,
    module: ModuleSpec,
    tilt_deg: float,
    target_kwp: float,
    hours_from_noon: float = 4.0,
    gcr: float | None = None,
    setback_m: float = 0.0,
) -> LayoutResult:
    """
    Pack panel rows into an arbitrary polygon site (Layer 3 — F1/F2).

    Parameters
    ----------
    boundary   : shapely.Polygon in metres (x = east, y = north from origin).
    lat, lon   : site location in decimal degrees (for sun-angle calculation).
    setback_m  : distance from polygon boundary to keep clear of panels (F2).
                 Source: RatedPower Design_Methodology §5.4.6.
    gcr        : if given, use pitch = module.length / gcr (F4).

    Each y level is intersected with the (setback-shrunk) boundary and modules
    are placed in every continuous x-interval.  Rows from different intervals
    at the same y level become separate entries in row_centroids.

    The rectangle case is the special case where boundary = shapely.box(...).

    Source (polygon approach): RatedPower Design_Methodology §5.2.1
    (available areas defined as polygons; rows clipped to each area).
    """
    _require_shapely()

    usable = apply_setback(boundary, setback_m) if setback_m > 0 else boundary

    # ── Pitch (same logic as place_panels) ───────────────────────────────────
    if gcr is not None:
        if gcr <= 0 or gcr >= 1:
            raise ValueError(f"gcr must be in (0, 1); got {gcr}")
        pitch = module.length / gcr
        gcr_val = gcr
        beta_used = float("nan")
    else:
        beta = _winter_solstice_min_elevation(lat, lon, hours_from_noon)
        if beta <= 0:
            raise ValueError(
                f"Sun elevation ≤ 0° at ±{hours_from_noon} h from solar noon "
                f"at lat={lat}°."
            )
        pitch = compute_pitch(module.length, tilt_deg, beta)
        gcr_val = compute_gcr(module.length, pitch)
        beta_used = beta

    footprint_ns = module.length * math.cos(math.radians(tilt_deg))
    _, y_min, _, y_max = usable.bounds

    # ── Place rows south → north ──────────────────────────────────────────────
    rows_xy: list[tuple[float, float]] = []   # (cx, cy)
    n_per_row_list: list[int] = []

    y_front = y_min  # front edge of first row
    while y_front + footprint_ns <= y_max:
        cy = y_front + footprint_ns / 2.0
        for x_start, x_end in _row_segments_at_y(usable, cy):
            n = int((x_end - x_start) // module.width)
            if n > 0:
                cx = x_start + n * module.width / 2.0
                rows_xy.append((cx, cy))
                n_per_row_list.append(n)
        y_front += pitch

    if not rows_xy:
        raise ValueError(
            "No panel rows fit inside the polygon with the given pitch. "
            "Try a smaller setback, a different tilt, or a larger site."
        )

    row_centroids = np.array(rows_xy, dtype=float)
    n_per_row_arr = np.array(n_per_row_list, dtype=int)
    n_rows = len(rows_xy)
    total_modules = int(n_per_row_arr.sum())
    total_kwp = total_modules * module.power_wp / 1000.0
    n_per_row_max = int(n_per_row_arr.max())

    # ── Target capacity ───────────────────────────────────────────────────────
    target_fits = total_kwp >= target_kwp

    # If target fits, trim rows from the north until we just meet the target
    if target_fits and target_kwp > 0:
        cumulative = 0.0
        keep = 0
        for i in range(n_rows):
            cumulative += n_per_row_arr[i] * module.power_wp / 1000.0
            keep += 1
            if cumulative >= target_kwp:
                break
        row_centroids = row_centroids[:keep]
        n_per_row_arr = n_per_row_arr[:keep]
        n_rows = keep
        total_modules = int(n_per_row_arr.sum())
        total_kwp = total_modules * module.power_wp / 1000.0
        n_per_row_max = int(n_per_row_arr.max())

    area_used = n_rows * pitch * float(usable.bounds[2] - usable.bounds[0])

    return LayoutResult(
        row_centroids=row_centroids,
        n_rows=n_rows,
        n_per_row=n_per_row_max,
        total_modules=total_modules,
        total_kwp=total_kwp,
        pitch=pitch,
        gcr=gcr_val,
        beta_deg=beta_used,
        target_fits=target_fits,
        target_kwp=target_kwp,
        area_used_m2=area_used,
        n_per_row_array=n_per_row_arr,
    )
