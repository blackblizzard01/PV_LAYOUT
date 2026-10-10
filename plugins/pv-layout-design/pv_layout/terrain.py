"""
Terrain slope analysis, slope-exclusion, and earthworks volume estimation.

Design philosophy (faithful to RatedPower):
  "RatedPower considers the terrain to be flat when creating the final layout,
  which means that shading due to terrain slopes is not taken into account."
  — Design_Methodology §5.2.2

This means:
  - Row pitch is NOT adjusted for slope (flat-layout approach).
  - Structures on steep slopes are REMOVED (slope-exclusion mask).
  - Structures on intermediate slopes get earthworks (cut/fill).
  - The earthworks volume is estimated from the least-squares plane fit.

Sources
-------
- Slope formula: Topography_Analysis_Methodology Eq 2.4:
    slope(P1, P2) = ΔZ / sqrt((x2−x1)² + (y2−y1)²)
- Slope default thresholds: Design_Methodology §5.2.2 (color bands):
    green < 5 %, orange 5–10 %, red 10–15 %, purple > 15 %.
  Earthworks range example (Design §5.5.2): 10–25 % → install with cut/fill;
  > 25 % → remove.
- Pile installation / linear regression: Topography_Analysis_Methodology §2.1.
- Earthworks cut/fill (least-squares plane): Topography_Analysis_Methodology
  Eq 3.3 and 3.4.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


@dataclass
class DEMGrid:
    """
    Digital Elevation Model as a uniform raster grid.

    Parameters
    ----------
    elevations : 2D numpy array, shape (n_north, n_east).
                 elevations[0, 0] = SW corner.
    x_origin   : x (easting) of the WEST edge, metres.
    y_origin   : y (northing) of the SOUTH edge, metres.
    cell_size  : metres per grid cell (square cells assumed).
    """
    elevations: np.ndarray
    x_origin: float
    y_origin: float
    cell_size: float

    def elevation_at(self, x: float, y: float) -> float:
        """Bilinear interpolation of elevation at (x, y)."""
        col_f = (x - self.x_origin) / self.cell_size
        row_f = (y - self.y_origin) / self.cell_size
        n_rows, n_cols = self.elevations.shape
        col_f = float(np.clip(col_f, 0, n_cols - 1))
        row_f = float(np.clip(row_f, 0, n_rows - 1))
        c0, r0 = int(col_f), int(row_f)
        c1 = min(c0 + 1, n_cols - 1)
        r1 = min(r0 + 1, n_rows - 1)
        dc, dr = col_f - c0, row_f - r0
        z = (
            self.elevations[r0, c0] * (1 - dr) * (1 - dc)
            + self.elevations[r0, c1] * (1 - dr) * dc
            + self.elevations[r1, c0] * dr * (1 - dc)
            + self.elevations[r1, c1] * dr * dc
        )
        return float(z)


@dataclass
class SlopeResult:
    """Per-row slope values computed from a DEM."""
    ns_slope_pct: np.ndarray   # (n_rows,) North-South slope %
    ew_slope_pct: np.ndarray   # (n_rows,) East-West slope %
    max_slope_pct: np.ndarray  # (n_rows,) max(|NS|, |EW|)


@dataclass
class TerrainResult:
    """
    Result of applying terrain limits to a layout.

    Source: Design_Methodology §5.5 (topology limitation tool).
    """
    row_mask: np.ndarray         # (n_rows,) bool — True = keep row
    earthworks_mask: np.ndarray  # (n_rows,) bool — True = row needs earthworks
    n_removed: int
    n_earthworks: int
    n_kept: int
    cut_fill_volume_m3: float    # total estimated earthworks volume
    slopes: SlopeResult


def synthetic_dem(
    x_origin: float,
    y_origin: float,
    width_m: float,
    height_m: float,
    cell_size: float = 5.0,
    ns_slope_pct: float = 0.0,
    ew_slope_pct: float = 0.0,
    undulation_m: float = 0.0,
    seed: int = 42,
) -> DEMGrid:
    """
    Generate a synthetic planar DEM with optional random undulation.

    Useful for testing without real elevation data (SRTM etc.).

    Parameters
    ----------
    ns_slope_pct  : North-South slope as a percentage (positive = rising north).
    ew_slope_pct  : East-West slope (positive = rising east).
    undulation_m  : Peak-to-peak random undulation amplitude, metres.
    """
    n_cols = int(math.ceil(width_m / cell_size)) + 1
    n_rows = int(math.ceil(height_m / cell_size)) + 1
    xs = np.linspace(x_origin, x_origin + width_m, n_cols)
    ys = np.linspace(y_origin, y_origin + height_m, n_rows)
    XX, YY = np.meshgrid(xs, ys)

    Z = (
        (YY - y_origin) * ns_slope_pct / 100.0
        + (XX - x_origin) * ew_slope_pct / 100.0
    )
    if undulation_m > 0:
        rng = np.random.default_rng(seed)
        Z += rng.uniform(-undulation_m / 2, undulation_m / 2, Z.shape)

    return DEMGrid(elevations=Z, x_origin=x_origin, y_origin=y_origin, cell_size=cell_size)


def compute_slopes(
    dem: DEMGrid,
    row_centroids: np.ndarray,
    pitch: float,
    module_length: float,
    tilt_deg: float,
) -> SlopeResult:
    """
    Compute North-South and East-West slopes at each row position.

    N-S slope: elevation difference between the south edge and north edge of
    the row's ground footprint, divided by the footprint length (Eq 2.4).

    E-W slope: elevation difference across one module width at the row centre,
    divided by that distance.

    Source: Topography_Analysis_Methodology Eq 2.4:
      slope(P1, P2) = ΔZ / sqrt((x2-x1)² + (y2-y1)²)
    """
    footprint_ns = module_length * math.cos(math.radians(tilt_deg))
    ns_slopes = []
    ew_slopes = []

    for cx, cy in row_centroids:
        # N-S: south edge vs north edge of footprint
        y_s = cy - footprint_ns / 2.0
        y_n = cy + footprint_ns / 2.0
        z_s = dem.elevation_at(cx, y_s)
        z_n = dem.elevation_at(cx, y_n)
        ns = 100.0 * (z_n - z_s) / footprint_ns if footprint_ns > 0 else 0.0
        ns_slopes.append(ns)

        # E-W: half module-width east vs west
        delta_x = dem.cell_size
        z_e = dem.elevation_at(cx + delta_x, cy)
        z_w = dem.elevation_at(cx - delta_x, cy)
        ew = 100.0 * (z_e - z_w) / (2.0 * delta_x)
        ew_slopes.append(ew)

    ns_arr = np.array(ns_slopes)
    ew_arr = np.array(ew_slopes)
    return SlopeResult(
        ns_slope_pct=ns_arr,
        ew_slope_pct=ew_arr,
        max_slope_pct=np.maximum(np.abs(ns_arr), np.abs(ew_arr)),
    )


def _earthworks_volume(
    dem: DEMGrid,
    row_centroids: np.ndarray,
    earthworks_mask: np.ndarray,
    footprint_ns: float,
    row_width_m: float,
    n_sample: int = 20,
) -> float:
    """
    Estimate earthworks cut/fill volume using grid sampling (Eq 3.4).

    For each row flagged for earthworks:
      1. Fit a least-squares plane to terrain elevations under the row (Eq 3.3).
      2. The cut/fill at each sample point = |z_plane − z_dem|.
      3. Volume = (area / n_points) × Σ |dz|

    Source: Topography_Analysis_Methodology §3.4, Eq 3.3–3.4.
    """
    total_vol = 0.0
    for i, (cx, cy) in enumerate(row_centroids):
        if not earthworks_mask[i]:
            continue
        # Sample grid under row footprint
        xs = np.linspace(cx - row_width_m / 2, cx + row_width_m / 2, n_sample)
        ys = np.linspace(cy - footprint_ns / 2, cy + footprint_ns / 2, n_sample)
        XX, YY = np.meshgrid(xs, ys)
        Z_dem = np.array([[dem.elevation_at(x, y) for x in xs] for y in ys])

        # Least-squares plane fit (Eq 3.3): a·x + b·y + c = z
        pts = XX.ravel()
        pty = YY.ravel()
        ptz = Z_dem.ravel()
        A = np.column_stack([pts, pty, np.ones_like(pts)])
        try:
            coeffs, _, _, _ = np.linalg.lstsq(A, ptz, rcond=None)
        except np.linalg.LinAlgError:
            continue
        z_plane = A @ coeffs
        dz = np.abs(z_plane - ptz)
        area = footprint_ns * row_width_m
        vol = (area / len(ptz)) * dz.sum()
        total_vol += vol

    return total_vol


def apply_terrain_limits(
    row_centroids: np.ndarray,
    slopes: SlopeResult,
    dem: DEMGrid | None = None,
    module_length: float = 2.0,
    tilt_deg: float = 30.0,
    row_width_m: float = 200.0,
    max_ns_slope_pct: float = 25.0,
    earthworks_ns_min_pct: float = 10.0,
    max_ew_slope_pct: float = 10.0,
) -> TerrainResult:
    """
    Apply slope-exclusion criteria and flag rows needing earthworks.

    Approach: FLAT-LAYOUT — row pitch is NOT adjusted for slope.
    Source: Design_Methodology §5.2.2:
      "RatedPower considers the terrain to be flat when creating the final
      layout... shading due to terrain slopes is not taken into account."

    Exclusion logic (Design_Methodology §5.5.2 example):
      |NS slope| < earthworks_ns_min_pct     → install, no earthworks.
      earthworks_ns_min_pct ≤ |NS slope| < max_ns_slope_pct
                                             → install with earthworks.
      |NS slope| ≥ max_ns_slope_pct          → remove.
      |EW slope| ≥ max_ew_slope_pct          → remove.

    Default thresholds: Design §5.5.2 example (10–25 % earthworks range,
    >25 % removed).

    Parameters
    ----------
    dem            : DEMGrid for cut/fill volume estimation (optional; if None,
                     volume is returned as 0.0).
    row_width_m    : East-West row width, metres (used for volume estimate).
    """
    abs_ns = np.abs(slopes.ns_slope_pct)
    abs_ew = np.abs(slopes.ew_slope_pct)

    removed = (abs_ns >= max_ns_slope_pct) | (abs_ew >= max_ew_slope_pct)
    earthworks = (~removed) & (abs_ns >= earthworks_ns_min_pct)
    keep = ~removed

    n_rows = len(row_centroids)
    footprint_ns = module_length * math.cos(math.radians(tilt_deg))

    if dem is not None:
        vol = _earthworks_volume(
            dem, row_centroids, earthworks, footprint_ns, row_width_m
        )
    else:
        vol = 0.0

    return TerrainResult(
        row_mask=keep,
        earthworks_mask=earthworks,
        n_removed=int(removed.sum()),
        n_earthworks=int(earthworks.sum()),
        n_kept=int(keep.sum()),
        cut_fill_volume_m3=vol,
        slopes=slopes,
    )
