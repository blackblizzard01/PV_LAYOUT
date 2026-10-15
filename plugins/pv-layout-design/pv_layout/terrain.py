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

    def elevations_at(self, xs, ys) -> np.ndarray:
        """Vectorised bilinear interpolation (clips to grid edges; NaN cells propagate)."""
        n_rows, n_cols = self.elevations.shape
        col_f = np.clip((np.asarray(xs, float) - self.x_origin) / self.cell_size, 0, n_cols - 1)
        row_f = np.clip((np.asarray(ys, float) - self.y_origin) / self.cell_size, 0, n_rows - 1)
        return _bilinear(self.elevations, col_f, row_f)

    @property
    def x_coords(self) -> np.ndarray:
        return self.x_origin + np.arange(self.elevations.shape[1]) * self.cell_size

    @property
    def y_coords(self) -> np.ndarray:
        return self.y_origin + np.arange(self.elevations.shape[0]) * self.cell_size

    def slope_grids_pct(self) -> tuple[np.ndarray, np.ndarray]:
        """(N-S slope %, E-W slope %) on the grid; positive = rising north / east."""
        dz_dy, dz_dx = np.gradient(self.elevations, self.cell_size)
        return 100.0 * dz_dy, 100.0 * dz_dx


def _bilinear(arr: np.ndarray, col_f: np.ndarray, row_f: np.ndarray) -> np.ndarray:
    n_rows, n_cols = arr.shape
    c0 = np.floor(col_f).astype(int)
    r0 = np.floor(row_f).astype(int)
    c1 = np.minimum(c0 + 1, n_cols - 1)
    r1 = np.minimum(r0 + 1, n_rows - 1)
    dc, dr = col_f - c0, row_f - r0
    return (
        arr[r0, c0] * (1 - dr) * (1 - dc)
        + arr[r0, c1] * (1 - dr) * dc
        + arr[r1, c0] * dr * (1 - dc)
        + arr[r1, c1] * dr * dc
    )


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


# ══════════════════════════════════════════════════════════════════════════════
# Per-table terrain analysis + real DEM sources
# ══════════════════════════════════════════════════════════════════════════════

ACCEPTED, EARTHWORKS, EXCLUDED = 0, 1, 2
STATUS_NAMES = ("accepted", "earthworks", "excluded")


@dataclass(frozen=True)
class SlopeLimits:
    """
    Structure installation limits (RatedPower Topography panel, Design §5.5.2):
      |slope| ≤ install            → install as is
      install < |slope| ≤ earthworks → install after grading to the install limit
      |slope| > earthworks         → structure removed
    """
    ns_install_pct: float = 10.0
    ns_earthworks_pct: float = 15.0
    ew_install_pct: float = 5.0
    ew_earthworks_pct: float = 10.0

    def __post_init__(self):
        if self.ns_earthworks_pct < self.ns_install_pct or self.ew_earthworks_pct < self.ew_install_pct:
            raise ValueError("earthworks limit must be ≥ installation limit")


@dataclass
class TableTerrain:
    ns_slope_pct: np.ndarray    # (N,) fitted plane N-S gradient, % (+ = rising north)
    ew_slope_pct: np.ndarray    # (N,) fitted plane E-W gradient, % (+ = rising east)
    z_center: np.ndarray        # (N,) terrain elevation at table centre, m
    status: np.ndarray          # (N,) 0 accepted / 1 earthworks / 2 excluded
    cut_m3: np.ndarray          # (N,) excavation volume (earthworks tables only)
    fill_m3: np.ndarray         # (N,) fill volume (earthworks tables only)
    nan_tables: int = 0         # tables whose footprint had no DEM data

    def count(self, status: int) -> int:
        return int((self.status == status).sum())


def _footprint_offsets(width: float, depth: float, nx: int = 7, ny: int = 3):
    gx = np.linspace(-width / 2, width / 2, nx)
    gy = np.linspace(-depth / 2, depth / 2, ny)
    XX, YY = np.meshgrid(gx, gy)
    return XX.ravel(), YY.ravel()


def classify_tables(
    dem: DEMGrid,
    centers: np.ndarray,
    width: float,
    depth: float,
    limits: SlopeLimits = SlopeLimits(),
) -> TableTerrain:
    """
    Fit a least-squares plane under every table footprint (Topography Eq 3.3)
    and classify it against ``limits``.  Earthworks tables are graded to a plane
    through the fitted centre elevation whose gradients are clipped to the
    installation limits; cut/fill = Σ ±(z_dem − z_target) × cell area (Eq 3.4).

    Fully vectorised: the sample offsets are identical for every table, so one
    pseudo-inverse solves all plane fits at once.
    """
    n = len(centers)
    if n == 0:
        z = np.zeros(0)
        return TableTerrain(z, z, z, np.zeros(0, int), z, z)

    ox, oy = _footprint_offsets(width, depth)
    X = centers[:, 0:1] + ox[None, :]
    Y = centers[:, 1:2] + oy[None, :]
    Z = dem.elevations_at(X.ravel(), Y.ravel()).reshape(X.shape)  # (N, S)

    finite = np.isfinite(Z)
    bad = ~finite.all(axis=1)
    if bad.any():
        n_ok = finite.sum(axis=1, keepdims=True)
        row_mean = np.where(n_ok > 0, np.where(finite, Z, 0.0).sum(axis=1, keepdims=True)
                            / np.maximum(n_ok, 1), 0.0)
        Z = np.where(finite, Z, row_mean)

    A = np.column_stack([np.ones_like(ox), ox, oy])          # (S, 3)
    coef = np.linalg.pinv(A) @ Z.T                            # (3, N)
    z0, gx, gy = coef
    ns_pct, ew_pct = 100.0 * gy, 100.0 * gx

    a_ns, a_ew = np.abs(ns_pct), np.abs(ew_pct)
    status = np.full(n, ACCEPTED, int)
    status[(a_ns > limits.ns_install_pct) | (a_ew > limits.ew_install_pct)] = EARTHWORKS
    status[(a_ns > limits.ns_earthworks_pct) | (a_ew > limits.ew_earthworks_pct)] = EXCLUDED
    status[bad] = ACCEPTED

    gx_t = np.clip(gx, -limits.ew_install_pct / 100, limits.ew_install_pct / 100)
    gy_t = np.clip(gy, -limits.ns_install_pct / 100, limits.ns_install_pct / 100)
    Zt = z0[:, None] + gx_t[:, None] * ox[None, :] + gy_t[:, None] * oy[None, :]
    dz = Z - Zt
    cell_area = width * depth / len(ox)
    ew_mask = status == EARTHWORKS
    cut = np.where(ew_mask, np.clip(dz, 0, None).sum(axis=1) * cell_area, 0.0)
    fill = np.where(ew_mask, np.clip(-dz, 0, None).sum(axis=1) * cell_area, 0.0)

    return TableTerrain(
        ns_slope_pct=ns_pct, ew_slope_pct=ew_pct, z_center=z0,
        status=status, cut_m3=cut, fill_m3=fill, nan_tables=int(bad.sum()),
    )


def flat_dem(bounds, cell_size: float = 10.0, margin: float = 20.0) -> DEMGrid:
    x0, y0, x1, y1 = bounds
    nx = int(math.ceil((x1 - x0 + 2 * margin) / cell_size)) + 1
    ny = int(math.ceil((y1 - y0 + 2 * margin) / cell_size)) + 1
    return DEMGrid(np.zeros((ny, nx)), x0 - margin, y0 - margin, cell_size)


def resample_dem(sampler, bounds, cell_size: float, margin: float = 30.0) -> DEMGrid:
    """Build a local DEMGrid over ``bounds`` by evaluating ``sampler(xs, ys)``."""
    x0, y0, x1, y1 = bounds
    xs = np.arange(x0 - margin, x1 + margin + cell_size, cell_size)
    ys = np.arange(y0 - margin, y1 + margin + cell_size, cell_size)
    XX, YY = np.meshgrid(xs, ys)
    Z = np.asarray(sampler(XX.ravel(), YY.ravel()), float).reshape(XX.shape)
    return DEMGrid(Z, float(xs[0]), float(ys[0]), cell_size)


def smooth_dem(dem: DEMGrid, sigma_m: float) -> DEMGrid:
    """
    Gaussian-smooth a DEM.  SRTM heights are integer metres at ~30 m spacing;
    without smoothing, 1 m quantisation steps read as 3–10 % slopes over a
    single table depth.  sigma ≈ source resolution removes the artefact
    without flattening real landforms.
    """
    if sigma_m <= 0:
        return dem
    from scipy.ndimage import gaussian_filter
    Z = gaussian_filter(dem.elevations, sigma=sigma_m / dem.cell_size, mode="nearest")
    return DEMGrid(Z, dem.x_origin, dem.y_origin, dem.cell_size)


def _require_rasterio():
    try:
        import rasterio
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            'rasterio is required for GeoTIFF DEMs. Install with: '
            'pip install "pv-layout-design[dem]"'
        ) from exc
    return rasterio


def geotiff_sampler(path, local_to_raster_crs_factory):
    """
    Return (sampler, coverage_fn) for a GeoTIFF DEM.

    ``local_to_raster_crs_factory(crs_wkt)`` must return f(xs, ys) → raster CRS
    coordinates (e.g. ``GeoFrame.local_to_crs``).  Points are sampled in the
    raster's own CRS, so no raster reprojection is needed.
    """
    rasterio = _require_rasterio()
    with rasterio.open(path) as src:
        arr = src.read(1, masked=True).astype(float).filled(np.nan)
        if src.nodata is not None:
            arr[arr == src.nodata] = np.nan
        arr[arr < -1000] = np.nan
        inv = ~src.transform
        to_raster = local_to_raster_crs_factory(src.crs.to_wkt())

    def sampler(xs, ys):
        X, Y = to_raster(xs, ys)
        col = inv.a * X + inv.b * Y + inv.c - 0.5
        row = inv.d * X + inv.e * Y + inv.f - 0.5
        out_of = (col < -0.5) | (row < -0.5) | (col > arr.shape[1] - 0.5) | (row > arr.shape[0] - 0.5)
        z = _bilinear(arr, np.clip(col, 0, arr.shape[1] - 1), np.clip(row, 0, arr.shape[0] - 1))
        z[out_of] = np.nan
        return z

    return sampler


def _http_get(url: str, timeout: float = 60.0) -> bytes:
    import urllib.error
    import urllib.request
    req = urllib.request.Request(url, headers={"User-Agent": "pv-layout-design/0.3"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.read()
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"HTTP {exc.code} fetching {url.split('?')[0]}") from exc
    except (urllib.error.URLError, TimeoutError) as exc:
        raise RuntimeError(f"Network error fetching {url.split('?')[0]}: {exc}") from exc


def fetch_terrarium_dem(lon_min, lat_min, lon_max, lat_max, out_path, zoom: int = 13) -> str:
    """
    Download AWS Terrain Tiles (Terrarium encoding, SRTM-based, no API key)
    covering the bbox, mosaic them and write an EPSG:3857 GeoTIFF.
    elevation = R·256 + G + B/256 − 32768.   Cached: skips if out_path exists.
    """
    import io
    import pathlib
    out_path = pathlib.Path(out_path)
    if out_path.exists():
        return str(out_path)
    rasterio = _require_rasterio()
    from affine import Affine
    from PIL import Image

    n = 2 ** zoom

    def tile_xy(lon, lat):
        x = (lon + 180.0) / 360.0 * n
        lr = math.radians(lat)
        y = (1.0 - math.asinh(math.tan(lr)) / math.pi) / 2.0 * n
        return int(x), int(y)

    tx0, ty0 = tile_xy(lon_min, lat_max)
    tx1, ty1 = tile_xy(lon_max, lat_min)
    if (tx1 - tx0 + 1) * (ty1 - ty0 + 1) > 64:
        raise ValueError("DEM bbox too large for zoom level; lower `zoom`.")
    mosaic = np.zeros(((ty1 - ty0 + 1) * 256, (tx1 - tx0 + 1) * 256), np.float32)
    for ty in range(ty0, ty1 + 1):
        for tx in range(tx0, tx1 + 1):
            png = _http_get(f"https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{zoom}/{tx}/{ty}.png")
            rgb = np.asarray(Image.open(io.BytesIO(png)).convert("RGB"), np.float32)
            z = rgb[..., 0] * 256 + rgb[..., 1] + rgb[..., 2] / 256 - 32768
            mosaic[(ty - ty0) * 256:(ty - ty0 + 1) * 256, (tx - tx0) * 256:(tx - tx0 + 1) * 256] = z

    half = 20037508.342789244
    tile_m = 2 * half / n
    res = tile_m / 256
    transform = Affine(res, 0.0, tx0 * tile_m - half, 0.0, -res, half - ty0 * tile_m)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_path, "w", driver="GTiff", height=mosaic.shape[0], width=mosaic.shape[1],
                       count=1, dtype="float32", crs="EPSG:3857", transform=transform,
                       nodata=-32768, compress="deflate") as dst:
        dst.write(mosaic, 1)
    return str(out_path)


def fetch_opentopography_dem(lon_min, lat_min, lon_max, lat_max, out_path,
                             api_key: str | None = None, demtype: str = "SRTMGL1") -> str:
    """Download a GeoTIFF from the OpenTopography globaldem API (needs a free API key)."""
    import os
    import pathlib
    out_path = pathlib.Path(out_path)
    if out_path.exists():
        return str(out_path)
    api_key = api_key or os.environ.get("OPENTOPOGRAPHY_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OpenTopography needs an API key: set OPENTOPOGRAPHY_API_KEY or use "
            "terrain source 'terrarium' (no key required)."
        )
    url = ("https://portal.opentopography.org/API/globaldem?"
           f"demtype={demtype}&south={lat_min}&north={lat_max}&west={lon_min}&east={lon_max}"
           f"&outputFormat=GTiff&API_Key={api_key}")
    data = _http_get(url, timeout=120)
    if not data.startswith(b"II") and not data.startswith(b"MM"):
        raise RuntimeError(f"OpenTopography returned a non-GeoTIFF response: {data[:200]!r}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(data)
    return str(out_path)
