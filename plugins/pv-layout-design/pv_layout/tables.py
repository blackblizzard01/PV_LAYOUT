"""
Structure (table) layout: fixed-tilt tables tiled in east-west rows.

Each table is a rigid structure ``modules_high`` modules up-slope by
``modules_along`` modules east-west.  Rows are spaced by ``pitch`` (front
edge to front edge).  A table is kept only if its full ground footprint is
covered by the usable area (fixes centre-line-only clipping).

Pitch from the shading criterion uses the *table* slope length:
  pitch = L·cos(tilt) + L·sin(tilt)/tan(β)       (Duffie & Beckman §1.7)
Source for table/structure model: RatedPower Design_Methodology §5.3.3, §5.4.3.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np
import shapely
from shapely.geometry import LineString, MultiPolygon

from .geometry import ModuleSpec, _winter_solstice_min_elevation, compute_pitch


@dataclass(frozen=True)
class StructureSpec:
    modules_high: int = 2            # modules stacked up-slope (2P = 2 portrait)
    modules_along: int = 28          # modules side by side east-west
    orientation: str = "portrait"    # "portrait" | "landscape"
    module_gap_m: float = 0.02       # gap between adjacent modules
    gap_ew_m: float = 0.5            # gap between tables in the same row
    post_height_m: float = 0.8       # ground clearance of the lowest edge

    def __post_init__(self):
        if self.orientation not in ("portrait", "landscape"):
            raise ValueError("orientation must be 'portrait' or 'landscape'")
        if self.modules_high < 1 or self.modules_along < 1:
            raise ValueError("modules_high and modules_along must be ≥ 1")

    @property
    def modules_per_table(self) -> int:
        return self.modules_high * self.modules_along

    def dims(self, module: ModuleSpec) -> tuple[float, float]:
        """(east-west width, up-slope length) of one table, metres."""
        up, across = (module.length, module.width) if self.orientation == "portrait" \
            else (module.width, module.length)
        width = self.modules_along * across + (self.modules_along - 1) * self.module_gap_m
        slope_len = self.modules_high * up + (self.modules_high - 1) * self.module_gap_m
        return width, slope_len


@dataclass
class TableLayout:
    centers: np.ndarray          # (N, 2) table centre (x, y) metres
    row_id: np.ndarray           # (N,) row index (south → north)
    width: float                 # E-W table width, m
    depth: float                 # N-S ground footprint, m
    slope_length: float          # up-slope table length, m
    modules_per_table: int
    pitch: float
    gcr: float
    tilt_deg: float
    beta_deg: float              # NaN if pitch came from a GCR input

    @property
    def n_tables(self) -> int:
        return len(self.centers)

    @property
    def n_modules(self) -> int:
        return self.n_tables * self.modules_per_table

    def boxes(self) -> np.ndarray:
        x, y = self.centers[:, 0], self.centers[:, 1]
        return shapely.box(x - self.width / 2, y - self.depth / 2,
                           x + self.width / 2, y + self.depth / 2)

    def subset(self, mask: np.ndarray) -> "TableLayout":
        mask = np.asarray(mask)
        return replace(self, centers=self.centers[mask], row_id=self.row_id[mask])

    def kwp(self, module_power_wp: float) -> float:
        return self.n_modules * module_power_wp / 1000.0


def _scan_intervals(area, y: float) -> list[tuple[float, float]]:
    x0, _, x1, _ = area.bounds
    inter = area.intersection(LineString([(x0 - 1, y), (x1 + 1, y)]))
    segs = []
    for g in getattr(inter, "geoms", [inter]):
        if g.is_empty or g.geom_type != "LineString":
            continue
        xs = [c[0] for c in g.coords]
        segs.append((min(xs), max(xs)))
    return sorted(segs)


def _intersect(a, b):
    out, i, j = [], 0, 0
    while i < len(a) and j < len(b):
        lo, hi = max(a[i][0], b[j][0]), min(a[i][1], b[j][1])
        if hi > lo:
            out.append((lo, hi))
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return out


def _tile(area, width, depth, pitch, gap, y_offset):
    _, y_min, _, y_max = area.bounds
    eps = 1e-6
    xs, ys, rows = [], [], []
    y, r = y_min + y_offset, 0
    while y + depth <= y_max + eps:
        ivs = _intersect(_scan_intervals(area, y + eps), _scan_intervals(area, y + depth - eps))
        ivs = _intersect(ivs, _scan_intervals(area, y + depth / 2))
        for a, b in ivs:
            n = int((b - a + gap + eps) // (width + gap))
            if n <= 0:
                continue
            used = n * width + (n - 1) * gap
            start = a + (b - a - used) / 2 + width / 2
            for k in range(n):
                xs.append(start + k * (width + gap))
                ys.append(y + depth / 2)
                rows.append(r)
        y += pitch
        r += 1
    if not xs:
        return np.empty((0, 2)), np.empty(0, int)
    centers = np.column_stack([xs, ys])
    boxes = shapely.box(centers[:, 0] - width / 2, centers[:, 1] - depth / 2,
                        centers[:, 0] + width / 2, centers[:, 1] + depth / 2)
    keep = shapely.covered_by(boxes, area)
    return centers[keep], np.asarray(rows)[keep]


def compute_table_pitch(slope_length, tilt_deg, lat, lon, gcr=None, hours_from_noon=4.0):
    """(pitch, gcr, beta_deg) for a table of given slope length."""
    if gcr is not None:
        if not 0 < gcr < 1:
            raise ValueError(f"gcr must be in (0, 1); got {gcr}")
        return slope_length / gcr, gcr, float("nan")
    beta = _winter_solstice_min_elevation(lat, lon, hours_from_noon)
    if beta <= 0:
        raise ValueError(
            f"Sun is below the horizon at ±{hours_from_noon} h on the winter solstice "
            f"(lat {lat}°). Reduce hours_from_noon or give a GCR instead."
        )
    pitch = compute_pitch(slope_length, tilt_deg, beta)
    return pitch, slope_length / pitch, beta


def layout_tables(
    usable: MultiPolygon,
    module: ModuleSpec,
    structure: StructureSpec,
    tilt_deg: float,
    lat: float,
    lon: float,
    gcr: float | None = None,
    hours_from_noon: float = 4.0,
    n_offsets: int = 4,
) -> TableLayout:
    """Fill the usable area with tables; tries ``n_offsets`` row phase shifts, keeps the best."""
    width, slope_len = structure.dims(module)
    depth = slope_len * math.cos(math.radians(tilt_deg))
    pitch, gcr_val, beta = compute_table_pitch(slope_len, tilt_deg, lat, lon, gcr, hours_from_noon)

    shapely.prepare(usable)
    best = None
    for k in range(max(1, n_offsets)):
        centers, rows = _tile(usable, width, depth, pitch, structure.gap_ew_m, k * pitch / n_offsets)
        if best is None or len(centers) > len(best[0]):
            best = (centers, rows)
    centers, rows = best
    if len(centers) == 0:
        _, y0, _, y1 = usable.bounds
        widest = max((b - a for p in usable.geoms
                      for a, b in _scan_intervals(p, p.centroid.y)), default=0.0)
        raise ValueError(
            f"No table fits: table {width:.1f} m × {depth:.1f} m, pitch {pitch:.2f} m, "
            f"usable area {usable.area / 1e4:.2f} ha, N-S extent {y1 - y0:.1f} m, "
            f"widest E-W interval {widest:.1f} m."
        )
    return TableLayout(
        centers=centers, row_id=rows, width=width, depth=depth, slope_length=slope_len,
        modules_per_table=structure.modules_per_table, pitch=pitch, gcr=gcr_val,
        tilt_deg=tilt_deg, beta_deg=beta,
    )


def trim_mask(layout: TableLayout, target_kwp: float, module_power_wp: float) -> np.ndarray:
    """Mask keeping the tables nearest the layout centroid until ``target_kwp`` is reached."""
    kwp_per_table = layout.modules_per_table * module_power_wp / 1000.0
    n_keep = math.ceil(target_kwp / kwp_per_table - 1e-9)
    keep = np.zeros(layout.n_tables, bool)
    if n_keep >= layout.n_tables:
        keep[:] = True
        return keep
    c = layout.centers.mean(axis=0)
    d = np.linalg.norm(layout.centers - c, axis=1)
    keep[np.argsort(d, kind="stable")[:n_keep]] = True
    return keep


def trim_to_target(layout: TableLayout, target_kwp: float, module_power_wp: float) -> TableLayout:
    return layout.subset(trim_mask(layout, target_kwp, module_power_wp))


def component_ids(layout: TableLayout, usable: MultiPolygon) -> np.ndarray:
    """Index of the usable polygon containing each table centre (−1 if none)."""
    out = np.full(layout.n_tables, -1, int)
    x, y = layout.centers[:, 0], layout.centers[:, 1]
    for i, poly in enumerate(usable.geoms):
        out[(out < 0) & shapely.contains_xy(poly, x, y)] = i
    return out
