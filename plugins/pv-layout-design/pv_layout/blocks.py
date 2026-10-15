"""
Electrical layout on top of a TableLayout.

  strings  → tables hold an integer number of strings
  blocks   → tables grouped into inverter blocks by recursive capacity bisection
             (split on the longer bbox axis at the capacity quantile; deterministic,
             compact, balanced to within one table)
  inverter → "string": one inverter per block, in the nearest row gap
             "central": one central inverter per block inside its power station;
                         strings → combiner boxes → inverter
  PS       → power stations host transformers; string-inverter blocks are
             grouped into stations of ≤ ps_kva.  Tables under a PS footprint
             are removed (RatedPower behaviour).
  cables   → Manhattan routes: along the row gap, then north-south.

Source: RatedPower Design_Methodology §5.3.4 (power stations), Electrical
Methodology §1–3 (strings, combiner boxes, LV/MV cabling).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import shapely
from shapely.ops import nearest_points

from .tables import TableLayout


# ── Partitioning ──────────────────────────────────────────────────────────────

def bisect_partition(points: np.ndarray, weights: np.ndarray, k: int) -> np.ndarray:
    """Split points into k groups of ≈ equal weight by recursive axis bisection."""
    n = len(points)
    labels = np.zeros(n, int)
    if n == 0 or k <= 1:
        return labels
    k = min(k, n)
    counter = [0]

    def rec(idx: np.ndarray, kk: int):
        if kk <= 1 or len(idx) <= 1:
            labels[idx] = counter[0]
            counter[0] += 1
            return
        p = points[idx]
        span = p.max(axis=0) - p.min(axis=0)
        axis = int(np.argmax(span))
        order = idx[np.argsort(p[:, axis], kind="stable")]
        k1 = kk // 2
        cw = np.cumsum(weights[order])
        cut = int(np.searchsorted(cw, cw[-1] * k1 / kk, side="left")) + 1
        cut = min(max(cut, k1), len(order) - (kk - k1))
        rec(order[:cut], k1)
        rec(order[cut:], kk - k1)

    rec(np.arange(n), k)
    return labels


def partition_by_capacity(points, weights, target, components=None, max_size=None) -> np.ndarray:
    """
    Fewest groups such that every group's weight ≤ ``target`` (and size ≤
    ``max_size``); each connected component is split separately so no group
    spans two disjoint site areas.  Items are indivisible, so k starts at
    ceil(W / target) and grows until the cap holds.
    """
    points = np.asarray(points, float)
    weights = np.asarray(weights, float)
    components = np.zeros(len(points), int) if components is None else np.asarray(components)
    labels = np.full(len(points), -1, int)
    nxt = 0
    tol = 1e-9 * max(target, 1.0)
    for c in np.unique(components):
        m = np.where(components == c)[0]
        k = max(1, math.ceil(weights[m].sum() / target - 1e-9))
        if max_size is not None:
            k = max(k, math.ceil(len(m) / max_size))
        while True:
            sub = bisect_partition(points[m], weights[m], k)
            sums = np.bincount(sub, weights=weights[m])
            sizes = np.bincount(sub)
            over = (sums > target + tol) & (sizes > 1)
            if max_size is not None:
                over |= sizes > max_size
            if not over.any() or k >= len(m):
                break
            k += 1
        labels[m] = sub + nxt
        nxt += sub.max() + 1
    return labels


# ── Routing ───────────────────────────────────────────────────────────────────

def gap_y(layout: TableLayout, y: np.ndarray) -> np.ndarray:
    """y of the row gap (corridor) just north of the table row containing y."""
    return y + layout.depth / 2 + (layout.pitch - layout.depth) / 2


def manhattan_route(a, b, corridor_y):
    """Polyline a → (a.x, corridor) → (b.x, corridor) → b and its length."""
    pts = np.array([a, (a[0], corridor_y), (b[0], corridor_y), b], float)
    return pts, float(np.abs(np.diff(pts, axis=0)).sum())


# ── Data model ────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ElectricalSpec:
    inverter_type: str = "string"        # "string" | "central"
    inverter_ac_kw: float = 300.0
    dc_ac_ratio: float = 1.25
    ps_kva: float = 6000.0               # transformer rating per power station
    ps_footprint_m: tuple[float, float] = (12.2, 2.5)
    ps_clearance_m: float = 1.0
    combiner_max_strings: int = 24      # central topology only
    power_factor: float = 0.98

    def __post_init__(self):
        if self.inverter_type not in ("string", "central"):
            raise ValueError("inverter_type must be 'string' or 'central'")
        if self.dc_ac_ratio <= 0 or self.inverter_ac_kw <= 0 or self.ps_kva <= 0:
            raise ValueError("inverter_ac_kw, dc_ac_ratio and ps_kva must be > 0")
        if self.inverter_ac_kw > self.ps_kva:
            raise ValueError(f"inverter {self.inverter_ac_kw} kW exceeds power station "
                             f"{self.ps_kva} kVA")


@dataclass
class Route:
    kind: str                 # "string" | "dc_main" | "lv_ac" | "mv"
    points: np.ndarray        # (k, 2)
    length_m: float
    power_kw: float           # DC kWp (string/dc_main) or AC kW (lv_ac/mv) at STC
    n_parallel: int = 1       # identical circuits sharing the route (strings of one table)


@dataclass
class ElectricalLayout:
    layout: TableLayout                  # tables after PS footprint removal
    removed_for_ps: int
    strings_per_table: int
    string_kwp: float
    table_block: np.ndarray              # (N,) block / inverter index
    inverter_xy: np.ndarray              # (n_inv, 2)
    inverter_dc_kwp: np.ndarray          # (n_inv,)
    inverter_ps: np.ndarray              # (n_inv,) PS index
    ps_xy: np.ndarray                    # (n_ps, 2)
    ps_polys: list
    combiner_xy: np.ndarray              # (n_cb, 2) — empty for string topology
    table_combiner: np.ndarray           # (N,) — −1 for string topology
    substation_xy: np.ndarray            # (2,)
    kept_mask: np.ndarray                # (N_in,) True for input tables that survive PS placement
    routes: list[Route] = field(default_factory=list)
    spec: ElectricalSpec | None = None

    @property
    def n_inverters(self) -> int:
        return len(self.inverter_xy)

    @property
    def n_power_stations(self) -> int:
        return len(self.ps_xy)

    def routes_of(self, kind: str) -> list[Route]:
        return [r for r in self.routes if r.kind == kind]

    def cable_length_m(self, kind: str) -> float:
        return float(sum(r.length_m * r.n_parallel for r in self.routes_of(kind)))

    def ps_ac_kw(self) -> np.ndarray:
        ac = np.minimum(self.inverter_dc_kwp, self.spec.inverter_ac_kw)
        return np.array([ac[self.inverter_ps == p].sum() for p in range(self.n_power_stations)])


# ── Builder ───────────────────────────────────────────────────────────────────

def _snap_to_gap(layout: TableLayout, xy: np.ndarray) -> np.ndarray:
    """Move a point to the nearest row-gap centreline (keeps it off the tables)."""
    rows_y = np.unique(layout.centers[:, 1])
    gaps = gap_y(layout, rows_y)
    gaps = np.concatenate([[rows_y.min() - layout.pitch / 2], gaps])
    j = np.argmin(np.abs(gaps - xy[1]))
    return np.array([xy[0], gaps[j]])


def build_electrical(
    layout: TableLayout,
    module_power_wp: float,
    modules_per_string: int,
    spec: ElectricalSpec,
    components: np.ndarray | None = None,
    substation_xy=None,
    usable=None,
) -> ElectricalLayout:
    if layout.n_tables == 0:
        raise ValueError("No tables to connect.")
    if layout.modules_per_table % modules_per_string:
        raise ValueError(
            f"Table of {layout.modules_per_table} modules is not a whole number of "
            f"{modules_per_string}-module strings."
        )
    spt = layout.modules_per_table // modules_per_string
    string_kwp = modules_per_string * module_power_wp / 1000.0
    table_kwp = spt * string_kwp
    block_target = spec.inverter_ac_kw * spec.dc_ac_ratio
    if table_kwp > block_target:
        raise ValueError(f"One table ({table_kwp:.1f} kWp) exceeds an inverter block "
                         f"({block_target:.1f} kWp); use a larger inverter.")
    comps = np.zeros(layout.n_tables, int) if components is None else components

    # ── 1. inverter blocks ──
    w = np.full(layout.n_tables, table_kwp)
    blocks = partition_by_capacity(layout.centers, w, block_target, comps)

    # ── 2. power station sites; remove tables under each footprint ──
    removed = 0
    ps_w, ps_h = spec.ps_footprint_m
    if spec.inverter_type == "central":
        ps_groups = np.unique(blocks)                       # one PS per central inverter
        ps_of_block = {b: i for i, b in enumerate(ps_groups)}
    else:
        inv_per_ps = max(1, int(spec.ps_kva * spec.power_factor // spec.inverter_ac_kw))
        bc = np.array([layout.centers[blocks == b].mean(axis=0) for b in np.unique(blocks)])
        bcomp = np.array([np.bincount(comps[blocks == b]).argmax() for b in np.unique(blocks)])
        ps_lab = partition_by_capacity(bc, np.ones(len(bc)), inv_per_ps, bcomp, max_size=inv_per_ps)
        ps_of_block = {b: int(ps_lab[i]) for i, b in enumerate(np.unique(blocks))}
    n_ps = max(ps_of_block.values()) + 1

    ps_xy = np.zeros((n_ps, 2))
    ps_polys = []
    keep = np.ones(layout.n_tables, bool)
    for p in range(n_ps):
        members = np.isin(blocks, [b for b, q in ps_of_block.items() if q == p])
        c = _snap_to_gap(layout, layout.centers[members].mean(axis=0))
        poly = shapely.box(c[0] - ps_w / 2, c[1] - ps_h / 2, c[0] + ps_w / 2, c[1] + ps_h / 2)
        ps_xy[p] = c
        ps_polys.append(poly)
        clear = poly.buffer(spec.ps_clearance_m)
        hit = shapely.intersects(layout.boxes(), clear)
        keep &= ~hit
    removed = int((~keep).sum())
    layout = layout.subset(keep)
    blocks = blocks[keep]
    comps = comps[keep]
    # re-index blocks (a block may vanish if fully under a PS)
    uniq = np.unique(blocks)
    remap = {b: i for i, b in enumerate(uniq)}
    inv_ps = np.array([ps_of_block[b] for b in uniq])
    blocks = np.array([remap[b] for b in blocks])
    n_inv = len(uniq)

    inv_dc = np.array([table_kwp * (blocks == i).sum() for i in range(n_inv)])

    # ── 3. inverter positions ──
    if spec.inverter_type == "central":
        inv_xy = ps_xy[inv_ps].copy()
    else:
        inv_xy = np.array([_snap_to_gap(layout, layout.centers[blocks == i].mean(axis=0))
                           for i in range(n_inv)])

    # ── 4. substation ──
    if substation_xy is None:
        target = ps_xy.mean(axis=0)
        if usable is not None:
            pt = nearest_points(usable.boundary, shapely.Point(*target))[0]
            substation_xy = np.array([pt.x, pt.y])
        else:
            substation_xy = target
    substation_xy = np.asarray(substation_xy, float)

    routes: list[Route] = []
    corridors = gap_y(layout, layout.centers[:, 1])

    # ── 5. DC routes ──
    table_cb = np.full(layout.n_tables, -1)
    cb_xy = np.empty((0, 2))
    if spec.inverter_type == "central":
        cb_list = []
        for i in range(n_inv):
            m = np.where(blocks == i)[0]
            max_tables = max(1, spec.combiner_max_strings // spt)
            sub = partition_by_capacity(layout.centers[m], np.ones(len(m)), max_tables,
                                        max_size=max_tables)
            for s in np.unique(sub):
                mm = m[sub == s]
                cxy = _snap_to_gap(layout, layout.centers[mm].mean(axis=0))
                table_cb[mm] = len(cb_list)
                cb_list.append(cxy)
                for t in mm:
                    pts, L = manhattan_route(layout.centers[t], cxy, corridors[t])
                    routes.append(Route("string", pts, L, string_kwp, spt))
                pts, L = manhattan_route(cxy, inv_xy[i], cxy[1])
                routes.append(Route("dc_main", pts, L, table_kwp * len(mm)))
        cb_xy = np.array(cb_list)
    else:
        for t in range(layout.n_tables):
            i = blocks[t]
            pts, L = manhattan_route(layout.centers[t], inv_xy[i], corridors[t])
            routes.append(Route("string", pts, L, string_kwp, spt))
        # ── 6. LV AC: inverter → PS ──
        for i in range(n_inv):
            ac = min(inv_dc[i], spec.inverter_ac_kw)
            pts, L = manhattan_route(inv_xy[i], ps_xy[inv_ps[i]], inv_xy[i][1])
            routes.append(Route("lv_ac", pts, L, ac))

    # ── 7. MV: PS → substation (radial) ──
    ac_inv = np.minimum(inv_dc, spec.inverter_ac_kw)
    for p in range(n_ps):
        pts, L = manhattan_route(ps_xy[p], substation_xy, ps_xy[p][1])
        routes.append(Route("mv", pts, L, float(ac_inv[inv_ps == p].sum())))

    return ElectricalLayout(
        layout=layout, removed_for_ps=removed, strings_per_table=spt, string_kwp=string_kwp,
        table_block=blocks, inverter_xy=inv_xy, inverter_dc_kwp=inv_dc, inverter_ps=inv_ps,
        ps_xy=ps_xy, ps_polys=ps_polys, combiner_xy=cb_xy, table_combiner=table_cb,
        substation_xy=substation_xy, kept_mask=keep, routes=routes, spec=spec,
    )
