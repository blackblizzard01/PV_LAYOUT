"""
Design configuration (TOML) and the end-to-end design pipeline.

    site → usable area → tables → terrain filter → (target trim) → strings →
    inverter blocks → power stations → cable routes → losses → BoQ → energy

Cable losses here use full circuit physics:
    DC:       P_loss = 2 · I² · R   (outgoing + return conductor)
    3-phase:  P_loss = 3 · I² · R   (three phase conductors)
with R = ρ(T) · L / A and ρ(T) from IEC 60228 (electrical._resistivity_at_t).
"""

from __future__ import annotations

import dataclasses
import json
import math
import pathlib
import time
import tomllib
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from shapely.geometry import LineString, MultiPolygon
from shapely.ops import unary_union

from .blocks import ElectricalLayout, ElectricalSpec, build_electrical
from .electrical import _resistivity_at_t, select_conductor_area, string_sizing
from .energy import EnergyResult, annual_energy_yield
from .geometry import ModuleSpec, recommend_tilt
from .losses import transformer_loss
from .site import Site
from .tables import StructureSpec, TableLayout, component_ids, layout_tables, trim_mask
from .terrain import (
    ACCEPTED, EARTHWORKS, EXCLUDED, DEMGrid, SlopeLimits, TableTerrain, classify_tables,
    fetch_opentopography_dem, fetch_terrarium_dem, flat_dem, geotiff_sampler,
    resample_dem, smooth_dem, synthetic_dem,
)


# ── Configuration ─────────────────────────────────────────────────────────────

@dataclass
class SiteCfg:
    file: str = "site.geojson"
    name: str | None = None
    perimeter_setback_m: float = 10.0
    restricted_setback_m: float = 5.0
    road_spacing_m: float = 0.0
    road_width_m: float = 5.0
    substation_lonlat: list | None = None


@dataclass
class ModuleCfg:
    length_m: float = 2.278
    width_m: float = 1.134
    power_wp: float = 550.0
    voc_v: float = 49.9
    vmp_v: float = 41.96
    voc_tc_pct: float = -0.27
    vmp_tc_pct: float = -0.35
    gamma_pmax_pct: float = -0.35
    noct_c: float = 45.0


@dataclass
class StructureCfg:
    modules_high: int = 2
    orientation: str = "portrait"
    strings_per_table: int = 2
    gap_ew_m: float = 0.5
    module_gap_m: float = 0.02
    post_height_m: float = 0.8
    tilt_deg: float | str = "auto"
    gcr: float | None = None
    hours_from_noon: float = 4.0
    pile_spacing_m: float = 6.0


@dataclass
class LayoutCfg:
    mode: str = "max"                 # "max" | "target"
    target_kwp: float | None = None


@dataclass
class TerrainCfg:
    source: str = "flat"              # flat | synthetic | geotiff | terrarium | opentopography
    path: str | None = None
    zoom: int = 13
    cell_size_m: float = 10.0
    smooth_m: float = 30.0            # Gaussian sigma for real DEMs (SRTM ≈ 30 m)
    ns_slope_pct: float = 0.0
    ew_slope_pct: float = 0.0
    undulation_m: float = 0.0
    seed: int = 42
    ns_install_pct: float = 10.0
    ns_earthworks_pct: float = 15.0
    ew_install_pct: float = 5.0
    ew_earthworks_pct: float = 10.0
    exclude_steep: bool = True


@dataclass
class ElectricalCfg:
    system_voltage_v: float = 1500.0
    mppt_min_v: float = 500.0
    mppt_max_v: float = 1500.0
    t_min_c: float = -5.0
    t_max_c: float = 45.0
    inverter_type: str = "string"
    inverter_ac_kw: float = 300.0
    dc_ac_ratio: float = 1.25
    inverter_efficiency_pct: float = 98.5
    combiner_max_strings: int = 24
    ps_kva: float = 6000.0
    ps_footprint_m: list = field(default_factory=lambda: [12.2, 2.5])
    tx_no_load_kw: float = 5.0
    tx_load_loss_kw: float = 45.0
    lv_ac_voltage_v: float = 800.0
    mv_voltage_v: float = 33000.0
    power_factor: float = 0.98
    string_cable_material: str = "copper"
    string_cable_mm2: float = 6.0
    dc_main_cable_material: str = "aluminum"
    dc_main_cable_mm2: float = 300.0
    lv_ac_cable_material: str = "aluminum"
    lv_ac_cable_mm2: float = 240.0
    mv_cable_material: str = "aluminum"
    mv_cable_mm2: float = 150.0
    cable_temp_c: float = 70.0
    trench_depth_m: float = 0.8
    trench_width_m: float = 0.6


@dataclass
class EnergyCfg:
    source: str = "pvgis"             # pvgis | file | clearsky | none
    tmy_file: str = "pvgis_tmy.csv"
    soiling_pct: float = 2.0
    mismatch_pct: float = 1.0
    lid_pct: float = 1.0
    availability_pct: float = 0.5
    iam_ar: float | None = 0.16       # Martin & Ruiz angular loss coefficient (glass)
    pv_model: str = "huld"            # "huld" (Faiman + Huld, as PVGIS) | "linear" (NOCT + γ)
    horizon: bool = True              # far-horizon shading from PVGIS terrain profile
    horizon_file: str = "pvgis_horizon.csv"
    row_shading: bool = True


@dataclass
class OutputCfg:
    png: bool = True
    map_html: bool = True
    view3d_html: bool = True
    dpi: int = 150
    z_exaggeration: float = 3.0


_SECTIONS = {
    "site": SiteCfg, "module": ModuleCfg, "structure": StructureCfg, "layout": LayoutCfg,
    "terrain": TerrainCfg, "electrical": ElectricalCfg, "energy": EnergyCfg, "output": OutputCfg,
}


@dataclass
class DesignConfig:
    site: SiteCfg = field(default_factory=SiteCfg)
    module: ModuleCfg = field(default_factory=ModuleCfg)
    structure: StructureCfg = field(default_factory=StructureCfg)
    layout: LayoutCfg = field(default_factory=LayoutCfg)
    terrain: TerrainCfg = field(default_factory=TerrainCfg)
    electrical: ElectricalCfg = field(default_factory=ElectricalCfg)
    energy: EnergyCfg = field(default_factory=EnergyCfg)
    output: OutputCfg = field(default_factory=OutputCfg)
    base_dir: pathlib.Path = field(default_factory=pathlib.Path.cwd)

    @classmethod
    def from_toml(cls, path: str | pathlib.Path) -> "DesignConfig":
        path = pathlib.Path(path)
        try:
            data = tomllib.loads(path.read_text(encoding="utf-8"))
        except tomllib.TOMLDecodeError as exc:
            raise ValueError(f"{path}: invalid TOML — {exc}") from exc
        return cls.from_dict(data, base_dir=path.parent)

    @classmethod
    def from_dict(cls, data: dict, base_dir=None) -> "DesignConfig":
        unknown = set(data) - set(_SECTIONS)
        if unknown:
            raise ValueError(f"Unknown config section(s) {sorted(unknown)}; valid: {sorted(_SECTIONS)}")
        kwargs = {name: _build(sc, data.get(name, {}), name) for name, sc in _SECTIONS.items()}
        cfg = cls(**kwargs, base_dir=pathlib.Path(base_dir or pathlib.Path.cwd()))
        cfg.validate()
        return cfg

    def path(self, rel: str) -> pathlib.Path:
        p = pathlib.Path(rel)
        return p if p.is_absolute() else self.base_dir / p

    def validate(self) -> None:
        if self.layout.mode not in ("max", "target"):
            raise ValueError("[layout] mode must be 'max' or 'target'")
        if self.layout.mode == "target" and not self.layout.target_kwp:
            raise ValueError("[layout] mode='target' needs target_kwp")
        if self.terrain.source not in ("flat", "synthetic", "geotiff", "terrarium", "opentopography"):
            raise ValueError(f"[terrain] unknown source '{self.terrain.source}'")
        if self.energy.source not in ("pvgis", "file", "clearsky", "none"):
            raise ValueError(f"[energy] unknown source '{self.energy.source}'")
        t = self.structure.tilt_deg
        if not (t == "auto" or isinstance(t, (int, float))):
            raise ValueError("[structure] tilt_deg must be a number or 'auto'")
        if isinstance(t, (int, float)) and not 0 <= t < 90:
            raise ValueError("[structure] tilt_deg must be in [0, 90)")


def _build(cls, values: dict, section: str):
    if not isinstance(values, dict):
        raise ValueError(f"[{section}] must be a table")
    names = {f.name: f for f in dataclasses.fields(cls)}
    unknown = set(values) - set(names)
    if unknown:
        raise ValueError(f"[{section}] unknown key(s) {sorted(unknown)}; valid: {sorted(names)}")
    out = {}
    for k, v in values.items():
        default = names[k].default
        if isinstance(default, float) and isinstance(v, int) and not isinstance(v, bool):
            v = float(v)
        out[k] = v
    return cls(**out)


# ── Result ────────────────────────────────────────────────────────────────────

@dataclass
class DesignResult:
    config: DesignConfig
    site: Site
    usable: object
    module: ModuleSpec
    structure: StructureSpec
    modules_per_string: int
    tables_all: TableLayout
    terrain_all: TableTerrain
    dem: DEMGrid
    elec: ElectricalLayout
    status: np.ndarray            # terrain status of final tables (accepted / earthworks)
    losses: dict
    boq: dict
    energy: EnergyResult | None
    kpis: dict
    warnings: list[str]
    timings: dict

    @property
    def tables(self) -> TableLayout:
        return self.elec.layout

    def summary(self) -> dict:
        e = self.energy
        return {
            "site": self.site.name,
            "kpis": self.kpis,
            "losses_stc": self.losses,
            "boq": self.boq,
            "energy": None if e is None else {k: _jsonable(v) for k, v in dataclasses.asdict(e).items()},
            "warnings": self.warnings,
            "timings_s": self.timings,
        }


def _jsonable(v):
    if isinstance(v, (np.floating, np.integer)):
        return v.item()
    if isinstance(v, np.ndarray):
        return v.tolist()
    return v


# ── Pipeline ──────────────────────────────────────────────────────────────────

def run_design(cfg: DesignConfig) -> DesignResult:
    timings: dict[str, float] = {}
    t0 = time.perf_counter()

    def lap(name):
        nonlocal t0
        t = time.perf_counter()
        timings[name] = round(t - t0, 3)
        t0 = t

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")

        # 1. site + usable area
        site = Site.from_file(cfg.path(cfg.site.file), name=cfg.site.name)
        s = cfg.site
        usable = site.usable_area(s.perimeter_setback_m, s.restricted_setback_m,
                                  s.road_spacing_m, s.road_width_m)
        lap("site")

        # 2. strings + structure
        m, e = cfg.module, cfg.electrical
        module = ModuleSpec(length=m.length_m, width=m.width_m, power_wp=m.power_wp)
        ss = string_sizing(m.voc_v, m.vmp_v, m.voc_tc_pct, m.vmp_tc_pct, e.system_voltage_v,
                           e.mppt_min_v, e.mppt_max_v, e.t_min_c, e.t_max_c, m.noct_c)
        if not ss.sizing_ok:
            raise ValueError(f"String sizing impossible: N_min={ss.n_min} > N_max={ss.n_max}. "
                             "Check MPPT window / system voltage / temperatures.")
        n_str = ss.n_recommended
        st = cfg.structure
        mpt = st.strings_per_table * n_str
        if mpt % st.modules_high:
            raise ValueError(
                f"{st.strings_per_table} string(s) × {n_str} modules = {mpt} modules cannot fill "
                f"a {st.modules_high}-high table evenly; change strings_per_table or modules_high."
            )
        structure = StructureSpec(modules_high=st.modules_high, modules_along=mpt // st.modules_high,
                                  orientation=st.orientation, module_gap_m=st.module_gap_m,
                                  gap_ew_m=st.gap_ew_m, post_height_m=st.post_height_m)
        tilt = recommend_tilt(site.lat) if st.tilt_deg == "auto" else float(st.tilt_deg)
        tables_all = layout_tables(usable, module, structure, tilt, site.lat, site.lon,
                                   gcr=st.gcr, hours_from_noon=st.hours_from_noon)
        lap("tables")

        # 3. terrain
        dem = _load_dem(cfg, site)
        t = cfg.terrain
        limits = SlopeLimits(t.ns_install_pct, t.ns_earthworks_pct, t.ew_install_pct, t.ew_earthworks_pct)
        terrain_all = classify_tables(dem, tables_all.centers, tables_all.width, tables_all.depth, limits)
        if terrain_all.nan_tables:
            warnings.warn(f"{terrain_all.nan_tables} table(s) lie outside DEM coverage; treated as accepted")
        keep = terrain_all.status != EXCLUDED if t.exclude_steep else np.ones(tables_all.n_tables, bool)
        tables = tables_all.subset(keep)
        status = terrain_all.status[keep]
        if tables.n_tables == 0:
            raise ValueError("Every table was excluded by the terrain slope limits.")
        lap("terrain")

        # 4. target trim
        if cfg.layout.mode == "target":
            tm = trim_mask(tables, cfg.layout.target_kwp, module.power_wp)
            if tm.all() and tables.kwp(module.power_wp) < cfg.layout.target_kwp:
                warnings.warn(f"Target {cfg.layout.target_kwp:.0f} kWp does not fit; "
                              f"max is {tables.kwp(module.power_wp):.0f} kWp")
            tables, status = tables.subset(tm), status[tm]

        # 5. electrical
        espec = ElectricalSpec(
            inverter_type=e.inverter_type, inverter_ac_kw=e.inverter_ac_kw, dc_ac_ratio=e.dc_ac_ratio,
            ps_kva=e.ps_kva, ps_footprint_m=tuple(e.ps_footprint_m),
            combiner_max_strings=e.combiner_max_strings, power_factor=e.power_factor,
        )
        sub_xy = None
        if s.substation_lonlat:
            if site.frame is None:
                raise ValueError("substation_lonlat needs a geo-referenced site")
            x, y = site.frame.to_local([s.substation_lonlat[0]], [s.substation_lonlat[1]])
            sub_xy = (float(x[0]), float(y[0]))
        # Blocks may cross internal roads / restricted strips (cables go in ducts) but
        # stay within one available area (field / parcel).
        areas = MultiPolygon([p for _, p in site.available])
        elec = build_electrical(tables, module.power_wp, n_str, espec,
                                component_ids(tables, areas), sub_xy, usable)
        status = status[elec.kept_mask]
        lap("electrical")

        # 6. losses, BoQ
        losses = _losses_stc(cfg, elec, n_str)
        terrain_final = _subset_terrain(terrain_all, tables_all, elec)
        boq = _boq(cfg, site, elec, terrain_final, structure)
        lap("losses_boq")

        # 7. energy
        energy = _energy(cfg, site, elec, losses, tilt, module)
        lap("energy")

    warns = [str(w.message) for w in caught]
    kpis = _kpis(site, usable, tables_all, terrain_all, elec, status, n_str, tilt, energy, module)
    return DesignResult(
        config=cfg, site=site, usable=usable, module=module, structure=structure,
        modules_per_string=n_str, tables_all=tables_all, terrain_all=terrain_all, dem=dem,
        elec=elec, status=status, losses=losses, boq=boq, energy=energy, kpis=kpis,
        warnings=warns, timings=timings,
    )


def _subset_terrain(terrain_all, tables_all, elec):
    """Cut/fill totals over the final tables (matched back to the candidate set by centre)."""
    lookup = {tuple(np.round(c, 6)): i for i, c in enumerate(tables_all.centers)}
    sel = np.array([lookup[tuple(np.round(c, 6))] for c in elec.layout.centers], int)
    return {
        "cut_m3": float(terrain_all.cut_m3[sel].sum()),
        "fill_m3": float(terrain_all.fill_m3[sel].sum()),
        "index": sel,
    }


# ── DEM ───────────────────────────────────────────────────────────────────────

def _load_dem(cfg: DesignConfig, site: Site) -> DEMGrid:
    t = cfg.terrain
    b = site.bounds
    if t.source == "flat":
        return flat_dem(b, t.cell_size_m)
    if t.source == "synthetic":
        m = 30.0
        return synthetic_dem(b[0] - m, b[1] - m, b[2] - b[0] + 2 * m, b[3] - b[1] + 2 * m,
                             cell_size=t.cell_size_m, ns_slope_pct=t.ns_slope_pct,
                             ew_slope_pct=t.ew_slope_pct, undulation_m=t.undulation_m, seed=t.seed)
    if site.frame is None:
        raise ValueError(f"terrain source '{t.source}' needs a geo-referenced site")
    path = cfg.path(t.path or "dem.tif")
    downloadable = t.source in ("terrarium", "opentopography")

    def fetch():
        margin = 200.0
        lon, lat = site.frame.to_lonlat([b[0] - margin, b[2] + margin], [b[1] - margin, b[3] + margin])
        bbox = (float(lon.min()), float(lat.min()), float(lon.max()), float(lat.max()))
        if t.source == "terrarium":
            fetch_terrarium_dem(*bbox, out_path=path, zoom=t.zoom)
        else:
            fetch_opentopography_dem(*bbox, out_path=path)

    if not path.exists():
        if not downloadable:
            raise FileNotFoundError(f"GeoTIFF DEM not found: {path}")
        fetch()
    dem = resample_dem(geotiff_sampler(path, site.frame.local_to_crs), b, t.cell_size_m)
    coverage = float(np.isfinite(dem.elevations).mean())
    if coverage < 0.98 and downloadable:
        warnings.warn(f"Cached DEM {path.name} covers {100 * coverage:.0f}% of the site; re-downloading")
        path.unlink()
        fetch()
        dem = resample_dem(geotiff_sampler(path, site.frame.local_to_crs), b, t.cell_size_m)
        coverage = float(np.isfinite(dem.elevations).mean())
    if coverage < 0.5:
        raise ValueError(f"DEM {path.name} covers only {100 * coverage:.0f}% of the site area; "
                         "supply a GeoTIFF that covers the site.")
    bad = ~np.isfinite(dem.elevations)
    if bad.any():
        warnings.warn(f"DEM covers {100 * coverage:.1f}% of the site; gaps filled with the mean elevation")
        dem.elevations[bad] = np.nanmean(dem.elevations)
    return smooth_dem(dem, t.smooth_m)


# ── Losses at STC ─────────────────────────────────────────────────────────────

def _circuit_loss(routes, material, area_mm2, voltage, dc, pf, temp_c):
    rho = _resistivity_at_t(material, temp_c)
    loss_w, max_vd = 0.0, 0.0
    for r in routes:
        R = rho * r.length_m / (area_mm2 * 1e-6)
        if dc:
            I = r.power_kw * 1000 / voltage
            loss_w += 2 * I * I * R * r.n_parallel
            vd = 2 * I * R / voltage
        else:
            I = r.power_kw * 1000 / (math.sqrt(3) * voltage * pf)
            loss_w += 3 * I * I * R * r.n_parallel
            vd = math.sqrt(3) * I * R * pf / voltage
        max_vd = max(max_vd, 100 * vd)
    return loss_w / 1000.0, max_vd


def _recommend(routes, voltage, dc, pf, material, max_vdrop):
    if not routes:
        return None
    r = max(routes, key=lambda r: r.length_m * r.power_kw)
    try:
        return select_conductor_area(r.power_kw, voltage, r.length_m, material=material,
                                     max_vdrop_pct=max_vdrop, three_phase=not dc, power_factor=pf)
    except ValueError:
        return "exceeds largest standard section"


def _losses_stc(cfg: DesignConfig, elec: ElectricalLayout, n_str: int) -> dict:
    e, m = cfg.electrical, cfg.module
    v_dc = n_str * m.vmp_v
    pf, T = e.power_factor, e.cable_temp_c
    dc_kwp = elec.inverter_dc_kwp.sum()
    ac_kw = float(np.minimum(elec.inverter_dc_kwp, e.inverter_ac_kw).sum())

    s_kw, s_vd = _circuit_loss(elec.routes_of("string"), e.string_cable_material, e.string_cable_mm2,
                               v_dc, True, pf, T)
    d_kw, d_vd = _circuit_loss(elec.routes_of("dc_main"), e.dc_main_cable_material,
                               e.dc_main_cable_mm2, v_dc, True, pf, T)
    l_kw, l_vd = _circuit_loss(elec.routes_of("lv_ac"), e.lv_ac_cable_material, e.lv_ac_cable_mm2,
                               e.lv_ac_voltage_v, False, pf, T)
    v_kw, v_vd = _circuit_loss(elec.routes_of("mv"), e.mv_cable_material, e.mv_cable_mm2,
                               e.mv_voltage_v, False, pf, T)
    tx_kw = sum(transformer_loss(e.ps_kva, e.tx_no_load_kw, e.tx_load_loss_kw, p, pf)[0]
                for p in elec.ps_ac_kw())
    return {
        "dc_voltage_v": round(v_dc, 1),
        "dc_string_loss_kw": s_kw, "dc_main_loss_kw": d_kw,
        "dc_cable_loss_pct": 100 * (s_kw + d_kw) / dc_kwp,
        "lv_ac_loss_kw": l_kw, "mv_loss_kw": v_kw,
        "ac_cable_loss_pct": 100 * (l_kw + v_kw) / ac_kw,
        "transformer_loss_kw": tx_kw, "transformer_loss_pct": 100 * tx_kw / ac_kw,
        "max_vdrop_pct": {"string": s_vd, "dc_main": d_vd, "lv_ac": l_vd, "mv": v_vd},
        "recommended_mm2": {
            "string": _recommend(elec.routes_of("string"), v_dc, True, pf, e.string_cable_material, 1.5),
            "dc_main": _recommend(elec.routes_of("dc_main"), v_dc, True, pf, e.dc_main_cable_material, 1.5),
            "lv_ac": _recommend(elec.routes_of("lv_ac"), e.lv_ac_voltage_v, False, pf,
                                e.lv_ac_cable_material, 1.5),
            "mv": _recommend(elec.routes_of("mv"), e.mv_voltage_v, False, pf, e.mv_cable_material, 1.0),
        },
    }


# ── BoQ ───────────────────────────────────────────────────────────────────────

_DENSITY = {"copper": 8.9e-3, "aluminum": 2.7e-3}   # kg per (m · mm²)


def _boq(cfg, site, elec, terrain_final, structure) -> dict:
    e = cfg.electrical
    specs = {
        "string": (e.string_cable_material, e.string_cable_mm2, 2),
        "dc_main": (e.dc_main_cable_material, e.dc_main_cable_mm2, 2),
        "lv_ac": (e.lv_ac_cable_material, e.lv_ac_cable_mm2, 3),
        "mv": (e.mv_cable_material, e.mv_cable_mm2, 3),
    }
    cables = {}
    for kind, (mat, mm2, n_cond) in specs.items():
        L = elec.cable_length_m(kind)
        if L == 0:
            continue
        cables[kind] = {
            "route_length_m": round(L, 1), "material": mat, "area_mm2": mm2, "conductors": n_cond,
            "conductor_length_m": round(L * n_cond, 1),
            "mass_kg": round(L * n_cond * mm2 * _DENSITY[mat], 1),
        }
    lines = [LineString(r.points) for r in elec.routes if r.length_m > 0]
    trench_m = float(unary_union(lines).length) if lines else 0.0
    lay = elec.layout
    posts_per_table = (math.ceil(lay.width / cfg.structure.pile_spacing_m) + 1) * \
        (1 if structure.modules_high == 1 else 2)
    roads = site.roads(cfg.site.road_spacing_m, cfg.site.road_width_m)
    return {
        "modules": lay.n_modules,
        "tables": lay.n_tables,
        "strings": lay.n_tables * elec.strings_per_table,
        "piles": lay.n_tables * posts_per_table,
        "inverters": elec.n_inverters,
        "combiner_boxes": len(elec.combiner_xy),
        "power_stations": elec.n_power_stations,
        "cables": cables,
        "trench_length_m": round(trench_m, 1),
        "trench_volume_m3": round(trench_m * e.trench_depth_m * e.trench_width_m, 1),
        "fence_length_m": round(sum(p.exterior.length for _, p in site.available), 1),
        "internal_road_area_m2": round(roads.area, 1),
        "earthworks_cut_m3": round(terrain_final["cut_m3"], 1),
        "earthworks_fill_m3": round(terrain_final["fill_m3"], 1),
    }


# ── Energy ────────────────────────────────────────────────────────────────────

def _load_tmy(cfg: DesignConfig, site: Site) -> pd.DataFrame | None:
    en = cfg.energy
    if en.source == "none":
        return None
    if en.source == "clearsky":
        import pvlib
        times = pd.date_range("2023-01-01", periods=8760, freq="h", tz="UTC")
        cs = pvlib.location.Location(site.lat, site.lon).get_clearsky(times)
        return pd.DataFrame({"ghi": cs["ghi"], "dni": cs["dni"], "dhi": cs["dhi"],
                             "temp_air": 25.0, "wind_speed": 2.0}, index=times)
    path = cfg.path(en.tmy_file)
    meta = path.with_suffix(".json")
    if path.exists() and en.source == "pvgis" and meta.exists():
        loc = json.loads(meta.read_text(encoding="utf-8"))
        if abs(loc["lat"] - site.lat) > 0.01 or abs(loc["lon"] - site.lon) > 0.01:
            warnings.warn(f"Cached {path.name} is for ({loc['lat']:.3f}, {loc['lon']:.3f}); "
                          "site moved, re-downloading PVGIS TMY")
            path.unlink()
    if path.exists():
        df = pd.read_csv(path, index_col=0)
        df.index = pd.to_datetime(df.index, utc=True)
        missing = {"ghi", "dni", "dhi", "temp_air"} - set(df.columns)
        if missing:
            raise ValueError(f"{path.name} lacks column(s) {sorted(missing)}")
        if "wind_speed" not in df.columns:
            df["wind_speed"] = 1.0
        if len(df) < 8760:
            warnings.warn(f"{path.name} has {len(df)} rows (< 8760); annual totals will be low")
        return df
    if en.source == "file":
        raise FileNotFoundError(f"TMY file not found: {path}")
    import pvlib
    try:
        data = pvlib.iotools.get_pvgis_tmy(site.lat, site.lon, map_variables=True, timeout=60)[0]
    except Exception as exc:
        raise RuntimeError(f"PVGIS TMY download failed ({exc}); provide {path.name} or set "
                           "[energy] source='clearsky'") from exc
    data = data[["ghi", "dni", "dhi", "temp_air", "wind_speed"]]
    path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(path)
    meta.write_text(json.dumps({"lat": site.lat, "lon": site.lon, "source": "PVGIS-5.2 TMY"}),
                    encoding="utf-8")
    return data


def _load_horizon(cfg: DesignConfig, site: Site) -> pd.Series | None:
    """Far-horizon profile (azimuth° → elevation°) from PVGIS, cached next to the config."""
    en = cfg.energy
    if not en.horizon or en.source in ("none", "clearsky"):
        return None
    path = cfg.path(en.horizon_file)
    meta = path.with_suffix(".json")
    if path.exists() and meta.exists():
        loc = json.loads(meta.read_text(encoding="utf-8"))
        if abs(loc["lat"] - site.lat) > 0.01 or abs(loc["lon"] - site.lon) > 0.01:
            path.unlink()
    if path.exists():
        s = pd.read_csv(path, index_col=0).iloc[:, 0]
        s.index = s.index.astype(float)
        return s
    import pvlib
    try:
        data = pvlib.iotools.get_pvgis_horizon(site.lat, site.lon, timeout=60)
    except Exception as exc:
        warnings.warn(f"PVGIS horizon download failed ({exc}); horizon shading skipped")
        return None
    s = data[0] if isinstance(data, tuple) else data
    s = s.astype(float)
    s.index = s.index.astype(float) % 360.0
    s = s.groupby(level=0).max().sort_index()
    s.rename("horizon_elevation_deg").to_frame().to_csv(path, index_label="azimuth_deg")
    meta.write_text(json.dumps({"lat": site.lat, "lon": site.lon, "source": "PVGIS-5.2 horizon"}),
                    encoding="utf-8")
    return s


def _energy(cfg, site, elec, losses, tilt, module) -> EnergyResult | None:
    tmy = _load_tmy(cfg, site)
    if tmy is None:
        return None
    if cfg.energy.pv_model not in ("huld", "linear"):
        raise ValueError("[energy] pv_model must be 'huld' or 'linear'")
    horizon = _load_horizon(cfg, site)
    e, m, en = cfg.electrical, cfg.module, cfg.energy
    lay = elec.layout
    kwp = lay.kwp(module.power_wp)
    n_ps = elec.n_power_stations
    return annual_energy_yield(
        lat=site.lat, lon=site.lon, total_kwp=kwp, tilt_deg=tilt, azimuth_deg=None,
        dc_cable_loss_pct=losses["dc_cable_loss_pct"], ac_cable_loss_pct=losses["ac_cable_loss_pct"],
        soiling_pct=en.soiling_pct, mismatch_pct=en.mismatch_pct, lid_pct=en.lid_pct,
        availability_pct=en.availability_pct, inverter_efficiency_pct=e.inverter_efficiency_pct,
        gamma_pmax_per_c=m.gamma_pmax_pct / 100, noct_c=m.noct_c, tmy_data=tmy,
        gcr=lay.gcr if en.row_shading else None, n_rows=len(np.unique(lay.row_id)),
        iam_ar=en.iam_ar, pv_model=en.pv_model, horizon=horizon,
        ac_capacity_kw=elec.n_inverters * e.inverter_ac_kw, ohmic_at_stc=True,
        transformer_losses=(e.tx_no_load_kw * n_ps, e.tx_load_loss_kw * n_ps, e.ps_kva * n_ps),
        power_factor=e.power_factor,
    )


# ── KPIs ──────────────────────────────────────────────────────────────────────

def _kpis(site, usable, tables_all, terrain_all, elec, status, n_str, tilt, energy, module) -> dict:
    lay = elec.layout
    kwp = lay.kwp(module.power_wp)
    ac_kw = elec.n_inverters * elec.spec.inverter_ac_kw
    avail_ha = site.available_union.area / 1e4
    k = {
        "lat": round(site.lat, 5), "lon": round(site.lon, 5),
        "available_ha": round(avail_ha, 2),
        "usable_ha": round(usable.area / 1e4, 2),
        "dc_kwp": round(kwp, 1),
        "ac_kw": round(ac_kw, 1),
        "dc_ac_ratio": round(kwp / ac_kw, 3),
        "mwp_per_ha": round(kwp / 1000 / avail_ha, 3),
        "tilt_deg": tilt,
        "azimuth_deg": 180.0 if site.lat >= 0 else 0.0,
        "pitch_m": round(lay.pitch, 3),
        "gcr": round(lay.gcr, 4),
        "table_w_m": round(lay.width, 2), "table_depth_m": round(lay.depth, 2),
        "modules_per_string": n_str,
        "strings_per_table": elec.strings_per_table,
        "n_tables": lay.n_tables,
        "n_modules": lay.n_modules,
        "n_inverters": elec.n_inverters,
        "n_power_stations": elec.n_power_stations,
        "tables_candidate": tables_all.n_tables,
        "tables_excluded_terrain": terrain_all.count(EXCLUDED),
        "tables_earthworks": int((status == EARTHWORKS).sum()),
        "tables_removed_for_ps": elec.removed_for_ps,
        "inverter_block_kwp_min": round(float(elec.inverter_dc_kwp.min()), 1),
        "inverter_block_kwp_max": round(float(elec.inverter_dc_kwp.max()), 1),
    }
    if energy is not None:
        k.update({
            "annual_mwh": round(energy.annual_kwh / 1000, 1),
            "specific_yield_kwh_kwp": round(energy.specific_yield_kwh_kwp, 1),
            "pr_poa": round(energy.pr_poa, 4),
            "poa_kwh_m2": round(energy.poa_annual_kwh_m2, 1),
            "ghi_kwh_m2": round(energy.ghi_annual_kwh_m2, 1),
        })
    return k


def save_summary(result: DesignResult, out_dir: pathlib.Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(
        json.dumps(result.summary(), indent=2, default=_jsonable), encoding="utf-8")
    rows = [("item", "quantity", "unit")]
    b = result.boq
    for key, unit in [("modules", "pcs"), ("tables", "pcs"), ("strings", "pcs"), ("piles", "pcs"),
                      ("inverters", "pcs"), ("combiner_boxes", "pcs"), ("power_stations", "pcs"),
                      ("trench_length_m", "m"), ("trench_volume_m3", "m3"), ("fence_length_m", "m"),
                      ("internal_road_area_m2", "m2"), ("earthworks_cut_m3", "m3"),
                      ("earthworks_fill_m3", "m3")]:
        rows.append((key, b[key], unit))
    for kind, c in b["cables"].items():
        rows.append((f"cable_{kind}_{c['material']}_{c['area_mm2']}mm2", c["conductor_length_m"], "m"))
        rows.append((f"cable_{kind}_mass", c["mass_kg"], "kg"))
    (out_dir / "boq.csv").write_text("\n".join(",".join(map(str, r)) for r in rows) + "\n",
                                     encoding="utf-8")
