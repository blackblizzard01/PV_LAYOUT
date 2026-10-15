"""
Site boundary I/O and coordinate frames.

Reads KML / GeoJSON (WGS84 lon/lat) and converts to a local metric frame:
UTM zone picked from the site centroid, shifted so the SW corner of the
site bounding box is (0, 0).  x = east, y = north, metres.

Feature roles (RatedPower naming convention):
  - property ``role`` = "available" | "restricted", or
  - name prefix ``AA``  → available area, ``RA`` → restricted area.
  Anything else defaults to available.
LineString / Point features need a ``buffer_m`` property (e.g. an HV line
corridor) and are buffered into polygons.
"""

from __future__ import annotations

import json
import math
import pathlib
import warnings
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

import numpy as np
import shapely
from shapely.geometry import (
    LineString, MultiPolygon, Point, Polygon, mapping, shape,
)
from shapely.ops import unary_union

_MIN_AREA_M2 = 1.0


def _require_pyproj():
    try:
        import pyproj
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            'pyproj is required for real-world sites. Install with: '
            'pip install "pv-layout-design[geo]"'
        ) from exc
    return pyproj


def utm_epsg(lon: float, lat: float) -> int:
    """EPSG code of the WGS84 UTM zone containing (lon, lat)."""
    zone = int((lon + 180.0) // 6.0) % 60 + 1
    return (32600 if lat >= 0 else 32700) + zone


@dataclass
class GeoFrame:
    """Local metric frame: UTM shifted by ``origin`` (UTM easting/northing of local 0,0)."""
    epsg: int
    origin: tuple[float, float]

    def __post_init__(self):
        pyproj = _require_pyproj()
        self._fwd = pyproj.Transformer.from_crs(4326, self.epsg, always_xy=True)
        self._inv = pyproj.Transformer.from_crs(self.epsg, 4326, always_xy=True)

    @classmethod
    def from_anchor(cls, lon: float, lat: float) -> "GeoFrame":
        """Frame whose local (0, 0) sits at (lon, lat)."""
        epsg = utm_epsg(lon, lat)
        pyproj = _require_pyproj()
        e, n = pyproj.Transformer.from_crs(4326, epsg, always_xy=True).transform(lon, lat)
        return cls(epsg=epsg, origin=(float(e), float(n)))

    def to_local(self, lons, lats):
        e, n = self._fwd.transform(np.asarray(lons, float), np.asarray(lats, float))
        return np.asarray(e) - self.origin[0], np.asarray(n) - self.origin[1]

    def to_lonlat(self, xs, ys):
        lon, lat = self._inv.transform(
            np.asarray(xs, float) + self.origin[0], np.asarray(ys, float) + self.origin[1]
        )
        return np.asarray(lon), np.asarray(lat)

    def geom_to_local(self, geom):
        return shapely.transform(geom, lambda c: np.column_stack(self.to_local(c[:, 0], c[:, 1])))

    def geom_to_lonlat(self, geom):
        return shapely.transform(geom, lambda c: np.column_stack(self.to_lonlat(c[:, 0], c[:, 1])))

    def local_to_crs(self, epsg_or_wkt):
        """Return f(xs, ys) -> (X, Y) mapping local metres into another CRS."""
        pyproj = _require_pyproj()
        t = pyproj.Transformer.from_crs(self.epsg, epsg_or_wkt, always_xy=True)
        ox, oy = self.origin

        def f(xs, ys):
            X, Y = t.transform(np.asarray(xs, float) + ox, np.asarray(ys, float) + oy)
            return np.asarray(X), np.asarray(Y)
        return f


@dataclass
class SiteFeature:
    name: str
    role: str              # "available" | "restricted"
    geom_lonlat: object    # shapely geometry in lon/lat
    buffer_m: float = 0.0


@dataclass
class GeoSite:
    """Site boundaries projected into a local metric frame."""
    name: str
    frame: GeoFrame
    available: list[tuple[str, Polygon]] = field(default_factory=list)
    restricted: list[tuple[str, Polygon]] = field(default_factory=list)
    lon: float = 0.0
    lat: float = 0.0


# ── Parsing ───────────────────────────────────────────────────────────────────

def _role_for(name: str, props: dict) -> str:
    role = str(props.get("role", "")).strip().lower()
    if role in ("available", "restricted"):
        return role
    up = name.strip().upper()
    if up.startswith("RA"):
        return "restricted"
    return "available"


def _parse_geojson(path: pathlib.Path) -> list[SiteFeature]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("type") == "FeatureCollection":
        feats = data["features"]
    elif data.get("type") == "Feature":
        feats = [data]
    else:
        feats = [{"type": "Feature", "properties": {}, "geometry": data}]
    out = []
    for i, f in enumerate(feats):
        if not f.get("geometry"):
            continue
        props = f.get("properties") or {}
        name = str(props.get("name", f"feature-{i}"))
        out.append(SiteFeature(
            name=name,
            role=_role_for(name, props),
            geom_lonlat=shape(f["geometry"]),
            buffer_m=float(props.get("buffer_m", 0.0) or 0.0),
        ))
    return out


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _kml_coords(text: str) -> list[tuple[float, float]]:
    pts = []
    for tok in (text or "").split():
        parts = tok.split(",")
        if len(parts) >= 2:
            pts.append((float(parts[0]), float(parts[1])))
    return pts


def _kml_find(el, name):
    return [e for e in el.iter() if _local(e.tag) == name]


def _kml_ring(el) -> list[tuple[float, float]]:
    coords = _kml_find(el, "coordinates")
    return _kml_coords(coords[0].text) if coords else []


def _parse_kml(path: pathlib.Path) -> list[SiteFeature]:
    root = ET.parse(path).getroot()
    out = []
    for i, pm in enumerate(_kml_find(root, "Placemark")):
        name_el = [c for c in pm if _local(c.tag) == "name"]
        name = name_el[0].text.strip() if name_el and name_el[0].text else f"placemark-{i}"
        props = {}
        for d in _kml_find(pm, "Data"):
            v = [c for c in d if _local(c.tag) == "value"]
            props[d.get("name", "")] = v[0].text if v else None
        for sd in _kml_find(pm, "SimpleData"):
            props[sd.get("name", "")] = sd.text

        geoms = []
        for poly in _kml_find(pm, "Polygon"):
            outer = [e for e in poly.iter() if _local(e.tag) == "outerBoundaryIs"]
            inner = [e for e in poly.iter() if _local(e.tag) == "innerBoundaryIs"]
            if not outer:
                continue
            geoms.append(Polygon(_kml_ring(outer[0]), [_kml_ring(r) for r in inner]))
        for ls in _kml_find(pm, "LineString"):
            geoms.append(LineString(_kml_ring(ls)))
        for pt in _kml_find(pm, "Point"):
            c = _kml_ring(pt)
            if c:
                geoms.append(Point(c[0]))
        if not geoms:
            continue
        geom = geoms[0] if len(geoms) == 1 else shapely.GeometryCollection(geoms)
        out.append(SiteFeature(
            name=name,
            role=_role_for(name, props),
            geom_lonlat=geom,
            buffer_m=float(props.get("buffer_m") or 0.0),
        ))
    return out


def _polygons_of(geom) -> list[Polygon]:
    if geom.is_empty:
        return []
    if isinstance(geom, Polygon):
        return [geom]
    if hasattr(geom, "geoms"):
        return [p for g in geom.geoms for p in _polygons_of(g)]
    return []


def _clean(geom, label: str) -> list[Polygon]:
    if not geom.is_valid:
        warnings.warn(f"{label}: invalid geometry repaired with make_valid", stacklevel=3)
        geom = shapely.make_valid(geom)
    polys = [p for p in _polygons_of(geom) if p.area >= _MIN_AREA_M2]
    return polys


def read_site(path: str | pathlib.Path, name: str | None = None) -> GeoSite:
    """Load a KML or GeoJSON site file into a local metric frame."""
    path = pathlib.Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Site file not found: {path}")
    suffix = path.suffix.lower()
    if suffix == ".kml":
        feats = _parse_kml(path)
    elif suffix in (".geojson", ".json"):
        feats = _parse_geojson(path)
    else:
        raise ValueError(f"Unsupported site file type '{suffix}' (use .kml or .geojson)")

    if not any(f.role == "available" for f in feats):
        raise ValueError(f"{path.name}: no available-area polygon found "
                         "(name it 'AA-...' or set role='available')")

    avail_ll = unary_union([f.geom_lonlat for f in feats if f.role == "available"])
    c = avail_ll.centroid
    lon0, lat0 = c.x, c.y
    epsg = utm_epsg(lon0, lat0)
    pyproj = _require_pyproj()
    fwd = pyproj.Transformer.from_crs(4326, epsg, always_xy=True)
    minx, miny, _, _ = shapely.transform(
        avail_ll, lambda c: np.column_stack(fwd.transform(c[:, 0], c[:, 1]))).bounds
    frame = GeoFrame(epsg=epsg, origin=(minx, miny))

    site = GeoSite(name=name or path.stem, frame=frame, lon=lon0, lat=lat0)
    for f in feats:
        g = frame.geom_to_local(f.geom_lonlat)
        if f.buffer_m > 0:
            g = g.buffer(f.buffer_m)
        polys = _clean(g, f.name)
        if not polys:
            warnings.warn(f"{f.name}: no usable polygon area, skipped", stacklevel=2)
            continue
        target = site.available if f.role == "available" else site.restricted
        for k, p in enumerate(polys):
            target.append((f.name if len(polys) == 1 else f"{f.name}.{k + 1}", p))
    return site


def to_geojson_feature(geom_lonlat, **props) -> dict:
    return {"type": "Feature", "properties": props, "geometry": mapping(geom_lonlat)}


def as_multipolygon(geom) -> MultiPolygon:
    polys = _polygons_of(geom)
    return MultiPolygon(polys) if polys else MultiPolygon()


def haversine_m(lon1, lat1, lon2, lat2) -> float:
    r = 6371008.8
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))
