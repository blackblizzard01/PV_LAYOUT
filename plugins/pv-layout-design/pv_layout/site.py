"""
Site model: available areas, restricted areas, setbacks, internal roads.

usable = ⋃_i ( AA_i.buffer(−perimeter_setback) ) − ⋃_j RA_j.buffer(restricted_setback) − roads

Source: RatedPower Design_Methodology §5.2.1 (available / restricted areas)
and §5.4.6 (setbacks: perimeter → fence → structures).
"""

from __future__ import annotations

import pathlib
import warnings
from dataclasses import dataclass, field

import shapely
from shapely.geometry import MultiPolygon, Polygon, box
from shapely.ops import unary_union

from .geo import GeoFrame, GeoSite, as_multipolygon, read_site


@dataclass
class Site:
    name: str
    lat: float
    lon: float
    available: list[tuple[str, Polygon]]
    restricted: list[tuple[str, Polygon]] = field(default_factory=list)
    frame: GeoFrame | None = None

    @classmethod
    def from_file(cls, path: str | pathlib.Path, name: str | None = None) -> "Site":
        g: GeoSite = read_site(path, name=name)
        return cls(name=g.name, lat=g.lat, lon=g.lon,
                   available=g.available, restricted=g.restricted, frame=g.frame)

    @classmethod
    def from_local(cls, name: str, lat: float, lon: float, available, restricted=(),
                   with_frame: bool = True) -> "Site":
        """Synthetic site in local metres; local (0,0) is anchored at (lon, lat)."""
        av = [(f"AA-{i + 1}", p) for i, p in enumerate(_as_list(available))]
        ra = [(f"RA-{i + 1}", p) for i, p in enumerate(_as_list(restricted))]
        frame = None
        if with_frame:
            try:
                frame = GeoFrame.from_anchor(lon, lat)
            except ImportError:
                frame = None
        return cls(name=name, lat=lat, lon=lon, available=av, restricted=ra, frame=frame)

    @property
    def available_union(self) -> MultiPolygon:
        return as_multipolygon(unary_union([p for _, p in self.available]))

    @property
    def restricted_union(self) -> MultiPolygon:
        return as_multipolygon(unary_union([p for _, p in self.restricted]))

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        return self.available_union.bounds

    def roads(self, spacing_m: float, width_m: float) -> MultiPolygon:
        """North-south internal road corridors every ``spacing_m`` across the site."""
        if spacing_m <= 0 or width_m <= 0:
            return MultiPolygon()
        x0, y0, x1, y1 = self.bounds
        xs = []
        x = x0 + spacing_m
        while x < x1 - width_m:
            xs.append(x)
            x += spacing_m
        strips = [box(x - width_m / 2, y0 - 1, x + width_m / 2, y1 + 1) for x in xs]
        return as_multipolygon(unary_union(strips).intersection(self.available_union))

    def usable_area(
        self,
        perimeter_setback_m: float = 0.0,
        restricted_setback_m: float = 0.0,
        road_spacing_m: float = 0.0,
        road_width_m: float = 0.0,
    ) -> MultiPolygon:
        parts = []
        for name, poly in self.available:
            inner = poly.buffer(-perimeter_setback_m, join_style="mitre") \
                if perimeter_setback_m > 0 else poly
            if inner.is_empty:
                warnings.warn(f"{name}: perimeter setback {perimeter_setback_m} m "
                              "erases this area", stacklevel=2)
                continue
            parts.append(inner)
        if not parts:
            raise ValueError(
                f"Perimeter setback of {perimeter_setback_m} m erases every available area "
                f"({', '.join(n for n, _ in self.available)})."
            )
        usable = unary_union(parts)
        if self.restricted:
            ra = unary_union([
                p.buffer(restricted_setback_m) if restricted_setback_m > 0 else p
                for _, p in self.restricted
            ])
            usable = usable.difference(ra)
        roads = self.roads(road_spacing_m, road_width_m)
        if not roads.is_empty:
            usable = usable.difference(roads)
        usable = shapely.make_valid(usable)
        result = as_multipolygon(usable)
        result = MultiPolygon([p for p in result.geoms if p.area >= 1.0])
        if result.is_empty:
            raise ValueError("No usable area left after setbacks, restricted areas and roads.")
        return result


def _as_list(x):
    if x is None:
        return []
    if isinstance(x, (Polygon, MultiPolygon)):
        return list(x.geoms) if isinstance(x, MultiPolygon) else [x]
    return list(x)
