import json

import numpy as np
import pytest
from shapely.geometry import box

from pv_layout.geo import GeoFrame, read_site, utm_epsg
from pv_layout.site import Site


def test_utm_epsg_hemispheres():
    assert utm_epsg(72.6, 23.0) == 32643
    assert utm_epsg(146.07, -34.62) == 32755
    assert utm_epsg(0.08, 52.3) == 32631
    assert utm_epsg(-179.9, 10) == 32601


def test_frame_round_trip():
    f = GeoFrame.from_anchor(140.2, 37.4)
    x, y = f.to_local([140.2, 140.21], [37.4, 37.41])
    assert abs(x[0]) < 1e-6 and abs(y[0]) < 1e-6
    assert 850 < x[1] < 900 and 1100 < y[1] < 1120          # ~0.01° at 37.4°N
    lon, lat = f.to_lonlat(x, y)
    np.testing.assert_allclose(lon, [140.2, 140.21], atol=1e-9)
    np.testing.assert_allclose(lat, [37.4, 37.41], atol=1e-9)


def _write(tmp_path, features):
    p = tmp_path / "site.geojson"
    p.write_text(json.dumps({"type": "FeatureCollection", "features": features}))
    return p


def _poly(coords, **props):
    return {"type": "Feature", "properties": props, "geometry": {"type": "Polygon", "coordinates": [coords]}}


def test_geojson_roles_and_line_buffer(tmp_path):
    sq = [[10.0, 50.0], [10.01, 50.0], [10.01, 50.01], [10.0, 50.01], [10.0, 50.0]]
    line = {"type": "Feature", "properties": {"name": "RA-road", "buffer_m": 5},
            "geometry": {"type": "LineString", "coordinates": [[10.005, 49.999], [10.005, 50.011]]}}
    p = _write(tmp_path, [_poly(sq, name="AA-1"), line])
    s = read_site(p)
    assert [n for n, _ in s.available] == ["AA-1"]
    assert [n for n, _ in s.restricted] == ["RA-road"]
    road = s.restricted[0][1]
    assert road.area == pytest.approx(10 * 1335, rel=0.05)    # 10 m wide × ~1.33 km
    assert s.available[0][1].bounds[0] == pytest.approx(0, abs=1e-6)


def test_geojson_without_available_area_raises(tmp_path):
    sq = [[10.0, 50.0], [10.01, 50.0], [10.01, 50.01], [10.0, 50.0]]
    p = _write(tmp_path, [_poly(sq, name="RA-only")])
    with pytest.raises(ValueError, match="no available-area"):
        read_site(p)


def test_self_intersecting_polygon_is_repaired(tmp_path):
    bowtie = [[10.0, 50.0], [10.01, 50.01], [10.01, 50.0], [10.0, 50.01], [10.0, 50.0]]
    p = _write(tmp_path, [_poly(bowtie, name="AA-bowtie")])
    with pytest.warns(UserWarning, match="make_valid"):
        s = read_site(p)
    assert len(s.available) == 2 and all(poly.is_valid for _, poly in s.available)


def test_kml_polygon_hole_point_and_line(tmp_path):
    kml = """<?xml version="1.0"?><kml xmlns="http://www.opengis.net/kml/2.2"><Document>
    <Placemark><name>AA-Field</name><Polygon>
      <outerBoundaryIs><LinearRing><coordinates>
        0,0,0 0.01,0,0 0.01,0.01,0 0,0.01,0 0,0,0</coordinates></LinearRing></outerBoundaryIs>
      <innerBoundaryIs><LinearRing><coordinates>
        0.004,0.004 0.006,0.004 0.006,0.006 0.004,0.006 0.004,0.004</coordinates></LinearRing></innerBoundaryIs>
    </Polygon></Placemark>
    <Placemark><name>RA-Mast</name><ExtendedData><Data name="buffer_m"><value>20</value></Data></ExtendedData>
      <Point><coordinates>0.002,0.002,0</coordinates></Point></Placemark>
    </Document></kml>"""
    p = tmp_path / "s.kml"
    p.write_text(kml)
    s = read_site(p)
    field = s.available[0][1]
    assert len(field.interiors) == 1
    assert s.restricted[0][1].area == pytest.approx(np.pi * 400, rel=0.02)


def test_unsupported_extension(tmp_path):
    p = tmp_path / "site.shp"
    p.write_text("x")
    with pytest.raises(ValueError, match="Unsupported"):
        read_site(p)


def test_usable_area_setbacks_and_errors():
    site = Site.from_local("t", 23.0, 72.6, [box(0, 0, 200, 100)], [box(80, 30, 120, 70)], with_frame=False)
    u = site.usable_area(perimeter_setback_m=10, restricted_setback_m=5)
    ra_buffered = 40 * 40 + 4 * 40 * 5 + np.pi * 5 ** 2      # rounded-corner buffer
    assert u.area == pytest.approx(180 * 80 - ra_buffered, rel=1e-3)
    with pytest.raises(ValueError, match="erases every available area"), \
            pytest.warns(UserWarning, match="erases this area"):
        site.usable_area(perimeter_setback_m=60)


def test_roads_split_usable_area():
    site = Site.from_local("t", 23.0, 72.6, [box(0, 0, 500, 100)], with_frame=False)
    u = site.usable_area(road_spacing_m=200, road_width_m=6)
    assert len(u.geoms) == 3
    assert u.area == pytest.approx(500 * 100 - 2 * 6 * 100)
