"""
Interactive 2D site map (Leaflet via folium): satellite / street basemaps,
toggleable layers matching RatedPower's Layers panel, per-structure popups,
measuring tool, fullscreen and live lat/lon readout.
"""

from __future__ import annotations

import html

import numpy as np
from shapely.geometry import mapping

from ..terrain import EARTHWORKS, EXCLUDED, STATUS_NAMES
from .png import COLORS

_MERGE_ABOVE = 20000


def _require_folium():
    try:
        import folium
        from folium import plugins
    except ImportError as exc:  # pragma: no cover
        raise ImportError('folium is required for map.html. Install with: '
                          'pip install "pv-layout-design[viz]"') from exc
    return folium, plugins


def _boxes_lonlat(frame, centers, w, d):
    x, y = centers[:, 0], centers[:, 1]
    xs = np.stack([x - w / 2, x + w / 2, x + w / 2, x - w / 2, x - w / 2], axis=1)
    ys = np.stack([y - d / 2, y - d / 2, y + d / 2, y + d / 2, y - d / 2], axis=1)
    lon, lat = frame.to_lonlat(xs.ravel(), ys.ravel())
    return np.stack([lon.reshape(xs.shape), lat.reshape(xs.shape)], axis=2)


def _table_fc(rings, props_list):
    return {"type": "FeatureCollection", "features": [
        {"type": "Feature", "properties": props,
         "geometry": {"type": "Polygon", "coordinates": [np.round(ring, 7).tolist()]}}
        for ring, props in zip(rings, props_list)
    ]}


def _merged_fc(frame, centers, w, d):
    from shapely.ops import unary_union
    import shapely
    boxes = shapely.box(centers[:, 0] - w / 2, centers[:, 1] - d / 2,
                        centers[:, 0] + w / 2, centers[:, 1] + d / 2)
    merged = frame.geom_to_lonlat(unary_union(boxes.tolist()))
    return {"type": "FeatureCollection", "features": [
        {"type": "Feature", "properties": {"merged": True}, "geometry": mapping(merged)}]}


def _line_ll(frame, pts):
    lon, lat = frame.to_lonlat(pts[:, 0], pts[:, 1])
    return list(zip(lat.tolist(), lon.tolist()))


def render_map_html(result, path) -> str:
    folium, plugins = _require_folium()
    site, frame, elec, lay = result.site, result.site.frame, result.elec, result.tables
    if frame is None:
        raise ValueError("map.html needs a geo-referenced site")

    m = folium.Map(location=[site.lat, site.lon], zoom_start=16, control_scale=True,
                   tiles=None, max_zoom=21)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri, Maxar, Earthstar Geographics", name="Satellite", max_zoom=21, max_native_zoom=19,
    ).add_to(m)
    folium.TileLayer("OpenStreetMap", name="Streets", max_zoom=21, max_native_zoom=19,
                     show=False).add_to(m)
    folium.TileLayer(
        tiles="https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png",
        attr="OpenTopoMap (CC-BY-SA)", name="Topographic", max_zoom=21, max_native_zoom=17,
        show=False,
    ).add_to(m)

    def group(name, show=True):
        g = folium.FeatureGroup(name=name, show=show)
        g.add_to(m)
        return g

    # areas
    g = group("Available areas")
    for name, p in site.available:
        folium.GeoJson(mapping(frame.geom_to_lonlat(p)), tooltip=name, style_function=lambda _: {
            "color": "#ffffff", "weight": 2.5, "fillColor": "#ffffff", "fillOpacity": 0.08}).add_to(g)
    if site.restricted:
        g = group("Restricted areas")
        for name, p in site.restricted:
            folium.GeoJson(mapping(frame.geom_to_lonlat(p)), tooltip=name, style_function=lambda _: {
                "color": COLORS["restricted_edge"], "weight": 2, "fillColor": "#e74c3c",
                "fillOpacity": 0.35, "dashArray": "6 4"}).add_to(g)
    g = group("Setback line", show=False)
    folium.GeoJson(mapping(frame.geom_to_lonlat(result.usable)), style_function=lambda _: {
        "color": "#f1c40f", "weight": 1.2, "fill": False, "dashArray": "4 4"}).add_to(g)
    roads = site.roads(result.config.site.road_spacing_m, result.config.site.road_width_m)
    if not roads.is_empty:
        g = group("Roads")
        folium.GeoJson(mapping(frame.geom_to_lonlat(roads)), style_function=lambda _: {
            "color": "#7f8c8d", "weight": 0.5, "fillColor": "#bdc3c7", "fillOpacity": 0.8}).add_to(g)

    # inverter blocks (hulls)
    from shapely.geometry import MultiPoint
    g = group("Inverter blocks", show=False)
    for b in range(elec.n_inverters):
        sel = lay.centers[elec.table_block == b]
        if len(sel) == 0:
            continue
        pts = np.concatenate([sel + [dx, dy] for dx in (-lay.width / 2, lay.width / 2)
                              for dy in (-lay.depth / 2, lay.depth / 2)])
        hull = frame.geom_to_lonlat(MultiPoint(pts).convex_hull.buffer(1.0))
        folium.GeoJson(mapping(hull), tooltip=f"Block {b + 1}: {elec.inverter_dc_kwp[b]:.0f} kWp",
                       style_function=lambda _: {"color": "#00e5ff", "weight": 1.5,
                                                 "fillOpacity": 0.05}).add_to(g)

    # structures
    idx = np.arange(lay.n_tables)
    tkwp = lay.modules_per_table * result.module.power_wp / 1000
    sel_all = _match(result.tables_all.centers, lay.centers)
    ns = result.terrain_all.ns_slope_pct[sel_all]
    ew = result.terrain_all.ew_slope_pct[sel_all]
    for status, label, color in [(0, "Structures – accepted", "#3d8bfd"),
                                 (EARTHWORKS, "Structures – earthworks", COLORS["earthworks"])]:
        mask = result.status == status
        if not mask.any():
            continue
        g = group(f"{label} ({int(mask.sum())})")
        if lay.n_tables > _MERGE_ABOVE:
            fc = _merged_fc(frame, lay.centers[mask], lay.width, lay.depth)
        else:
            rings = _boxes_lonlat(frame, lay.centers[mask], lay.width, lay.depth)
            props = [{"id": int(i + 1), "row": int(lay.row_id[i]), "block": int(elec.table_block[i] + 1),
                      "kWp": round(tkwp, 2), "NS %": round(float(ns[i]), 1),
                      "EW %": round(float(ew[i]), 1), "status": STATUS_NAMES[status]}
                     for i in idx[mask]]
            fc = _table_fc(rings, props)
        folium.GeoJson(
            fc, style_function=lambda _, c=color: {"color": c, "weight": 0.5, "fillColor": c,
                                                    "fillOpacity": 0.85},
            tooltip=None if lay.n_tables > _MERGE_ABOVE else folium.GeoJsonTooltip(
                fields=["id", "block", "row", "kWp", "NS %", "EW %", "status"]),
        ).add_to(g)
    ex = result.terrain_all.status == EXCLUDED
    if ex.any():
        g = group(f"Structures – excluded by slope ({int(ex.sum())})")
        rings = _boxes_lonlat(frame, result.tables_all.centers[ex], lay.width, lay.depth)
        props = [{"NS %": round(float(a), 1), "EW %": round(float(b), 1), "status": "excluded"}
                 for a, b in zip(result.terrain_all.ns_slope_pct[ex], result.terrain_all.ew_slope_pct[ex])]
        folium.GeoJson(_table_fc(rings, props), style_function=lambda _: {
            "color": COLORS["excluded"], "weight": 0.5, "fillColor": COLORS["excluded"], "fillOpacity": 0.6},
            tooltip=folium.GeoJsonTooltip(fields=["NS %", "EW %", "status"])).add_to(g)

    # cables
    g = group("MV cables")
    for r in elec.routes_of("mv"):
        folium.PolyLine(_line_ll(frame, r.points), color="#ff3b30", weight=3,
                        tooltip=f"MV {r.length_m:.0f} m, {r.power_kw:.0f} kW").add_to(g)
    lv = elec.routes_of("dc_main") + elec.routes_of("lv_ac")
    if lv:
        g = group("DC main / LV AC cables")
        for r in lv:
            folium.PolyLine(_line_ll(frame, r.points), color="#ff9500", weight=1.5,
                            tooltip=f"{r.kind} {r.length_m:.0f} m, {r.power_kw:.0f} kW").add_to(g)

    # equipment
    if len(elec.combiner_xy):
        g = group("Combiner boxes", show=False)
        lon, lat = frame.to_lonlat(elec.combiner_xy[:, 0], elec.combiner_xy[:, 1])
        for a, b in zip(lat, lon):
            folium.CircleMarker([a, b], radius=3, color="#2c3e50", fill=True, fill_opacity=1).add_to(g)
    if elec.spec.inverter_type == "string":
        g = group("String inverters")
        lon, lat = frame.to_lonlat(elec.inverter_xy[:, 0], elec.inverter_xy[:, 1])
        for i, (a, b) in enumerate(zip(lat, lon)):
            folium.RegularPolygonMarker([a, b], number_of_sides=3, radius=5, color="#000",
                                        weight=0.5, fill_color="#ff9500", fill_opacity=1,
                                        tooltip=f"Inverter {i + 1}: {elec.inverter_dc_kwp[i]:.0f} kWp").add_to(g)
    g = group("Power stations")
    ac = elec.ps_ac_kw()
    for i, poly in enumerate(elec.ps_polys):
        folium.GeoJson(mapping(frame.geom_to_lonlat(poly.buffer(1.0))),
                       tooltip=f"PS{i + 1}: {ac[i]:.0f} kW AC / {elec.spec.ps_kva:.0f} kVA",
                       style_function=lambda _: {"color": "#000", "weight": 1, "fillColor": "#8e1b1b",
                                                 "fillOpacity": 1}).add_to(g)
    g = group("Substation")
    lon, lat = frame.to_lonlat([elec.substation_xy[0]], [elec.substation_xy[1]])
    folium.Marker([lat[0], lon[0]], tooltip="Substation / grid point",
                  icon=folium.Icon(color="black", icon="flash")).add_to(g)

    # controls
    b = frame.geom_to_lonlat(site.available_union).bounds
    m.fit_bounds([[b[1], b[0]], [b[3], b[2]]])
    folium.LayerControl(collapsed=False, position="topright").add_to(m)
    plugins.Fullscreen().add_to(m)
    plugins.MeasureControl(primary_length_unit="meters", primary_area_unit="hectares").add_to(m)
    plugins.MousePosition(position="bottomleft", separator=" , ", num_digits=6,
                          prefix="lat , lon:").add_to(m)

    k = result.kpis
    rows = [
        ("DC / AC", f"{k['dc_kwp'] / 1000:.2f} MWp / {k['ac_kw'] / 1000:.2f} MW"),
        ("Area", f"{k['available_ha']:.1f} ha ({k['mwp_per_ha']:.2f} MWp/ha)"),
        ("Tilt · GCR", f"{k['tilt_deg']:.1f}° · {k['gcr']:.3f}"),
        ("Tables", f"{k['n_tables']:,}  (+{k['tables_earthworks']} earthworks)"),
        ("Excluded", f"{k['tables_excluded_terrain']} by slope"),
        ("Inverters · PS", f"{k['n_inverters']} · {k['n_power_stations']}"),
    ]
    if "annual_mwh" in k:
        rows += [("Energy", f"{k['annual_mwh']:,.0f} MWh/yr"),
                 ("Yield · PR", f"{k['specific_yield_kwh_kwp']:.0f} kWh/kWp · {100 * k['pr_poa']:.1f}%")]
    table = "".join(f"<tr><td style='color:#666;padding-right:10px'>{html.escape(a)}</td>"
                    f"<td style='text-align:right'><b>{html.escape(v)}</b></td></tr>" for a, v in rows)
    box = (
        "<div style='position:fixed;top:12px;left:56px;z-index:9999;background:rgba(255,255,255,.93);"
        "padding:10px 14px;border-radius:8px;font:12px/1.45 system-ui,sans-serif;"
        "box-shadow:0 2px 8px rgba(0,0,0,.3);max-width:330px'>"
        f"<div style='font-size:14px;font-weight:700;margin-bottom:4px'>{html.escape(site.name)}</div>"
        f"<table>{table}</table></div>"
    )
    m.get_root().html.add_child(folium.Element(box))
    m.get_root().header.add_child(folium.Element(
        f"<title>{html.escape(site.name)} — PV layout</title>"))
    m.save(str(path))
    return str(path)


def _match(all_centers, centers) -> np.ndarray:
    lookup = {tuple(np.round(c, 6)): i for i, c in enumerate(all_centers)}
    return np.array([lookup[tuple(np.round(c, 6))] for c in centers], int)
