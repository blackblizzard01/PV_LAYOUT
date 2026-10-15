"""
3D terrain view (plotly → standalone HTML): DEM surface, tilted tables on
their posts, power stations, MV routes and boundary draped on the terrain.
Drag to rotate, scroll to zoom, click legend entries to toggle layers,
buttons switch vertical exaggeration.
"""

from __future__ import annotations

import math
import pathlib

import numpy as np

from ..terrain import EARTHWORKS, EXCLUDED

_MAX_GRID = 250


def _require_plotly():
    try:
        import plotly.graph_objects as go
    except ImportError as exc:  # pragma: no cover
        raise ImportError('plotly is required for view3d.html. Install with: '
                          'pip install "pv-layout-design[viz]"') from exc
    return go


def _table_mesh(dem, centers, width, slope_len, depth, tilt_deg, post_h, south_low: bool):
    n = len(centers)
    x, y = centers[:, 0], centers[:, 1]
    low_y = y - depth / 2 if south_low else y + depth / 2
    high_y = y + depth / 2 if south_low else y - depth / 2
    rise = slope_len * math.sin(math.radians(tilt_deg))
    xw, xe = x - width / 2, x + width / 2
    z_lw = dem.elevations_at(xw, low_y) + post_h
    z_le = dem.elevations_at(xe, low_y) + post_h
    verts = np.empty((n, 4, 3))
    verts[:, 0] = np.column_stack([xw, low_y, z_lw])
    verts[:, 1] = np.column_stack([xe, low_y, z_le])
    verts[:, 2] = np.column_stack([xe, high_y, z_le + rise])
    verts[:, 3] = np.column_stack([xw, high_y, z_lw + rise])
    base = np.arange(n) * 4
    i = np.concatenate([base, base])
    j = np.concatenate([base + 1, base + 2])
    k = np.concatenate([base + 2, base + 3])
    v = verts.reshape(-1, 3)
    return v[:, 0], v[:, 1], v[:, 2], i, j, k


def _drape(dem, pts, step=5.0, lift=0.5):
    out = []
    for a, b in zip(pts[:-1], pts[1:]):
        L = float(np.hypot(*(b - a)))
        n = max(2, int(L / step) + 1)
        t = np.linspace(0, 1, n)
        out.append(a + (b - a) * t[:, None])
    xy = np.vstack(out)
    return xy[:, 0], xy[:, 1], dem.elevations_at(xy[:, 0], xy[:, 1]) + lift


def render_view3d_html(result, path, z_exaggeration: float = 3.0, offline: bool = True) -> str:
    go = _require_plotly()
    dem, lay, elec, site = result.dem, result.tables, result.elec, result.site
    south_low = site.lat >= 0
    s = result.structure

    sr = max(1, int(math.ceil(dem.elevations.shape[0] / _MAX_GRID)))
    sc = max(1, int(math.ceil(dem.elevations.shape[1] / _MAX_GRID)))
    Z = dem.elevations[::sr, ::sc]
    X = dem.x_coords[::sc]
    Y = dem.y_coords[::sr]

    traces = [go.Surface(
        x=X, y=Y, z=Z, name="Terrain", showscale=False, opacity=1.0,
        colorscale=[[0, "#3b5d3a"], [0.35, "#6b8f4e"], [0.7, "#b5a46a"], [1, "#e4d9b5"]],
        lighting=dict(ambient=0.55, diffuse=0.8, roughness=0.9, specular=0.05),
        hovertemplate="x %{x:.0f} m<br>y %{y:.0f} m<br>z %{z:.1f} m<extra>terrain</extra>",
        showlegend=True,
    )]

    groups = [
        (result.status != EARTHWORKS, lay.centers, "Structures – accepted", "#2f6db5"),
        (result.status == EARTHWORKS, lay.centers, "Structures – earthworks", "#f0a020"),
        (result.terrain_all.status == EXCLUDED, result.tables_all.centers,
         "Structures – excluded", "#d62728"),
    ]
    for mask, centers, name, color in groups:
        if not np.any(mask):
            continue
        x, y, z, i, j, k = _table_mesh(dem, centers[mask], lay.width, lay.slope_length, lay.depth,
                                       lay.tilt_deg, s.post_height_m, south_low)
        traces.append(go.Mesh3d(x=x, y=y, z=z, i=i, j=j, k=k, color=color, flatshading=True,
                                name=f"{name} ({int(np.sum(mask))})", showlegend=True,
                                opacity=0.55 if "excluded" in name else 1.0,
                                lighting=dict(ambient=0.6, diffuse=0.7, specular=0.3),
                                hoverinfo="name"))

    ps = elec.ps_xy
    traces.append(go.Scatter3d(
        x=ps[:, 0], y=ps[:, 1], z=dem.elevations_at(ps[:, 0], ps[:, 1]) + 3,
        mode="markers+text", marker=dict(size=6, symbol="square", color="#8e1b1b",
                                         line=dict(color="white", width=1)),
        text=[f"PS{i + 1}" for i in range(len(ps))], textfont=dict(color="white", size=10),
        name=f"Power stations ({len(ps)})"))
    sx, sy = elec.substation_xy
    traces.append(go.Scatter3d(
        x=[sx], y=[sy], z=dem.elevations_at([sx], [sy]) + 4, mode="markers+text",
        marker=dict(size=8, symbol="diamond", color="white"), text=["Substation"],
        textfont=dict(color="white"), name="Substation"))

    mx, my, mz = [], [], []
    for r in elec.routes_of("mv"):
        x, y, z = _drape(dem, r.points)
        mx += [*x, None]; my += [*y, None]; mz += [*z, None]
    if mx:
        traces.append(go.Scatter3d(x=mx, y=my, z=mz, mode="lines",
                                   line=dict(color="#ff3b30", width=5), name="MV cables"))

    bx, by, bz = [], [], []
    for _, p in site.available:
        x, y, z = _drape(dem, np.asarray(p.exterior.coords), lift=1.0)
        bx += [*x, None]; by += [*y, None]; bz += [*z, None]
    traces.append(go.Scatter3d(x=bx, y=by, z=bz, mode="lines", line=dict(color="white", width=4),
                               name="Site boundary"))
    if site.restricted:
        rx, ry, rz = [], [], []
        for _, p in site.restricted:
            x, y, z = _drape(dem, np.asarray(p.exterior.coords), lift=1.0)
            rx += [*x, None]; ry += [*y, None]; rz += [*z, None]
        traces.append(go.Scatter3d(x=rx, y=ry, z=rz, mode="lines",
                                   line=dict(color="#ff6b6b", width=4, dash="dash"),
                                   name="Restricted areas"))

    xr = float(X.max() - X.min())
    yr = float(Y.max() - Y.min())
    zr = max(float(Z.max() - Z.min()), 2.0)

    def aspect(f):
        return dict(x=1.0, y=yr / xr, z=max(zr / xr * f, 0.02))

    buttons = [dict(label=f"{f:g}× vertical", method="relayout",
                    args=[{"scene.aspectratio": aspect(f)}]) for f in (1, 3, 5, 10)]
    k = result.kpis
    fig = go.Figure(traces)
    fig.update_layout(
        title=dict(text=f"{site.name} — {k['dc_kwp'] / 1000:.2f} MWp · {k['n_tables']:,} tables · "
                        f"terrain {Z.min():.0f}–{Z.max():.0f} m", font=dict(color="white", size=16)),
        paper_bgcolor="#0b1026", font=dict(color="#dfe6ee"),
        legend=dict(bgcolor="rgba(20,25,50,.8)", bordercolor="#445", borderwidth=1,
                    itemclick="toggle", title=dict(text="Layers (click to toggle)")),
        scene=dict(
            xaxis=dict(title="East (m)", backgroundcolor="#0b1026", gridcolor="#334", showbackground=True),
            yaxis=dict(title="North (m)", backgroundcolor="#0b1026", gridcolor="#334", showbackground=True),
            zaxis=dict(title="Elevation (m)", backgroundcolor="#0b1026", gridcolor="#334", showbackground=True),
            aspectmode="manual", aspectratio=aspect(z_exaggeration),
            camera=dict(eye=dict(x=-0.9, y=-1.5, z=0.8)),
        ),
        updatemenus=[dict(type="buttons", direction="right", x=0.01, y=0.02, xanchor="left",
                          yanchor="bottom", active=_closest(z_exaggeration), buttons=buttons,
                          bgcolor="#dfe6ee", bordercolor="#8895b3", font=dict(color="#0b1026"))],
        margin=dict(l=0, r=0, t=60, b=0),
    )
    fig.write_html(str(path), include_plotlyjs=True if offline else "cdn", full_html=True,
                   default_height="100vh", default_width="100%",
                   config={"displaylogo": False, "scrollZoom": True})
    text = pathlib.Path(path).read_text(encoding="utf-8")
    text = text.replace("<head>", "<head><meta charset='utf-8'><title>3D view</title>"
                        "<style>html,body{margin:0;background:#0b1026;overflow:hidden}</style>", 1)
    pathlib.Path(path).write_text(text, encoding="utf-8")
    return str(path)


def _closest(f: float) -> int:
    opts = (1, 3, 5, 10)
    return min(range(len(opts)), key=lambda i: abs(opts[i] - f))
