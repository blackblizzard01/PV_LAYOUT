"""Renderers for DesignResult: static PNGs, 2D satellite map (HTML), 3D terrain view (HTML)."""

from .png import render_kpis_png, render_layout_png, render_slope_png


def render_all(result, out_dir, png=True, map_html=True, view3d_html=True) -> dict:
    """Write every enabled output into ``out_dir``; returns {name: path}."""
    import pathlib
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = result.config.output
    paths = {}
    if png:
        paths["layout_png"] = render_layout_png(result, out_dir / "layout.png", dpi=cfg.dpi)
        paths["slope_png"] = render_slope_png(result, out_dir / "slope.png", dpi=cfg.dpi)
        if result.energy is not None:
            paths["kpis_png"] = render_kpis_png(result, out_dir / "kpis.png", dpi=cfg.dpi)
    if map_html and result.site.frame is not None:
        from .map_html import render_map_html
        paths["map_html"] = render_map_html(result, out_dir / "map.html")
    if view3d_html:
        from .view3d_html import render_view3d_html
        paths["view3d_html"] = render_view3d_html(result, out_dir / "view3d.html",
                                                  z_exaggeration=cfg.z_exaggeration)
    return paths
