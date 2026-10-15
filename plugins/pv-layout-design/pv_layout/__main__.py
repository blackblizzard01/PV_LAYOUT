"""
Command line:  python -m pv_layout run <config.toml> [--out DIR] [--no-png] [--no-map] [--no-3d]
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import time


def _utf8_stdout():
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass


def main(argv=None) -> int:
    _utf8_stdout()
    p = argparse.ArgumentParser(prog="python -m pv_layout", description="PV site layout design")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run a design from a TOML config")
    r.add_argument("config", type=pathlib.Path)
    r.add_argument("--out", type=pathlib.Path, default=None,
                   help="output folder (default: <config dir>/output)")
    r.add_argument("--no-png", action="store_true")
    r.add_argument("--no-map", action="store_true")
    r.add_argument("--no-3d", action="store_true")
    args = p.parse_args(argv)

    from .design import DesignConfig, run_design, save_summary
    from .render import render_all

    t0 = time.perf_counter()
    try:
        cfg = DesignConfig.from_toml(args.config)
        result = run_design(cfg)
    except (ValueError, FileNotFoundError, RuntimeError, ImportError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    out = args.out or (args.config.parent / "output")
    save_summary(result, out)
    paths = render_all(result, out, png=cfg.output.png and not args.no_png,
                       map_html=cfg.output.map_html and not args.no_map,
                       view3d_html=cfg.output.view3d_html and not args.no_3d)

    k = result.kpis
    print(f"{result.site.name}")
    print(f"  DC / AC            {k['dc_kwp'] / 1000:9.3f} MWp / {k['ac_kw'] / 1000:.3f} MW "
          f"(ratio {k['dc_ac_ratio']:.3f})")
    print(f"  Tables / modules   {k['n_tables']:9,d} / {k['n_modules']:,d}   "
          f"(excluded {k['tables_excluded_terrain']}, earthworks {k['tables_earthworks']})")
    print(f"  Tilt / pitch / GCR {k['tilt_deg']:9.1f}° / {k['pitch_m']:.2f} m / {k['gcr']:.3f}")
    print(f"  Inverters / PS     {k['n_inverters']:9d} / {k['n_power_stations']}")
    if "annual_mwh" in k:
        print(f"  Energy             {k['annual_mwh']:9,.0f} MWh/yr   "
              f"{k['specific_yield_kwh_kwp']:.0f} kWh/kWp   PR {100 * k['pr_poa']:.1f} %")
    for w in result.warnings:
        print(f"  warning: {w}")
    print(f"  outputs → {out}")
    for name, path in paths.items():
        print(f"    {name:12s} {pathlib.Path(path).name}")
    print(f"  done in {time.perf_counter() - t0:.1f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
