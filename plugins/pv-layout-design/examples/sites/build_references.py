"""
Rebuild expected.json for every example site (needs network; tests do not).

For each site:
  1. run the design (uses cached dem.tif / pvgis_tmy.csv / pvgis_horizon.csv),
  2. query PVGIS PVcalc at the same tilt / azimuth / system loss → reference yield,
  3. query PVGIS MRcalc 2005–2020 horizontal irradiation → long-term GHI, so the
     test can correct for the TMY year being sunnier/duller than average,
  4. store a KPI snapshot for regression checks.

Usage:  python examples/sites/build_references.py [site ...]
"""

from __future__ import annotations

import datetime
import json
import pathlib
import statistics
import sys
import urllib.request

HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(HERE.parents[1]))

from pv_layout.design import DesignConfig, run_design  # noqa: E402

SITES = ["bhadla", "sanand", "fukushima", "darlington_point", "cambridgeshire"]
SYSTEM_LOSSES = ["soiling_loss_frac", "mismatch_loss_frac", "lid_loss_frac", "dc_cable_loss_frac",
                 "inverter_loss_frac", "clipping_loss_frac", "ac_cable_loss_frac",
                 "transformer_loss_frac", "availability_loss_frac"]
SNAPSHOT = ["dc_kwp", "ac_kw", "n_tables", "n_inverters", "n_power_stations", "gcr", "pitch_m",
            "tilt_deg", "azimuth_deg", "usable_ha", "tables_excluded_terrain", "tables_earthworks",
            "specific_yield_kwh_kwp", "pr_poa"]
API = "https://re.jrc.ec.europa.eu/api/v5_2"


def _get(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=90) as r:
        return json.loads(r.read())


def build(site: str) -> dict:
    res = run_design(DesignConfig.from_toml(HERE / site / "config.toml"))
    k, e = res.kpis, res.energy
    keep = 1.0
    for f in SYSTEM_LOSSES:
        keep *= 1 - float(getattr(e, f))
    loss = round(100 * (1 - keep), 2)
    aspect = 0 if res.site.lat >= 0 else 180          # PVGIS: 0 = south, 180 = north
    lat, lon = round(res.site.lat, 5), round(res.site.lon, 5)
    pvcalc = (f"{API}/PVcalc?lat={lat}&lon={lon}&peakpower=1&loss={loss}&angle={k['tilt_deg']}"
              f"&aspect={aspect}&mountingplace=free&outputformat=json")
    t = _get(pvcalc)["outputs"]["totals"]["fixed"]
    mrcalc = f"{API}/MRcalc?lat={lat}&lon={lon}&horirrad=1&startyear=2005&endyear=2020&outputformat=json"
    years: dict[int, float] = {}
    for row in _get(mrcalc)["outputs"]["monthly"]:
        years[row["year"]] = years.get(row["year"], 0.0) + row["H(h)_m"]
    ghi_lt = statistics.mean(years.values())
    return {
        "site": res.site.name,
        "retrieved": datetime.date.today().isoformat(),
        "pvgis": {
            "E_y_kwh_kwp": t["E_y"], "H_i_y_kwh_m2": t["H(i)_y"], "system_loss_pct": loss,
            "pvcalc_url": pvcalc,
            "ghi_longterm_kwh_m2": round(ghi_lt, 1),
            "tmy_ghi_kwh_m2": round(e.ghi_annual_kwh_m2, 1),
            "tmy_over_longterm": round(e.ghi_annual_kwh_m2 / ghi_lt, 4),
        },
        "snapshot": {key: k[key] for key in SNAPSHOT},
        "tolerance": {"snapshot_rel": 0.02, "yield_vs_pvgis_rel": 0.04},
    }


if __name__ == "__main__":
    for s in sys.argv[1:] or SITES:
        ref = build(s)
        (HERE / s / "expected.json").write_text(json.dumps(ref, indent=2) + "\n", encoding="utf-8")
        p = ref["pvgis"]
        ratio = ref["snapshot"]["specific_yield_kwh_kwp"] / (p["E_y_kwh_kwp"] * p["tmy_over_longterm"])
        print(f"{s:17s} ours {ref['snapshot']['specific_yield_kwh_kwp']:7.1f}  PVGIS {p['E_y_kwh_kwp']:7.1f}"
              f"  TMY/long-term {p['tmy_over_longterm']:.3f}  weather-corrected ratio {ratio:.3f}")
