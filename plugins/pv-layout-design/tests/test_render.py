import pathlib

import pytest

from pv_layout.__main__ import main
from pv_layout.design import DesignConfig, run_design, save_summary
from pv_layout.render import render_all

EXAMPLES = pathlib.Path(__file__).parent.parent / "examples" / "sites"


@pytest.fixture(scope="module")
def sanand():
    return run_design(DesignConfig.from_toml(EXAMPLES / "sanand" / "config.toml"))


def test_render_all_outputs(sanand, tmp_path):
    paths = render_all(sanand, tmp_path)
    assert set(paths) == {"layout_png", "slope_png", "kpis_png", "map_html", "view3d_html"}
    for p in ("layout.png", "slope.png", "kpis.png"):
        data = (tmp_path / p).read_bytes()
        assert data[:8] == b"\x89PNG\r\n\x1a\n" and len(data) > 50_000
    html = (tmp_path / "map.html").read_text(encoding="utf-8")
    for layer in ["Satellite", "Available areas", "Restricted areas", "accepted (",
                  "MV cables", "Power stations", "Substation", "Inverter blocks"]:
        assert layer in html, layer
    v3 = (tmp_path / "view3d.html").read_text(encoding="utf-8")
    assert "plotly" in v3.lower() and "Terrain" in v3 and "vertical" in v3


def test_summary_and_boq_files(sanand, tmp_path):
    save_summary(sanand, tmp_path)
    import json
    s = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert s["kpis"]["n_tables"] == sanand.kpis["n_tables"]
    boq = (tmp_path / "boq.csv").read_text(encoding="utf-8").splitlines()
    assert boq[0] == "item,quantity,unit" and any(l.startswith("modules,") for l in boq)


def test_cli_end_to_end(tmp_path, capsys):
    rc = main(["run", str(EXAMPLES / "darlington_point" / "config.toml"), "--out", str(tmp_path),
               "--no-map", "--no-3d"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Darlington Point" in out and (tmp_path / "layout.png").exists()


def test_cli_reports_config_errors(tmp_path, capsys):
    bad = tmp_path / "c.toml"
    bad.write_text("[structure]\ntilt = 5\n")
    assert main(["run", str(bad)]) == 2
    assert "unknown key" in capsys.readouterr().err
