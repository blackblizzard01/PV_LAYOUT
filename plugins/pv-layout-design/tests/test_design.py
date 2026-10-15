import json

import pytest

from pv_layout.design import DesignConfig, run_design


def _site(tmp_path):
    p = tmp_path / "site.geojson"
    p.write_text(json.dumps({"type": "FeatureCollection", "features": [{
        "type": "Feature", "properties": {"name": "AA-1"},
        "geometry": {"type": "Polygon", "coordinates": [[[72.40, 23.00], [72.404, 23.00],
                                                        [72.404, 23.003], [72.40, 23.003], [72.40, 23.00]]]}}]}))
    return p


def test_unknown_key_lists_valid_keys():
    with pytest.raises(ValueError, match=r"\[structure\] unknown key\(s\) \['tilt'\]"):
        DesignConfig.from_dict({"structure": {"tilt": 20}})


def test_unknown_section():
    with pytest.raises(ValueError, match="Unknown config section"):
        DesignConfig.from_dict({"sites": {}})


def test_target_mode_requires_target():
    with pytest.raises(ValueError, match="target_kwp"):
        DesignConfig.from_dict({"layout": {"mode": "target"}})


def test_invalid_toml(tmp_path):
    p = tmp_path / "c.toml"
    p.write_text("[site\nfile=")
    with pytest.raises(ValueError, match="invalid TOML"):
        DesignConfig.from_toml(p)


def test_string_sizing_conflict_is_explained(tmp_path):
    _site(tmp_path)
    cfg = DesignConfig.from_dict({"electrical": {"mppt_min_v": 1400, "mppt_max_v": 1450},
                                  "energy": {"source": "none"}}, base_dir=tmp_path)
    with pytest.raises(ValueError, match="String sizing impossible"):
        run_design(cfg)


def test_odd_string_on_two_high_table(tmp_path):
    _site(tmp_path)
    cfg = DesignConfig.from_dict({"structure": {"strings_per_table": 1}, "electrical": {"t_min_c": -5},
                                  "energy": {"source": "none"}}, base_dir=tmp_path)
    with pytest.raises(ValueError, match="cannot fill"):
        run_design(cfg)


def test_missing_geotiff(tmp_path):
    _site(tmp_path)
    cfg = DesignConfig.from_dict({"terrain": {"source": "geotiff", "path": "nope.tif"},
                                  "energy": {"source": "none"}}, base_dir=tmp_path)
    with pytest.raises(FileNotFoundError, match="nope.tif"):
        run_design(cfg)


def test_offline_pvgis_fails_with_guidance(tmp_path):
    _site(tmp_path)
    cfg = DesignConfig.from_dict({"electrical": {"t_min_c": 0}}, base_dir=tmp_path)
    with pytest.raises(RuntimeError, match="clearsky"):
        run_design(cfg)


def test_small_clearsky_design_runs(tmp_path):
    _site(tmp_path)
    cfg = DesignConfig.from_dict({"electrical": {"t_min_c": 0}, "energy": {"source": "clearsky"}},
                                 base_dir=tmp_path)
    r = run_design(cfg)
    assert r.kpis["n_tables"] > 0 and r.energy.annual_kwh > 0
    assert r.losses["dc_cable_loss_pct"] > 0
