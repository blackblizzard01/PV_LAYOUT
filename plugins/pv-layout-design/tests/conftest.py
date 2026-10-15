"""Add the package root to sys.path and forbid network access during tests."""
import pathlib
import sys
import urllib.request

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))


def _no_network(*args, **kwargs):
    raise RuntimeError("network access is disabled in tests — use the cached files")


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    import pvlib
    monkeypatch.setattr(urllib.request, "urlopen", _no_network)
    monkeypatch.setattr(pvlib.iotools, "get_pvgis_tmy", _no_network)
    monkeypatch.setattr(pvlib.iotools, "get_pvgis_horizon", _no_network)
