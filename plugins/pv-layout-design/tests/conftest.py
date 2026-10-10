"""Add the package root to sys.path so tests can import pv_layout directly."""
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
