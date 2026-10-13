"""Inspect local USD closure and the production cylinder authoring helper."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from conftest import MODULE_ROOT


def test_local_asset_closure_and_bottom_heavy_cylinder_geometry():
    configured = os.environ.get("INSPIRE_SWEEP_USD_TEST_PYTHON")
    usd_python = Path(configured) if configured else Path(sys.executable).resolve().parents[2] / "newton/bin/python"
    if not usd_python.is_file():
        pytest.skip("Set INSPIRE_SWEEP_USD_TEST_PYTHON to a Python with pxr installed")
    probe = Path(__file__).with_name("_usd_probe.py")
    completed = subprocess.run(
        [str(usd_python), str(probe), str(MODULE_ROOT)],
        capture_output=True, text=True, check=True,
    )
    result = json.loads(completed.stdout)
    assert result["asset_layers"] == 5
    assert len(result["joint_names"]) == 18
    assert result["bottom_z"] == pytest.approx(0.0, abs=1e-7)
    assert result["top_z"] == pytest.approx(0.10)
    assert result["mass"] == pytest.approx(1.0)
    assert result["com_z"] == pytest.approx(0.02)
    assert result["inertia"] == pytest.approx([0.000933333333, 0.000933333333, 0.0008])
    center_z = (result["bottom_z"] + result["top_z"]) / 2
    assert 0.09 - center_z == pytest.approx(0.04)
