"""Check all 17 tactile channels with neighboring sensor frames from the real USD."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from conftest import MODULE_ROOT


def test_actual_usd_sensor_frames_activate_each_channel_and_preserve_assets():
    configured = os.environ.get("INSPIRE_SWEEP_USD_TEST_PYTHON")
    usd_python = Path(configured) if configured else Path(sys.executable).resolve().parents[2] / "newton/bin/python"
    if not usd_python.is_file():
        pytest.skip("Set INSPIRE_SWEEP_USD_TEST_PYTHON to a Python with pxr and torch installed")
    probe = Path(__file__).with_name("_tactile_geometry_probe.py")
    completed = subprocess.run(
        [str(usd_python), "-B", str(probe), str(MODULE_ROOT)],
        capture_output=True, text=True, check=True, timeout=60,
    )
    result = json.loads(completed.stdout)
    assert result["region_count"] == 17
    assert result["parent_count"] == 12
    assert result["activated_channels"] == list(range(17))
    assert result["simultaneous_activated_channels"] == list(range(17))
    assert result["neighbor_frames"] == "original_usd"
    assert result["source_hashes_unchanged"]
    assert result["metadata_source_hashes_match"]
