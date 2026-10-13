"""USD material binding including the actual Hand's instance proxies."""

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from conftest import MODULE_ROOT


def test_realistic_materials_bind_hand_proxies_and_remain_private_and_reusable():
    configured = os.environ.get("INSPIRE_SWEEP_USD_TEST_PYTHON")
    python = Path(configured) if configured else Path(sys.executable).resolve().parents[2] / "newton/bin/python"
    if not python.is_file():
        pytest.skip("Set INSPIRE_SWEEP_USD_TEST_PYTHON to a Python with pxr installed")
    result = subprocess.run(
        [str(python), str(Path(__file__).with_name("_realistic_materials_probe.py")), str(MODULE_ROOT)],
        check=True, capture_output=True, text=True,
    )
    result = json.loads(result.stdout)
    assert result == {"hand_colliders_per_env": 30, "private_tool_materials": 2, "count_stable": True, "restitution_preserved": True}
