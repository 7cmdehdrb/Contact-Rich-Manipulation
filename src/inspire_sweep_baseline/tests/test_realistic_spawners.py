"""Production spawn-time bindings and preservation of source configuration."""

from _source_units import source_text

import ast
from copy import deepcopy
from dataclasses import dataclass, field, fields
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from conftest import MODULE_ROOT


def test_spawn_time_materials_and_cylinder_origin_before_physics_cloning():
    configured = os.environ.get("INSPIRE_SWEEP_USD_TEST_PYTHON")
    python = Path(configured) if configured else Path(sys.executable).resolve().parents[2] / "newton/bin/python"
    if not python.is_file():
        pytest.skip("Set INSPIRE_SWEEP_USD_TEST_PYTHON to a Python with pxr installed")
    completed = subprocess.run(
        [str(python), str(Path(__file__).with_name("_realistic_spawners_probe.py")), str(MODULE_ROOT)],
        check=True, capture_output=True, text=True,
    )
    result = json.loads(completed.stdout)
    assert result == {"delegated_roles": ["robot", "shelf", "object"], "mean_mass": 0.365, "com_z": 0.02, "hand_bodies": 30}


def test_spawner_conversion_preserves_all_source_settings_without_mutating_baseline():
    source = ast.parse(source_text(MODULE_ROOT / "assets/realistic_spawners.py"))
    source.body = [node for node in source.body if isinstance(node, ast.FunctionDef) and node.name == "replace_friction_spawner"]
    namespace = {"deepcopy": deepcopy, "fields": fields}
    exec(compile(source, "realistic_spawners.py", "exec"), namespace)

    @dataclass
    class Source:
        func: str = "baseline"
        usd_path: str = "local_robot.usd"
        sensor_height: float = 0.0254
        rigid_props: dict = field(default_factory=lambda: {"max_depenetration_velocity": 5.0})
        articulation_props: dict = field(default_factory=lambda: {"enabled_self_collisions": False})

    @dataclass
    class Replacement(Source):
        func: str = "realistic"
        friction_defaults: dict = field(default_factory=dict)

    original = Source()
    converted = namespace["replace_friction_spawner"](original, Replacement, {"restitution": 0.0})
    assert converted.func == "realistic" and original.func == "baseline"
    for name in ("usd_path", "sensor_height", "rigid_props", "articulation_props"):
        assert getattr(converted, name) == getattr(original, name)
    converted.rigid_props["max_depenetration_velocity"] = 1.0
    assert original.rigid_props["max_depenetration_velocity"] == 5.0
