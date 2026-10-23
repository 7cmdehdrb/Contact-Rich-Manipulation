"""CPU fixtures exercise terminal capture and schema; not physical samples."""

import csv
import importlib.util
import json
from pathlib import Path
import sys
import types
from types import SimpleNamespace

import pytest
import torch

from conftest import MODULE_ROOT, PACKAGE_ROOT
from test_omni_configuration import _execute_definitions


def load_file(path):
    from conftest import _module_from_file
    return _module_from_file("_fusion_telemetry_test", path)


@pytest.fixture
def harness(monkeypatch):
    # The shared runtime file also contains simulator Env subclasses. Load only
    # the telemetry unit and give its method-local relative imports a CPU package.
    package = types.ModuleType("_fusion_telemetry_cpu")
    package.__path__ = []
    monkeypatch.setitem(sys.modules, package.__name__, package)
    module = types.ModuleType("_fusion_telemetry_cpu.contact_sweep_clutter_fusion_telemetry")
    module.__package__ = package.__name__
    module.__dict__.update(csv=csv, Path=Path, torch=torch)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    _execute_definitions(MODULE_ROOT / "contact_sweep_clutter_fusion/telemetry.py", module.__dict__)
    sample = SimpleNamespace(pure_world=torch.arange(12, dtype=torch.float32).reshape(2, 6))
    sample.pure_eef = sample.pure_world.clone()
    velocities = torch.arange(24, dtype=torch.float32).reshape(2, 4, 3)
    env = SimpleNamespace(
        num_envs=2, device="cpu", sensor_data_fresh=torch.ones(2, dtype=torch.bool),
        episode_length_buf=torch.ones(2, dtype=torch.long),
        reset_terminated=torch.tensor([False, True]), reset_time_outs=torch.zeros(2, dtype=torch.bool),
        target_id=torch.zeros((2, 1), dtype=torch.long),
        scene={"object_collection": SimpleNamespace(
            object_names=["mass_cylinder", "obstacle_0", "obstacle_1", "obstacle_2"],
            data=SimpleNamespace(object_lin_vel_w=velocities))},
        observation_manager=SimpleNamespace(cfg=SimpleNamespace(policy=SimpleNamespace(
            wrist_wrench=SimpleNamespace(func=SimpleNamespace(_read=lambda env: sample))))),
    )
    telemetry = module.PlaybackTelemetry(env)
    telemetry.reset(torch.arange(2))
    return SimpleNamespace(module=module, env=env, sample=sample, telemetry=telemetry)


def test_terminal_ft_and_all_object_velocities_survive_partial_reset(harness, tmp_path):
    h = harness; t = h.telemetry; env = h.env
    writer = h.module.PlaybackCSVWriter(tmp_path, t)
    t.begin_step()
    t.capture(env, torch.tensor([1]))
    t.reset(torch.tensor([1]))
    env.sensor_data_fresh[1] = False
    h.sample.pure_world[1] = 999
    h.sample.pure_eef[1] = 999
    env.scene["object_collection"].data.object_lin_vel_w[1] = 999
    t.capture(env, torch.tensor([0])); t.finish_step()
    writer.write(t, .02)
    assert t.values["wrench_world.csv"][1, 0] == 6
    assert t.values["object_velocity.csv"][1].tolist() == list(range(12, 24))
    assert t.values["episodes.csv"][1].tolist() == [0, 1, 1, 0, 0, 0]

    env.sensor_data_fresh[:] = True
    env.reset_terminated[:] = False
    env.episode_length_buf[:] = torch.tensor([2, 1])
    t.begin_step(); t.capture(env, torch.arange(2)); writer.write(t, .04); writer.close()
    assert t.values["episodes.csv"][0, 4] == 1
    assert t.values["episodes.csv"][1].tolist() == [1, 1, 0, 0, 0, 0]
    with (tmp_path / "object_velocity.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert float(rows[0]["vx_Target_1"]) == 12
    assert float(rows[0]["vz_obstacle_2_1"]) == 23
    assert float(rows[1]["vx_Target_1"]) == 999
    with pytest.raises(FileExistsError):
        h.module.PlaybackCSVWriter(tmp_path, t)

    metadata = dict(status="complete", arguments=dict(num_envs=2, steps=2),
                    object_labels=t.object_labels, object_names=t.object_names,
                    control_dt_s=.02, csv_rows_per_file=2)
    (tmp_path / "metadata.json").write_text(json.dumps(metadata))
    validator = load_file(PACKAGE_ROOT / "scripts/validate_fusion_telemetry.py")
    assert validator.validate(tmp_path)["terminal_transitions"] == 1
    metadata["arguments"]["steps"] = 3
    (tmp_path / "metadata.json").write_text(json.dumps(metadata))
    with pytest.raises(AssertionError, match="Incomplete"):
        validator.validate(tmp_path)
    metadata["arguments"]["steps"] = 2
    (tmp_path / "metadata.json").write_text(json.dumps(metadata))
    path = tmp_path / "episodes.csv"
    with path.open() as stream:
        reader = csv.DictReader(stream)
        columns = reader.fieldnames
        rows = list(reader)
    rows[1]["delta_valid_1"] = "1"
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader(); writer.writerows(rows)
    with pytest.raises(AssertionError, match="Delta crosses"):
        validator.validate(tmp_path)


def test_reject_missing_duplicate_stale_and_nonfinite_measurements(harness):
    t = harness.telemetry; env = harness.env
    t.begin_step(); t.capture(env, torch.tensor([0]))
    with pytest.raises(RuntimeError, match="Missing"): t.finish_step()
    with pytest.raises(RuntimeError, match="twice"): t.capture(env, torch.tensor([0]))
    env.sensor_data_fresh[1] = False
    with pytest.raises(RuntimeError, match="Stale"): t.capture(env, torch.tensor([1]))
    env.sensor_data_fresh[1] = True
    env.scene["object_collection"].data.object_lin_vel_w[1, 3, 2] = float("nan")
    with pytest.raises(RuntimeError, match="Non-finite"): t.capture(env, torch.tensor([1]))


def test_runtime_keeps_parent_step_and_captures_before_reset():
    events = []
    class Parent:
        def __init__(self, cfg, render_mode=None, **kwargs):
            self.sensor_data_fresh = torch.ones(2, dtype=torch.bool)
        def _reset_idx(self, ids):
            events.append("parent_reset")
            self.sensor_data_fresh[ids] = False
        def step(self, actions):
            self._reset_idx(torch.tensor([1]))
            return "parent_result"
    class Telemetry:
        def __init__(self, env): pass
        def begin_step(self): events.append("begin")
        def capture(self, env, ids): events.append(("capture", ids.tolist()))
        def reset(self, ids): events.append("reset_history")
        def finish_step(self): events.append("finish")
    module = _execute_definitions(MODULE_ROOT / "contact_sweep_clutter_fusion/telemetry_env.py",
        {"ContactSweepClutterFusionEnv": Parent, "PlaybackTelemetry": Telemetry})
    env = module.ContactSweepClutterFusionTelemetryEnv(None)
    assert env.step(None) == "parent_result"
    assert events == ["begin", ("capture", [1]), "parent_reset", "reset_history", ("capture", [0]), "finish"]
