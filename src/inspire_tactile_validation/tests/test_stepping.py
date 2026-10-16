import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

path = Path(__file__).resolve().parents[1] / "inspire_tactile_validation/stepping.py"
spec = importlib.util.spec_from_file_location("validation_stepping", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def fixture(step_multiplier=1, render_advances=False):
    calls = []
    sim = SimpleNamespace(current_time=0.0)
    def step(render=True):
        calls.append(("step", render))
        sim.current_time += .01 * (2 if render else step_multiplier)
    def render():
        calls.append(("render",))
        if render_advances:
            sim.current_time += .01
    sim.step, sim.render = step, render
    scene = SimpleNamespace(write_data_to_sim=lambda: calls.append(("write",)),
                            update=lambda dt: calls.append(("update", dt)))
    return sim, scene, calls


def test_physics_tick_does_not_use_rendering_dt_and_orders_sensor_update():
    sim, scene, calls = fixture()
    assert module.advance_physics(sim, scene, .01, render=True) == pytest.approx(.01)
    assert calls == [("write",), ("step", False), ("update", .01), ("render",)]


def test_extra_physics_tick_is_rejected():
    sim, scene, _ = fixture(step_multiplier=2)
    with pytest.raises(RuntimeError, match="Expected one"):
        module.advance_physics(sim, scene, .01)


def test_rendering_must_not_advance_physics():
    sim, scene, _ = fixture(render_advances=True)
    with pytest.raises(RuntimeError, match="Rendering advanced"):
        module.advance_physics(sim, scene, .01, render=True)
