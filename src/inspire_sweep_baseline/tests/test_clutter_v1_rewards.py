"""Behavioral checks for Clutter v1 rewards and reset snapshot lifecycle."""

import importlib.util
import math
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from conftest import MODULE_ROOT
from test_contact_sweep_clutter_mdp import multiobject_env
from test_omni_configuration import _execute_definitions


@pytest.fixture
def v1_mdp(load_mdp, monkeypatch):
    tactile = load_mdp("tactile_observations")
    package_name = "_baseline_cpu_pkg.contact_sweep_clutter_v1"
    package = ModuleType(package_name)
    package.__path__ = [str(MODULE_ROOT / "contact_sweep_clutter_v1")]
    monkeypatch.setitem(sys.modules, package_name, package)
    name = f"{package_name}.mdp"
    spec = importlib.util.spec_from_file_location(
        name, MODULE_ROOT / "contact_sweep_clutter_v1/mdp.py",
    )
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return SimpleNamespace(mdp=module, tactile=tactile)


def test_goal_hold_uses_selected_target_world_xy_and_includes_one_cm(v1_mdp, make_env):
    env = multiobject_env(make_env, count=4)
    positions = env.scene["object_collection"].data.object_pos_w
    positions.zero_()
    env.target_id[:, 0] = torch.tensor([1, 3, 2, 0])
    rows = torch.arange(4)
    positions[rows, env.target_id[:, 0]] = torch.tensor([
        [0.0099, 0.0, 100.0],
        [0.01, 0.0, -100.0],
        [0.0101, 0.0, 0.0],
        [0.008, 0.008, 0.0],
    ])
    requested_commands = []

    def command(name):
        requested_commands.append(name)
        return torch.zeros(4, 3)

    env.command_manager.get_command = command
    env.scene["shelf"].data.root_quat_w[:] = torch.tensor([0.5, 0.5, 0.5, 0.5])
    assert v1_mdp.mdp.goal_hold_xy(env).tolist() == [1.0, 1.0, 0.0, 0.0]
    assert requested_commands == ["target_goal_pos"]
    positions[..., 2].add_(1000.0)
    assert v1_mdp.mdp.goal_hold_xy(env).tolist() == [1.0, 1.0, 0.0, 0.0]


@pytest.mark.parametrize("tolerance", [0.0, -0.01, math.inf, math.nan])
def test_goal_hold_rejects_invalid_radius(v1_mdp, make_env, tolerance):
    with pytest.raises(ValueError, match="position_tolerance"):
        v1_mdp.mdp.goal_hold_xy(make_env(), position_tolerance=tolerance)


def test_tactile_rewards_count_channels_and_saturate_at_five(v1_mdp, make_env):
    env = make_env(5)
    counts = torch.tensor([0, 1, 5, 6, 17])
    bits = (torch.arange(17)[None, :] < counts[:, None]).to(torch.float32)
    env.observation_manager = SimpleNamespace(cfg=SimpleNamespace(
        policy=SimpleNamespace(tactile=SimpleNamespace(func=lambda env: bits, params={})),
    ))
    any_contact = v1_mdp.mdp.contact_any(env)
    contact_count = v1_mdp.mdp.contact_count(env)
    assert torch.equal(any_contact, torch.tensor([0.0, 1.0, 1.0, 1.0, 1.0]))
    assert torch.equal(contact_count, torch.tensor([0.0, 0.2, 1.0, 1.0, 1.0]))
    assert torch.equal(any_contact[2:], contact_count[2:])


@pytest.mark.parametrize("max_contacts", [0, -1, 2.5, math.inf])
def test_contact_count_rejects_invalid_limit(v1_mdp, make_env, max_contacts):
    with pytest.raises(ValueError, match="max_contacts"):
        v1_mdp.mdp.contact_count(make_env(), max_contacts=max_contacts)


def test_shared_tactile_cache_reads_once_per_step_and_invalidates_on_partial_reset(
    v1_mdp, monkeypatch, make_env,
):
    env = make_env(3)
    env.common_step_counter = 42
    live_bits = torch.zeros(3, 17)
    live_bits[:, :5] = 1.0
    calls = []

    def read_current(self, env, sensor_name="hand_tactile", threshold_n=0.01):
        calls.append((env.common_step_counter, sensor_name, threshold_n))
        return torch.where(env.sensor_data_fresh[:, None], live_bits, 0.0)

    parent = v1_mdp.tactile.ProjectedTactileContactBits
    monkeypatch.setattr(parent, "__init__", lambda self, cfg, env: None)
    monkeypatch.setattr(parent, "__call__", read_current)
    shared = v1_mdp.mdp.SharedTactileContactBits(SimpleNamespace(params={}), env)
    term = SimpleNamespace(func=shared, params={"threshold_n": 0.01})
    env.observation_manager = SimpleNamespace(cfg=SimpleNamespace(
        policy=SimpleNamespace(tactile=term),
    ))
    assert v1_mdp.mdp.contact_any(env).tolist() == [1.0, 1.0, 1.0]
    assert v1_mdp.mdp.contact_count(env).tolist() == [1.0, 1.0, 1.0]
    shared(env).zero_()  # Observation mutation must not contaminate rewards.
    assert shared(env).sum().item() == 15.0
    assert len(calls) == 1

    env.common_step_counter += 1
    live_bits[:, 1:] = 0.0
    assert torch.equal(v1_mdp.mdp.contact_count(env), torch.full((3,), 0.2))
    assert len(calls) == 2

    env.sensor_data_fresh[1] = False
    shared.reset(torch.tensor([1]))
    assert v1_mdp.mdp.contact_any(env).tolist() == [1.0, 0.0, 1.0]
    assert len(calls) == 3
    # A changed read parameter cannot reuse a cached reading.
    shared(env, threshold_n=0.02)
    assert len(calls) == 4


def test_obstacle_cost_uses_exact_3d_snapshot_and_excludes_each_selected_target(
    v1_mdp, make_env,
):
    env = multiobject_env(make_env, count=3)
    positions = env.scene["object_collection"].data.object_pos_w
    env._initial_object_positions_w = positions.clone()
    env.target_id[:, 0] = torch.tensor([0, 2, 3])
    # Observation noise is deliberately unrelated to the physical reset state.
    env._initial_obstacle_observation = torch.full((3, 40), 123.0)
    for row, target in enumerate(env.target_id[:, 0]):
        obstacles = [index for index in range(4) if index != target]
        positions[row, target] += 20.0
        positions[row, obstacles[0], 0] += 0.1
        positions[row, obstacles[1], 2] += 0.1
    assert torch.allclose(
        v1_mdp.mdp.obstacle_displacement_l2(env), torch.full((3,), 0.02), atol=1e-7,
    )
    assert torch.allclose(
        v1_mdp.mdp.obstacle_displacement_l2(env, max_cost=0.01),
        torch.full((3,), 0.01),
    )


def test_obstacle_cost_is_finite_bounded_and_ignores_invalid_target_state(v1_mdp, make_env):
    env = multiobject_env(make_env, count=7)
    positions = env.scene["object_collection"].data.object_pos_w
    env._initial_object_positions_w = positions.clone()
    positions[0, 1, 0] = 1.0e30
    positions[1, 1, 0] = math.inf
    positions[2, 1, 0] = -math.inf
    positions[3, 1, 0] = math.nan
    positions[4, 0, 0] = math.nan
    positions[5, 0, 0] = math.inf
    positions[6, 0, 0] = -math.inf
    actual = v1_mdp.mdp.obstacle_displacement_l2(env)
    assert torch.isfinite(actual).all()
    assert torch.equal(actual, torch.tensor([0.04] * 4 + [0.0] * 3))


@pytest.mark.parametrize("max_cost", [0.0, -0.01, math.inf, math.nan])
def test_obstacle_cost_rejects_invalid_cap(v1_mdp, make_env, max_cost):
    with pytest.raises(ValueError, match="max_cost"):
        v1_mdp.mdp.obstacle_displacement_l2(make_env(), max_cost=max_cost)


def test_partial_reset_captures_new_physical_positions_after_parent_reset(make_env):
    class Parent:
        def _reset_idx(self, ids):
            # Parent reset events must have run before the child's snapshot.
            self.scene["object_collection"].data.object_pos_w[ids] += 7.0
            self.parent_reset_ids = ids.clone()

    namespace = {"torch": torch, "ContactSweepClutterEnv": Parent}
    runtime = _execute_definitions(
        MODULE_ROOT / "contact_sweep_clutter_v1/env.py", namespace,
    ).ContactSweepClutterV1Env
    env = runtime.__new__(runtime)
    env.__dict__.update(vars(multiobject_env(make_env, count=4)))
    env._initial_object_positions_w = torch.full((4, 4, 3), -10.0)
    before = env._initial_object_positions_w.clone()
    positions = env.scene["object_collection"].data.object_pos_w
    old_positions = positions.clone()
    ids = torch.tensor([3, 1])
    env._reset_idx(ids)
    assert torch.equal(env.parent_reset_ids, ids)
    assert torch.equal(env._initial_object_positions_w[ids], old_positions[ids] + 7.0)
    assert torch.equal(env._initial_object_positions_w[[0, 2]], before[[0, 2]])
    positions[ids] += 99.0
    assert torch.equal(env._initial_object_positions_w[ids], old_positions[ids] + 7.0)


def test_runtime_allocates_snapshot_before_parent_initialization():
    class Parent:
        def __init__(self, cfg, render_mode, **kwargs):
            assert self._initial_object_positions_w.shape == (3, 4, 3)
            assert torch.count_nonzero(self._initial_object_positions_w) == 0
            self.parent_args = (render_mode, kwargs)

    namespace = {"torch": torch, "ContactSweepClutterEnv": Parent}
    runtime = _execute_definitions(
        MODULE_ROOT / "contact_sweep_clutter_v1/env.py", namespace,
    ).ContactSweepClutterV1Env
    cfg = SimpleNamespace(
        scene=SimpleNamespace(
            num_envs=3, object_collection=SimpleNamespace(rigid_objects=dict.fromkeys(range(4))),
        ),
        sim=SimpleNamespace(device="cpu"),
    )
    env = runtime(cfg, render_mode="rgb_array", example_argument=7)
    assert env.parent_args == ("rgb_array", {"example_argument": 7})


def test_v1_registration_selects_new_runtime_and_preserves_v0_registration():
    registrations = {}
    gym = SimpleNamespace(registry=registrations)
    gym.register = lambda **kwargs: registrations.update({kwargs["id"]: kwargs})
    _execute_definitions(MODULE_ROOT / "__init__.py", {"gym": gym})
    for version, package, prefix in (
        ("v0", "contact_sweep_clutter", "ContactSweepClutter"),
        ("v1", "contact_sweep_clutter_v1", "ContactSweepClutterV1"),
    ):
        registered = registrations[f"Isaac-Inspire-Contact-Sweep-Clutter-{version}"]
        assert registered["entry_point"] == f"inspire_sweep_baseline.{package}.env:{prefix}Env"
        assert registered["kwargs"]["env_cfg_entry_point"] == (
            f"inspire_sweep_baseline.{package}.env_cfg:{prefix}EnvCfg"
        )
        assert registered["kwargs"]["rsl_rl_cfg_entry_point"] == (
            f"inspire_sweep_baseline.{package}.ppo_cfg:{prefix}PPORunnerCfg"
        )
