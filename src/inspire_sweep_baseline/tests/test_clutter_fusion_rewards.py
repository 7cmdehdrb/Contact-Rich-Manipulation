"""Reward merges retain separate saturation and unequal Arm/Hand weighting."""

import importlib.util
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from conftest import MODULE_ROOT
from test_contact_sweep_clutter_mdp import multiobject_env


@pytest.fixture
def fusion_rewards(load_mdp, monkeypatch):
    load_mdp("goal_rewards")
    load_mdp("tactile_observations")
    result = {}
    for directory, name in (
        ("contact_sweep_clutter", "mdp"),
        ("contact_sweep_clutter_v1", "mdp"),
        ("contact_sweep_clutter_fusion", "reward_functions"),
    ):
        package_name = f"_baseline_cpu_pkg.{directory}"
        package = ModuleType(package_name)
        package.__path__ = [str(MODULE_ROOT / directory)]
        monkeypatch.setitem(sys.modules, package_name, package)
        fullname = f"{package_name}.{name}"
        spec = importlib.util.spec_from_file_location(fullname, MODULE_ROOT / directory / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, fullname, module)
        spec.loader.exec_module(module)
        result[directory] = module
    return result["contact_sweep_clutter_fusion"]


@pytest.mark.parametrize("dt", [0.01, 0.02, 0.1])
def test_merged_action_keeps_raw_inputs_separate_caps_and_exact_reward(fusion_rewards, make_env, dt):
    env = make_env(5)
    env.action_manager.action[:] = torch.tensor([
        [1., -2., 3., -4., 5., -6., 8., -9.],
        [1., 2., 0., 0., 0., 0., 2., -3.],
        [100., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 100., -100.],
        [100., 0., 0., 0., 0., 0., 100., 0.],
    ])
    env.action_manager.prev_action[:, -2:] = 1.
    delta = env.action_manager.action - env.action_manager.prev_action
    arm = delta[:, :6].square().sum(-1).clamp_max(100)
    hand = delta[:, 6:].square().sum(-1).clamp_max(100)
    merged = fusion_rewards.combined_action_rate(env)
    assert torch.equal(merged, arm + hand)
    assert merged[-1] == 200  # A cap on the combined sum would change the reward.
    assert torch.allclose(-.03 * merged * dt, (-.03 * arm - .03 * hand) * dt)
    assert torch.allclose(-.06 * merged * dt, 2 * (-.03 * arm - .03 * hand) * dt)


def test_merged_joint_velocity_uses_named_groups_and_unequal_weights(fusion_rewards, make_env):
    env = make_env(3)
    # Interleave indices to reject accidental first-six-DOF assumptions.
    arm = [0, 3, 6, 9, 12, 15]
    hand = [i for i in range(18) if i not in arm]
    env.scene["robot"].data.joint_vel[:] = torch.arange(54).reshape(3, 18) / 10
    velocity = env.scene["robot"].data.joint_vel
    a = velocity[:, arm].square().sum(-1).clamp_max(100)
    h = velocity[:, hand].square().sum(-1).clamp_max(100)
    merged = fusion_rewards.combined_joint_velocity(
        env, SimpleNamespace(name="robot", joint_ids=arm),
        SimpleNamespace(name="robot", joint_ids=hand),
    )
    assert torch.allclose(-.03 * merged, -.03 * a - .01 * h)


def test_object_merge_preserves_all_tilts_excludes_target_motion_and_scales_15x(
    fusion_rewards, make_env, tensor_math,
):
    env = multiobject_env(make_env, count=4)
    env._initial_object_positions_w = env.scene["object_collection"].data.object_pos_w.clone()
    angles = torch.tensor([[.0, .1, .3, .7], [.9, .9, .9, .9], [.0, .0, .0, .0], [.4, .2, .0, .0]])
    env.scene["object_collection"].data.object_quat_w[:] = tensor_math.quat_from_euler_xyz(
        angles, torch.zeros_like(angles), torch.zeros_like(angles),
    )
    env.scene["object_collection"].data.object_pos_w[:, 0, 0] += 100
    env.scene["object_collection"].data.object_pos_w[0, 1, 0] += .1
    env.scene["object_collection"].data.object_pos_w[1, 2, 0] += 100
    tilt = fusion_rewards.all_object_tilt_cost(env)
    displacement = fusion_rewards.obstacle_displacement_l2(env)
    merged = fusion_rewards.combined_object_cost(env)
    assert torch.allclose(-2 * merged, -2 * tilt - 5 * displacement)
    assert torch.allclose(-30 * merged, 15 * (-2 * tilt - 5 * displacement))
    assert displacement[2] == 0
    assert displacement[1] == pytest.approx(.04)
