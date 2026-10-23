"""Independent numeric oracle, target selection and Arm-only speed reduction."""

from _source_units import source_text, unit_paths

import ast
import math
from types import SimpleNamespace

import pytest
import torch

from conftest import MODULE_ROOT


def test_no_renewal_config_references_quadratic_excess_speed_functions():
    legacy = {"pushing_speed_cost", "joint_speed_cost"}
    violations = []
    files = unit_paths(pattern="*renewal*env_cfg.py")
    assert files
    for file in files:
        for node in ast.walk(ast.parse(source_text(file))):
            if isinstance(node, (ast.Attribute, ast.Name)):
                name = node.attr if isinstance(node, ast.Attribute) else node.id
                if name in legacy:
                    violations.append(f"{file.name}:{node.lineno}: {name}")
    assert not violations, violations


def test_object_exponential_cost_uses_live_target_and_full_3d_speed(load_mdp, make_env):
    rewards = load_mdp("goal_rewards")
    env = make_env(7)
    velocities = torch.zeros(7, 2, 3, dtype=torch.float64)
    velocities[:, 0] = 20.0  # A different object's speed is irrelevant.
    velocities[:, 1] = torch.tensor([
        [0., 0., 0.], [0.3, 0., 0.], [0.300001, 0., 0.],
        [0.4, 0., 0.], [0.5, 0., 0.], [0., 0., -0.5], [0.3, 0.4, 0.],
    ], dtype=torch.float64)
    env.scene["object_collection"].data.object_lin_vel_w = velocities
    env.target_id[:] = 1
    expected = torch.tensor([
        0., 0., math.expm1(0.000002) / 2., math.expm1(0.2) / 2.,
        math.expm1(0.4) / 2., math.expm1(0.4) / 2., math.expm1(0.4) / 2.,
    ], dtype=torch.float64)
    actual = rewards.exponential_pushing_speed_cost(env)
    assert torch.allclose(actual, expected, rtol=1e-9, atol=1e-14)
    assert (actual[3] * -1.0 * 0.02).item() == pytest.approx(-0.0022140275816)
    velocities[:, 1].zero_()
    assert rewards.exponential_pushing_speed_cost(env).count_nonzero() == 0


def test_arm_exponential_cost_uses_max_absolute_selected_joint_not_sum(load_mdp, make_env):
    rewards = load_mdp("goal_rewards")
    env = make_env(6)
    ids = [1, 4, 7, 10, 13, 16]
    cfg = SimpleNamespace(name="robot", joint_ids=ids)
    env.scene["robot"].data.joint_vel = torch.zeros(6, 18, dtype=torch.float64)
    velocity = env.scene["robot"].data.joint_vel
    velocity[:, 0] = 100.0  # Excluded Hand joint has no effect.
    velocity[:, 7] = torch.tensor([0., 1., 1.5, 2., -2., 5.], dtype=torch.float64)
    expected = torch.tensor([
        0., 0., math.expm1(1.) / 2., math.expm1(2.) / 2.,
        math.expm1(2.) / 2., math.expm1(8.) / 2.,
    ], dtype=torch.float64)
    actual = rewards.exponential_joint_speed_cost(env, cfg)
    assert torch.allclose(actual, expected)
    velocity[3, ids] = 2.0
    assert rewards.exponential_joint_speed_cost(env, cfg)[3].item() == pytest.approx(expected[3].item())
    assert (actual[3] * -1.0 * 0.02).item() == pytest.approx(-0.0638905609893)
    # Exponential growth continues above one unit of excess; no saturation.
    assert actual[-1] > actual[3] * 100


@pytest.mark.parametrize("function", ["exponential_pushing_speed_cost", "exponential_joint_speed_cost"])
@pytest.mark.parametrize("threshold", [-0.1, math.nan, math.inf])
def test_exponential_threshold_requires_finite_nonnegative_value(load_mdp, make_env, function, threshold):
    rewards = load_mdp("goal_rewards")
    with pytest.raises(ValueError, match="finite and nonnegative"):
        getattr(rewards, function)(make_env(1), free_speed=threshold)
