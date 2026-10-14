"""Smoke tests for the reaching environment contracts."""

import torch

from src.learning.envs import ReachCartesianEnv, ReachEnvConfig, ReachJointEnv
from src.learning.utils.assets import validate_ur5e_description


def test_joint_env_shapes_and_step() -> None:
    env = ReachJointEnv(ReachEnvConfig(num_envs=3, seed=1))
    obs = env.get_observations()["policy"]
    assert obs.shape == (3, 21)
    actions = torch.zeros((3, 6))
    obs, rewards, dones, extras = env.step(actions)
    assert obs["policy"].shape == (3, 21)
    assert rewards.shape == (3,)
    assert dones.shape == (3,)
    assert "/reach/mean_distance" in extras["log"]


def test_cartesian_env_uses_three_dimensional_actions() -> None:
    env = ReachCartesianEnv(ReachEnvConfig(num_envs=2, seed=2))
    assert env.num_actions == 3
    actions = torch.zeros((2, 3))
    obs, rewards, dones, _ = env.step(actions)
    assert obs["policy"].shape == (2, 21)
    assert rewards.shape == (2,)
    assert dones.shape == (2,)


def test_ur5e_description_submodule_paths_exist() -> None:
    paths = validate_ur5e_description()
    assert paths.urdf_xacro.name == "ur.urdf.xacro"
