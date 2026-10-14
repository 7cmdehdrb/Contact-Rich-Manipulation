"""Behavioral manager-term tests using real Torch without launching Isaac Sim."""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import torch


PACKAGE_ROOT = Path(__file__).resolve().parents[1] / "hand_manipulation_test"
TEST_PACKAGE = "_hand_manipulation_test_mdp_fixture"


class _ManagerTerm:
    def __init__(self, cfg, env):
        self.cfg = cfg
        self._env = env

    @property
    def num_envs(self):
        return self._env.num_envs

    @property
    def device(self):
        return self._env.device


class _CommandTerm(_ManagerTerm):
    """Small adapter for Isaac's public reset/compute manager protocol."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.metrics = {}
        self.time_left = torch.zeros(self.num_envs)
        self.command_counter = torch.zeros(self.num_envs, dtype=torch.long)
        self._set_debug_vis_impl(cfg.debug_vis)

    def reset(self, env_ids=None):
        index = slice(None) if env_ids is None else env_ids
        self.time_left[index] = self.cfg.resampling_time_range[0]
        self.command_counter[index] = 1
        self._resample_command(index)

    def compute(self, dt):
        self._update_metrics()
        self.time_left -= dt
        index = (self.time_left <= 0).nonzero().flatten()
        if len(index):
            self.reset(index)
        self._update_command()


def _load(name, relative_path):
    spec = importlib.util.spec_from_file_location(name, PACKAGE_ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _load_mdp():
    # Private package names avoid touching runtime registration or other tests'
    # imports. Only the two Isaac manager bases are replaced during loading.
    package = ModuleType(TEST_PACKAGE)
    package.__path__ = [str(PACKAGE_ROOT)]
    mdp = ModuleType(f"{TEST_PACKAGE}.mdp")
    mdp.__path__ = [str(PACKAGE_ROOT / "mdp")]
    sys.modules[TEST_PACKAGE] = package
    sys.modules[mdp.__name__] = mdp
    _load(f"{TEST_PACKAGE}.action_math", "action_math.py")
    geometry = ModuleType(f"{TEST_PACKAGE}.geometry")
    geometry.control_point_pose_w = lambda env: (env.eef_position, env.eef_quaternion)
    geometry.base_link_pose_w = lambda env: (env.base_position, env.base_quaternion)
    geometry.palm_tactile_bits = lambda env: env.palm_bits
    geometry.wrist_wrench_c = lambda env: SimpleNamespace(measured_c=env.wrench)
    sys.modules[geometry.__name__] = geometry
    managers = ModuleType("isaaclab.managers")
    managers.ManagerTermBase = _ManagerTerm
    managers.CommandTerm = _CommandTerm
    managers.CommandTermCfg = type("CommandTermCfg", (), {})
    utils = ModuleType("isaaclab.utils")
    utils.configclass = lambda cls: cls
    with patch.dict(sys.modules, {"isaaclab": ModuleType("isaaclab"), managers.__name__: managers, utils.__name__: utils}):
        commands = _load(f"{TEST_PACKAGE}.mdp.commands", "mdp/commands.py")
        observations = _load(f"{TEST_PACKAGE}.mdp.observations", "mdp/observations.py")
        rewards = _load(f"{TEST_PACKAGE}.mdp.rewards", "mdp/rewards.py")
        events = _load(f"{TEST_PACKAGE}.mdp.events", "mdp/events.py")
    return commands, observations, rewards, events


class _Scene(dict):
    pass


def _environment(count=2):
    scene = _Scene()
    scene.env_origins = torch.zeros(count, 3)
    scene["robot"] = SimpleNamespace(data=SimpleNamespace(joint_pos=torch.zeros(count, 18), joint_vel=torch.zeros(count, 18)))
    arm = SimpleNamespace(joint_ids=tuple(range(6)), raw_actions=torch.zeros(count, 6))
    hand = SimpleNamespace(
        raw_actions=torch.zeros(count, 2), processed_actions=torch.zeros(count, 2), actual_synergy=torch.full((count, 2), 0.5)
    )
    terms = {"arm_action": arm, "hand_action": hand}
    target = SimpleNamespace(
        target_pos_w=torch.zeros(count, 3), initial_eef_pos_w=torch.zeros(count, 3),
        initial_eef_quat_w=torch.tensor((1.0, 0.0, 0.0, 0.0)).expand(count, -1).clone(),
    )
    task = SimpleNamespace(
        arm_velocity_observation_scale=3.14, rotation_observation_scale_rad=math.pi,
        wrench_force_observation_scale_n=40.0, wrench_moment_observation_scale_nm=4.0,
        goal_reward_sigma_m=0.05, success_distance_m=0.01,
    )
    return SimpleNamespace(
        num_envs=count, device="cpu", scene=scene, cfg=SimpleNamespace(task=task),
        action_manager=SimpleNamespace(get_term=terms.__getitem__, action=torch.zeros(count, 8)),
        command_manager=SimpleNamespace(get_term=lambda name: target),
        eef_position=torch.zeros(count, 3), eef_quaternion=target.initial_eef_quat_w.clone(),
        base_position=torch.zeros(count, 3), base_quaternion=target.initial_eef_quat_w.clone(),
        palm_bits=torch.zeros(count, 17), wrench=torch.zeros(count, 6),
        episode_length_buf=torch.ones(count, dtype=torch.long), _sim_step_counter=0,
    )


class ReachingMdpTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.commands, cls.observations, cls.rewards, cls.events = _load_mdp()

    def test_target_position_uses_translated_rotated_base_link_without_clipping_metres(self):
        env = _environment()
        env.base_position[:] = torch.tensor(((3.0, -2.0, 0.79505), (-4.0, 1.0, 1.2)))
        env.base_quaternion[:] = torch.tensor(
            ((math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5)), (0.0, 0.0, 1.0, 0.0))
        )
        env.command_manager.get_term("target_position").target_pos_w[:] = torch.tensor(
            ((5.0, -2.0, 1.19505), (-4.0, 4.0, 1.8))
        )
        # Environment origins are deliberately unrelated to the actual base
        # poses. They must never stand in for the base_link transform.
        env.scene.env_origins[:] = torch.tensor(((100.0, 200.0, 300.0), (-100.0, -200.0, -300.0)))
        expected = torch.tensor(((0.0, -2.0, 0.4), (0.0, 3.0, -0.6)))
        before = self.observations.initial_target_relative_position(env)
        torch.testing.assert_close(before, expected, atol=1e-6, rtol=0)
        env.eef_position[:] = torch.tensor(((0.2, 0.4, 0.8), (-1.0, 2.0, 3.0)))
        env.eef_quaternion[:] = torch.tensor(
            ((0.0, 1.0, 0.0, 0.0), (math.sqrt(0.5), 0.0, 0.0, -math.sqrt(0.5)))
        )
        torch.testing.assert_close(
            self.observations.initial_target_relative_position(env), before, atol=0, rtol=0
        )

    def test_eef_displacement_uses_reset_frame_and_rotation_is_shortest(self):
        env = _environment()
        target = env.command_manager.get_term("target_position")
        rotation = torch.tensor((math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5)))
        target.initial_eef_quat_w[:] = rotation
        env.eef_quaternion[:] = rotation
        env.eef_position[:, 0] = 0.8
        torch.testing.assert_close(
            self.observations.eef_relative_position(env), torch.tensor(((0.0, -0.8, 0.0),) * 2), atol=1e-6, rtol=0
        )
        torch.testing.assert_close(self.observations.eef_relative_orientation(env), torch.zeros(2, 3))
        target.initial_eef_quat_w[:] = torch.tensor((1.0, 0.0, 0.0, 0.0))
        expected = torch.tensor(((0.0, 0.0, 0.5),) * 2)
        torch.testing.assert_close(self.observations.eef_relative_orientation(env), expected)
        env.eef_quaternion *= -1
        torch.testing.assert_close(self.observations.eef_relative_orientation(env), expected)

    def test_sensor_reset_mask_constant_palm_header_and_uncancelled_wrench(self):
        env = _environment()
        env.palm_bits[:] = 1
        env.wrench[:] = torch.tensor((40.0, 80.0, -10.0, 4.0, 8.0, -2.0))
        env.episode_length_buf[:] = torch.tensor((0, 1))
        tactile = self.observations.surface_header_and_tactile(env)
        self.assertEqual(tuple(tactile.shape), (2, 18))
        torch.testing.assert_close(tactile[:, 0], torch.zeros(2))
        torch.testing.assert_close(tactile[0], torch.zeros(18))
        torch.testing.assert_close(tactile[1, 1:], torch.ones(17))
        wrench = self.observations.wrist_wrench(env)
        torch.testing.assert_close(wrench[0], torch.zeros(6))
        torch.testing.assert_close(wrench[1], torch.tensor((1.0, 1.0, -0.25, 1.0, 1.0, -0.5)))
        del env.episode_length_buf
        torch.testing.assert_close(self.observations.surface_header_and_tactile(env), torch.zeros(2, 18))

    def test_observation_layout_55_and_last_input_uses_sanitized_terms(self):
        env = _environment()
        names = (
            "arm_joint_position", "arm_joint_velocity", "eef_relative_position", "eef_relative_orientation",
            "hand_state", "surface_header_and_tactile", "wrist_wrench", "initial_target_relative_position", "last_action",
        )
        vector = torch.cat([getattr(self.observations, name)(env) for name in names], dim=-1)
        self.assertEqual(tuple(vector.shape), (2, 55))
        torch.testing.assert_close(vector[:, 18:20], torch.full((2, 2), 0.5))
        env.action_manager.action[:] = float("nan")
        env.action_manager.get_term("arm_action").raw_actions[:] = 0.7
        env.action_manager.get_term("hand_action").raw_actions[:] = 0.8
        env.action_manager.get_term("hand_action").processed_actions[:] = 0.1
        last = self.observations.last_action(env)
        torch.testing.assert_close(last[:, :6], torch.full((2, 6), 0.7))
        torch.testing.assert_close(last[:, 6:], torch.full((2, 2), 0.8))
        env.episode_length_buf[0] = 0
        # The inverse hand-synergy mapping can leave float roundoff after
        # reset, but a reset row has no previous action at all.
        env.action_manager.get_term("arm_action").raw_actions[0] = 0.0
        env.action_manager.get_term("hand_action").raw_actions[0] = 1.19e-7
        partial_reset = self.observations.last_action(env)
        self.assertTrue(torch.equal(partial_reset[0], torch.zeros(8)))
        torch.testing.assert_close(partial_reset[1], last[1], atol=0, rtol=0)

    def test_distance_reward_uses_full_3d_distance(self):
        env = _environment(4)
        env.eef_position[:] = torch.tensor(((0, 0, 0), (0.05, 0, 0), (0, 0, 0.05), (0.1, 0, 0)))
        reward = self.rewards.eef_target_distance_reward(env)
        torch.testing.assert_close(reward, torch.tensor((1.0, math.exp(-1), math.exp(-1), math.exp(-2))))

    def test_action_rate_executed_values_first_step_cache_and_partial_reset(self):
        env = _environment()
        rate = self.rewards.ExecutedActionRate(SimpleNamespace(), env)
        arm = env.action_manager.get_term("arm_action")
        hand = env.action_manager.get_term("hand_action")
        arm.raw_actions[:] = 0.2
        hand.processed_actions[:] = 0.3
        torch.testing.assert_close(rate(env), torch.zeros(2))
        arm.raw_actions[:] = 0.8
        hand.raw_actions[:] = 1.0
        hand.processed_actions[:] = 0.5
        env._sim_step_counter += 2
        expected = torch.full((2,), -(0.6**2 + 0.2**2))
        torch.testing.assert_close(rate(env), expected)
        torch.testing.assert_close(rate(env), expected)  # repeated diagnostics do not consume history
        rate.reset([0])
        env._sim_step_counter += 2
        arm.raw_actions[:] = 0.2
        hand.processed_actions[:] = 0.3
        torch.testing.assert_close(rate(env), torch.tensor((0.0, -(0.6**2 + 0.2**2))))
        env._sim_step_counter += 2
        torch.testing.assert_close(rate(env), torch.zeros(2))

    def test_target_sampling_is_episode_fixed_and_partial_reset_isolated(self):
        env = _environment()
        env.scene.env_origins[1, 0] = 5
        cfg = SimpleNamespace(
            target_position_range_low=(-0.77, -0.22, 1.08), target_position_range_high=(-0.58, 0.22, 1.08),
            resampling_time_range=(1e6, 1e6), debug_vis=False, asset_name="robot",
        )
        command = self.commands.EpisodeTargetPositionCommand(cfg, env)
        command.reset()
        before = command.target_pos_w.clone()
        local = before - env.scene.env_origins
        self.assertTrue(torch.all(local >= torch.tensor(cfg.target_position_range_low)))
        self.assertTrue(torch.all(local <= torch.tensor(cfg.target_position_range_high)))
        env.eef_position[:] = 0.6
        command.compute(0.02)
        torch.testing.assert_close(command.target_pos_w, before)
        torch.testing.assert_close(command.initial_eef_pos_w, torch.zeros(2, 3))
        command.reset([1])
        torch.testing.assert_close(command.target_pos_w[0], before[0])
        torch.testing.assert_close(command.initial_eef_pos_w[0], torch.zeros(3))
        torch.testing.assert_close(command.initial_eef_pos_w[1], torch.full((3,), 0.6))
        env.eef_position[:] = command.target_pos_w
        command._update_metrics()
        torch.testing.assert_close(command.metrics["distance_m"], torch.zeros(2))
        torch.testing.assert_close(command.metrics["reached"], torch.ones(2))

    def test_reset_writes_default_physical_state_and_clears_selected_targets(self):
        env = _environment()
        env.scene.env_origins[1, 0] = 5
        default_root = torch.zeros(2, 13)
        default_root[:, 3] = 1
        default_root[:, 2] = 0.79505
        default_joints = torch.arange(18).float().repeat(2, 1)
        calls = {}
        robot = env.scene["robot"]
        robot.data.default_root_state = default_root
        robot.data.default_joint_pos = default_joints
        for name in (
            "write_root_pose_to_sim", "write_root_velocity_to_sim", "set_joint_position_target",
            "set_joint_velocity_target", "set_joint_effort_target",
        ):
            setattr(robot, name, lambda value, env_ids, key=name: calls.update({key: (value.clone(), env_ids.clone())}))
        robot.write_joint_state_to_sim = lambda pos, vel, env_ids: calls.update(
            {"joints": (pos.clone(), vel.clone(), env_ids.clone())}
        )
        self.events.reset_robot_to_initial_state(env, [1])
        torch.testing.assert_close(calls["write_root_pose_to_sim"][0][0, :3], torch.tensor((5.0, 0.0, 0.79505)))
        torch.testing.assert_close(calls["joints"][0], default_joints[1:2])
        torch.testing.assert_close(calls["joints"][1], torch.zeros(1, 18))
        torch.testing.assert_close(calls["set_joint_position_target"][0], default_joints[1:2])
        torch.testing.assert_close(calls["set_joint_effort_target"][0], torch.zeros(1, 18))
        torch.testing.assert_close(calls["joints"][2], torch.tensor((1,)))


if __name__ == "__main__":
    unittest.main()
