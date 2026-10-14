"""Real Torch regressions for Cube snapshots, palm contact, and table failure."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import torch


PACKAGE_ROOT = Path(__file__).resolve().parents[1] / "hand_manipulation_test"
TEST_PACKAGE = "_hand_manipulation_test_contact_fixture"


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
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.metrics = {}
        self.time_left = torch.zeros(self.num_envs)
        self._set_debug_vis_impl(cfg.debug_vis)

    def reset(self, env_ids=None):
        index = slice(None) if env_ids is None else env_ids
        self.time_left[index] = self.cfg.resampling_time_range[0]
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


def _load_terms():
    package = ModuleType(TEST_PACKAGE)
    package.__path__ = [str(PACKAGE_ROOT)]
    mdp = ModuleType(f"{TEST_PACKAGE}.mdp")
    mdp.__path__ = [str(PACKAGE_ROOT / "mdp")]
    sys.modules[TEST_PACKAGE] = package
    sys.modules[mdp.__name__] = mdp
    geometry = ModuleType(f"{TEST_PACKAGE}.geometry")
    geometry.control_point_pose_w = lambda env: (env.eef_position, env.eef_quaternion)
    sys.modules[geometry.__name__] = geometry
    sensors = _load(f"{TEST_PACKAGE}.contact_sensors", "contact_sensors.py")
    managers = ModuleType("isaaclab.managers")
    managers.CommandTerm = _CommandTerm
    managers.CommandTermCfg = type("CommandTermCfg", (), {})
    managers.ManagerTermBase = _ManagerTerm
    utils = ModuleType("isaaclab.utils")
    utils.configclass = lambda cls: cls
    env_mdp = ModuleType("isaaclab.envs.mdp")
    env_mdp.time_out = lambda env: env.episode_length_buf >= env.max_episode_length
    replacements = {
        "isaaclab": ModuleType("isaaclab"), "isaaclab.envs": ModuleType("isaaclab.envs"),
        managers.__name__: managers, utils.__name__: utils, env_mdp.__name__: env_mdp,
    }
    with patch.dict(sys.modules, replacements):
        _load(f"{TEST_PACKAGE}.mdp.commands", "mdp/commands.py")
        commands = _load(f"{TEST_PACKAGE}.mdp.contact_commands", "mdp/contact_commands.py")
        rewards = _load(f"{TEST_PACKAGE}.mdp.contact_rewards", "mdp/contact_rewards.py")
        base_rewards = _load(f"{TEST_PACKAGE}.mdp.rewards", "mdp/rewards.py")
        terminations = _load(f"{TEST_PACKAGE}.mdp.contact_terminations", "mdp/contact_terminations.py")
    return sensors, commands, rewards, base_rewards, terminations


class _Scene(dict):
    pass


def _environment(count=3):
    scene = _Scene()
    scene.env_origins = torch.zeros(count, 3)
    scene["target_object"] = SimpleNamespace(data=SimpleNamespace(root_pos_w=torch.zeros(count, 3)))
    scene["table_contacts"] = SimpleNamespace(data=SimpleNamespace(
        force_matrix_w_history=torch.zeros(count, 2, 1, 38, 3),
        net_forces_w=torch.full((count, 1, 3), 10.0),
    ))
    scene["cube_palm_contacts"] = SimpleNamespace(data=SimpleNamespace(force_matrix_w=torch.zeros(count, 1, 17, 3)))
    arm = SimpleNamespace(raw_actions=torch.zeros(count, 6))
    hand = SimpleNamespace(processed_actions=torch.zeros(count, 2))
    terms = {"arm_action": arm, "hand_action": hand}
    return SimpleNamespace(
        num_envs=count, device="cpu", scene=scene,
        cfg=SimpleNamespace(task=SimpleNamespace(contact_threshold_n=0.01, goal_reward_sigma_m=0.05, success_distance_m=0.01),
                            decimation=2),
        eef_position=torch.zeros(count, 3),
        eef_quaternion=torch.tensor((1.0, 0.0, 0.0, 0.0)).expand(count, -1).clone(),
        episode_length_buf=torch.ones(count, dtype=torch.long), max_episode_length=500, _sim_step_counter=0,
        action_manager=SimpleNamespace(get_term=terms.__getitem__),
    )


class ContactTermsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sensors, cls.commands, cls.rewards, cls.base_rewards, cls.terminations = _load_terms()

    def test_cube_contact_threshold_is_inclusive_and_uses_vector_norm(self):
        matrix = torch.zeros(4, 1, 17, 3)
        matrix[0, 0, 0, 0] = 0.009
        matrix[1, 0, 0, 0] = 0.01
        matrix[2, 0, 16, 0] = -0.011
        matrix[3, 0, 8, :2] = torch.tensor((0.008, 0.008))
        self.assertEqual(self.sensors.cube_palm_contact_from_forces(matrix, 0.01).tolist(), [False, True, True, True])

    def test_table_history_catches_transient_and_opposing_contacts_without_summing(self):
        history = torch.zeros(3, 2, 1, 38, 3)
        history[0, 1, 0, 37, 0] = 0.01  # last substep ended with no contact
        history[1, 0, 0, :2, 0] = torch.tensor((0.01, -0.01))
        history[2, :, 0, :, 0] = 0.009  # many small pairs must remain subthreshold
        self.assertEqual(self.sensors.table_contact_from_history(history, 0.01).tolist(), [True, True, False])

    def test_table_support_load_cannot_count_as_robot_contact(self):
        env = _environment()
        # net_forces_w includes the supported Cube and is deliberately large.
        # The robot-filtered pair history is empty and must be authoritative.
        self.assertEqual(self.sensors.table_contact_mask(env).tolist(), [False, False, False])
        env.scene["table_contacts"].data.force_matrix_w_history[1, 0, 0, 4, 2] = 0.01
        self.assertEqual(self.terminations.table_contact_failure(env).tolist(), [False, True, False])

    def test_reward_is_continuous_real_pad_contact_and_masks_reset_and_table_failure(self):
        env = _environment()
        env.scene["cube_palm_contacts"].data.force_matrix_w[:, 0, 0, 0] = 0.01
        env.episode_length_buf[0] = 0
        env.scene["table_contacts"].data.force_matrix_w_history[2, 1, 0, 5, 0] = 0.01
        expected = torch.tensor((0.0, 1.0, 0.0))
        torch.testing.assert_close(self.rewards.palm_cube_contact_reward(env), expected)
        env._sim_step_counter += 2
        torch.testing.assert_close(self.rewards.palm_cube_contact_reward(env), expected)
        env.scene["cube_palm_contacts"].data.force_matrix_w[1] = 0.0
        # Broad hand/carrier or unfiltered palm signals cannot earn this term.
        env.unfiltered_palm_bits = torch.ones(3, 17)
        env.carrier_cube_force = torch.full((3,), 5.0)
        torch.testing.assert_close(self.rewards.palm_cube_contact_reward(env), torch.zeros(3))
        env.scene["cube_palm_contacts"].data.force_matrix_w[1, 0, 16, 1] = 0.01
        torch.testing.assert_close(self.rewards.palm_cube_contact_reward(env), expected)
        del env.episode_length_buf
        torch.testing.assert_close(self.rewards.palm_cube_contact_reward(env), torch.zeros(3))

    def test_table_termination_has_precedence_over_exact_timeout_boundary(self):
        env = _environment()
        env.episode_length_buf[:] = torch.tensor((500, 500, 499))
        env.scene["table_contacts"].data.force_matrix_w_history[0, 1, 0, 37, 1] = 0.01
        self.assertEqual(self.terminations.table_contact_failure(env).tolist(), [True, False, False])
        self.assertEqual(self.terminations.contact_time_out(env).tolist(), [False, True, False])

    def test_missing_history_bad_layout_and_invalid_threshold_fail_clearly(self):
        env = _environment()
        env.scene["table_contacts"].data.force_matrix_w_history = None
        with self.assertRaises(RuntimeError):
            self.sensors.table_contact_mask(env)
        env.scene["table_contacts"].data.force_matrix_w_history = torch.zeros(3, 1, 1, 38, 3)
        with self.assertRaises(RuntimeError):
            self.sensors.table_contact_mask(env)
        with self.assertRaises(ValueError):
            self.sensors.cube_palm_contact_from_forces(torch.zeros(3, 1, 30, 3), 0.01)
        with self.assertRaises(ValueError):
            self.sensors.table_contact_from_history(torch.zeros(3, 2, 2, 38, 3), 0.01)
        for threshold in (0.0, -0.01, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                self.sensors.cube_palm_contact_from_forces(torch.zeros(3, 1, 17, 3), threshold)

    def test_cube_command_snapshots_actual_world_pose_without_target_sampling(self):
        env = _environment()
        env.scene.env_origins[:] = torch.tensor(((0.0, 0.0, 0.0), (5.0, 0.0, 0.0), (-5.0, 0.0, 0.0)))
        cube = env.scene["target_object"].data.root_pos_w
        cube[:] = torch.tensor(((-0.6, 0.1, 1.08), (4.4, -0.2, 1.08), (-5.7, 0.0, 1.08)))
        cfg = SimpleNamespace(
            target_position_range_low=(-0.77, -0.22, 1.08), target_position_range_high=(-0.58, 0.22, 1.08),
            resampling_time_range=(1e6, 1e6), debug_vis=False, asset_name="robot", object_name="target_object",
        )
        command = self.commands.CubeInitialPositionCommand(cfg, env)
        rng_before = torch.random.get_rng_state().clone()
        command.reset()
        self.assertTrue(torch.equal(torch.random.get_rng_state(), rng_before))
        torch.testing.assert_close(command.target_pos_w, cube, atol=0, rtol=0)
        snapshot = command.target_pos_w.clone()
        cube[:, 0] += 0.3
        env.eef_position[:] = 0.2
        command.compute(0.02)
        torch.testing.assert_close(command.target_pos_w, snapshot, atol=0, rtol=0)
        torch.testing.assert_close(command.initial_eef_pos_w, torch.zeros(3, 3))
        command.reset([1])
        torch.testing.assert_close(command.target_pos_w[1], cube[1], atol=0, rtol=0)
        torch.testing.assert_close(command.target_pos_w[[0, 2]], snapshot[[0, 2]], atol=0, rtol=0)
        torch.testing.assert_close(command.initial_eef_pos_w[1], env.eef_position[1], atol=0, rtol=0)
        torch.testing.assert_close(command.initial_eef_pos_w[[0, 2]], torch.zeros(2, 3))
        # Public manager adapters accept Sequence[int], including tuples.
        cube[2] = torch.tensor((-5.6, 0.1, 1.08))
        command.reset((2,))
        torch.testing.assert_close(command.target_pos_w[2], cube[2], atol=0, rtol=0)
        torch.testing.assert_close(command.target_pos_w[0], snapshot[0], atol=0, rtol=0)

    def test_inherited_distance_reward_targets_initial_cube_after_cube_motion(self):
        env = _environment()
        initial_target = torch.zeros(3, 3)
        env.command_manager = SimpleNamespace(get_term=lambda name: SimpleNamespace(target_pos_w=initial_target))
        env.eef_position[:, 0] = torch.tensor((0.1, 0.05, 0.0))
        before = self.base_rewards.eef_target_distance_reward(env)
        env.scene["target_object"].data.root_pos_w[:] = 2.0
        torch.testing.assert_close(self.base_rewards.eef_target_distance_reward(env), before, atol=0, rtol=0)
        self.assertTrue(before[0] < before[1] < before[2])

    def test_tiny_action_changes_and_full_sign_flip_have_bounded_small_cost(self):
        env = _environment()
        term = self.base_rewards.ExecutedActionRate(SimpleNamespace(), env)
        arm = env.action_manager.get_term("arm_action")
        hand = env.action_manager.get_term("hand_action")
        torch.testing.assert_close(term(env), torch.zeros(3))
        arm.raw_actions[:] = 0.1
        hand.processed_actions[:] = 0.1
        env._sim_step_counter += 2
        torch.testing.assert_close(0.001 * term(env), torch.full((3,), -0.00002))
        arm.raw_actions[:] = -1.0
        hand.processed_actions[:] = -1.0
        env._sim_step_counter += 2
        term(env)
        arm.raw_actions[:] = 1.0
        hand.processed_actions[:] = 1.0
        env._sim_step_counter += 2
        torch.testing.assert_close(0.001 * term(env), torch.full((3,), -0.008))


if __name__ == "__main__":
    unittest.main()
