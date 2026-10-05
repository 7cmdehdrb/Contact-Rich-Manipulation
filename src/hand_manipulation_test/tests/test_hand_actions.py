"""Exercise hand range, reset and executed-action contracts without Kit."""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import pytest
import torch


PACKAGE_ROOT = Path(__file__).resolve().parents[1] / "hand_manipulation_test"
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
TEST_PACKAGE = "_hand_manipulation_action_fixture"


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


class _ActionTerm(_ManagerTerm):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._asset = env.scene[cfg.asset_name]


class _Articulation:
    def __init__(self, math, synergy):
        limits = torch.tensor((0.0, 2.0)).expand(len(synergy), 12, 2).clone()
        self.data = SimpleNamespace(
            joint_pos=math.inspire_synergy_to_joint_positions(synergy),
            default_joint_pos=math.inspire_synergy_to_joint_positions(synergy),
            joint_pos_limits=limits.clone(), soft_joint_pos_limits=limits,
        )
        self.joint_names = list(math.INSPIRE_HAND_JOINT_NAMES)
        self.target_writes = []
        self.limit_writes = []

    def find_joints(self, names, preserve_order):
        assert preserve_order
        return [self.joint_names.index(name) for name in names], list(names)

    def set_joint_position_target(self, targets, joint_ids, env_ids=None):
        self.target_writes.append((targets.clone(), tuple(joint_ids), None if env_ids is None else env_ids.clone()))

    def write_joint_position_limit_to_sim(self, limits, joint_ids, warn_limit_violation):
        self.limit_writes.append((limits.clone(), tuple(joint_ids), warn_limit_violation))
        self.data.joint_pos_limits[:, joint_ids] = limits
        self.data.default_joint_pos[:, joint_ids] = self.data.default_joint_pos[:, joint_ids].clamp(
            min=limits[..., 0], max=limits[..., 1]
        )
        # The real API rebuilds soft limits about the newly installed window.
        # Use a factor below one to catch accidental double narrowing of the
        # configured open endpoint in the action term.
        midpoint = limits.mean(dim=-1)
        half_range = 0.45 * (limits[..., 1] - limits[..., 0])
        self.data.soft_joint_pos_limits[:, joint_ids, 0] = midpoint - half_range
        self.data.soft_joint_pos_limits[:, joint_ids, 1] = midpoint + half_range


def _load(name, relative_path):
    spec = importlib.util.spec_from_file_location(name, PACKAGE_ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def backend():
    # Isolated module names keep these small simulator adapters local to this
    # fixture and avoid changing the runtime package registration.
    package = ModuleType(TEST_PACKAGE)
    package.__path__ = [str(PACKAGE_ROOT)]
    mdp = ModuleType(f"{TEST_PACKAGE}.mdp")
    mdp.__path__ = [str(PACKAGE_ROOT / "mdp")]
    sys.modules[package.__name__] = package
    sys.modules[mdp.__name__] = mdp
    math = _load(f"{TEST_PACKAGE}.action_math", "action_math.py")
    geometry = ModuleType(f"{TEST_PACKAGE}.geometry")
    geometry.control_point_pose_w = lambda env: (env.eef_position, env.eef_quaternion)
    sys.modules[geometry.__name__] = geometry
    assets = ModuleType("isaaclab.assets")
    assets.Articulation = _Articulation
    controllers = ModuleType("isaaclab.controllers")
    controllers.OperationalSpaceController = type("OperationalSpaceController", (), {})
    controllers.OperationalSpaceControllerCfg = type("OperationalSpaceControllerCfg", (), {})
    managers = ModuleType("isaaclab.managers")
    managers.ActionTerm = _ActionTerm
    managers.ActionTermCfg = type("ActionTermCfg", (), {})
    managers.ManagerTermBase = _ManagerTerm
    utils = ModuleType("isaaclab.utils")
    utils.configclass = lambda cls: cls
    with patch.dict(sys.modules, {
        "isaaclab": ModuleType("isaaclab"), assets.__name__: assets,
        controllers.__name__: controllers, managers.__name__: managers,
        utils.__name__: utils,
    }):
        actions = _load(f"{TEST_PACKAGE}.mdp.actions", "mdp/actions.py")
        rewards = _load(f"{TEST_PACKAGE}.mdp.rewards", "mdp/rewards.py")
    return math, actions, rewards


def _environment(
    backend, initial_synergy, *, interval=None, synergy_rate=100.0,
    joint_rate=100.0, enforce=False, original_hard_limits=None,
):
    math, actions, _ = backend
    robot = _Articulation(math, torch.tensor(initial_synergy, dtype=torch.float32))
    if original_hard_limits is not None:
        robot.data.joint_pos_limits[:] = original_hard_limits
    defaults = actions.InspireHandSynergyActionCfg
    cfg = SimpleNamespace(
        asset_name="robot", joint_names=defaults.joint_names,
        open_positions=defaults.open_positions, closed_positions=defaults.closed_positions,
        synergy_range=defaults.synergy_range if interval is None else interval,
        enforce_synergy_joint_limits=enforce,
        max_synergy_rate=(synergy_rate, synergy_rate), max_joint_target_rate=(joint_rate,) * 12,
        saturation_tolerance=defaults.saturation_tolerance,
    )
    env = SimpleNamespace(num_envs=len(initial_synergy), device="cpu", step_dt=0.02, scene={"robot": robot})
    hand = actions.InspireHandSynergyAction(cfg, env)
    arm = SimpleNamespace(raw_actions=torch.zeros(env.num_envs, 6))
    env.action_manager = SimpleNamespace(get_term={"arm_action": arm, "hand_action": hand}.__getitem__)
    env._sim_step_counter = 0
    return env, hand, robot


def test_push_endpoints_control_both_physical_synergies_and_one_means_open(backend):
    math, _, _ = backend
    _, hand, robot = _environment(backend, [[0.9, 0.9]] * 3, interval=(0.8, 1.0))
    actions = torch.tensor(((-1.0, 1.0), (0.0, 0.0), (1.0, -1.0)))
    hand.process_actions(actions)
    hand.apply_actions()
    expected = torch.tensor(((0.8, 1.0), (0.9, 0.9), (1.0, 0.8)))
    torch.testing.assert_close(hand.synergy_target, expected)
    torch.testing.assert_close(math.inspire_joint_positions_to_synergy(robot.target_writes[-1][0]), expected)
    torch.testing.assert_close(hand.processed_actions, actions, atol=2e-6, rtol=1e-5)
    # Physical extension: the +1 common-flexion coordinate drives the four
    # finger masters to zero bend, while +1 thumb_1 reaches its 0.2rad open stop.
    torch.testing.assert_close(hand.joint_targets[2, [4, 6, 8, 10]], torch.zeros(4))
    assert hand.joint_targets[0, 0].item() == pytest.approx(0.2)


def test_reach_and_contact_keep_original_full_range_action_mapping(backend):
    math, _, _ = backend
    _, hand, _ = _environment(backend, [[0.5, 0.5]] * 3)
    actions = torch.tensor(((-1.0, 1.0), (0.0, 0.0), (1.0, -1.0)))
    hand.process_actions(actions)
    expected = torch.tensor(((0.0, 1.0), (0.5, 0.5), (1.0, 0.0)))
    torch.testing.assert_close(hand.synergy_target, expected)
    torch.testing.assert_close(math.inspire_joint_positions_to_synergy(hand.joint_targets), expected)
    torch.testing.assert_close(hand.processed_actions, actions, atol=2e-6, rtol=1e-5)


def test_hold_action_inverts_restricted_range_without_changing_physical_state_observation(backend):
    _, hand, _ = _environment(backend, [[0.9, 0.98], [0.95, 0.91]], interval=(0.8, 1.0))
    initial_targets = hand.joint_targets.clone()
    actual = hand.actual_synergy.clone()
    hold = hand.synergy_to_action(actual)
    torch.testing.assert_close(hold, torch.tensor(((0.0, 0.8), (0.5, 0.1))), atol=2e-6, rtol=1e-5)
    hand.process_actions(hold)
    torch.testing.assert_close(hand.joint_targets, initial_targets, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(hand.actual_synergy, actual)
    torch.testing.assert_close(hand.action_to_synergy(hold), actual)


def test_joint_rate_limit_and_action_rate_use_executed_restricted_range_history(backend):
    math, _, rewards = backend
    env, hand, _ = _environment(backend, [[0.9, 0.9]], interval=(0.8, 1.0), synergy_rate=2.0, joint_rate=1.0)
    reward = rewards.ExecutedActionRate(SimpleNamespace(), env)
    assert reward(env).item() == 0.0
    before = hand.joint_targets.clone()
    hand.process_actions(torch.ones(1, 2))
    executed_synergy = math.inspire_joint_positions_to_synergy(hand.joint_targets)
    assert torch.all((hand.joint_targets - before).abs() <= 0.020001)
    assert torch.all((hand.synergy_target - 0.9).abs() <= 0.040001)
    assert torch.all(hand.synergy_target >= 0.8)
    torch.testing.assert_close(hand.processed_actions, (executed_synergy - 0.8) * 10.0 - 1.0)
    assert torch.all(hand.processed_actions < 1.0)
    env._sim_step_counter += 1
    expected_reward = -hand.processed_actions.square().mean().item()
    assert reward(env).item() == pytest.approx(expected_reward, abs=1e-6)


def test_partial_reset_projects_invalid_targets_and_preserves_other_environments(backend):
    math, _, _ = backend
    _, hand, robot = _environment(backend, [[0.9, 0.98], [0.95, 0.91], [0.98, 0.98]], interval=(0.8, 1.0))
    hand.process_actions(torch.tensor(((0.4, -0.2), (-1.0, 1.0), (0.6, 0.8))))
    other_rows = torch.tensor((0, 2))
    saved = [value[other_rows].clone() for value in (hand.raw_actions, hand.processed_actions, hand.joint_targets)]
    robot.data.joint_pos[1] = math.inspire_synergy_to_joint_positions(torch.tensor((0.7, 1.0)))
    hand.reset_to_current(torch.tensor((1,)))
    for value, expected in zip((hand.raw_actions, hand.processed_actions, hand.joint_targets), saved):
        torch.testing.assert_close(value[other_rows], expected)
    torch.testing.assert_close(hand.synergy_target[1], torch.tensor((0.8, 1.0)))
    torch.testing.assert_close(hand.raw_actions[1], torch.tensor((-1.0, 1.0)))
    torch.testing.assert_close(hand.processed_actions[1], hand.raw_actions[1])
    torch.testing.assert_close(hand.actual_synergy[1], torch.tensor((0.7, 1.0)))
    torch.testing.assert_close(robot.target_writes[-1][2], torch.tensor((1,)))
    assert not hand.action_saturated[1]


def test_out_of_range_startup_state_is_projected_before_first_drive_command(backend):
    math, _, _ = backend
    _, hand, robot = _environment(backend, [[0.5, 0.5]], interval=(0.8, 1.0))
    hand.apply_actions()
    torch.testing.assert_close(hand.actual_synergy, torch.tensor(((0.5, 0.5),)))
    torch.testing.assert_close(math.inspire_joint_positions_to_synergy(robot.target_writes[-1][0]), torch.tensor(((0.8, 0.8),)))


def test_physical_hand_limits_cover_all_twelve_joints_once_and_preserve_full_open_endpoint(backend):
    math, _, _ = backend
    _, hand, robot = _environment(backend, [[0.5, 0.5]] * 2, interval=(0.8, 1.0), enforce=True)
    assert len(robot.limit_writes) == 1
    limits, joint_ids, warn = robot.limit_writes[0]
    assert joint_ids == tuple(range(12))
    assert warn is False
    open_positions = torch.tensor(math.INSPIRE_OPEN_JOINT_POSITIONS)
    lower_open_positions = math.inspire_synergy_to_joint_positions(torch.tensor((0.8, 0.8)))
    expected_limits = torch.stack((open_positions, lower_open_positions), dim=-1).expand(2, -1, -1)
    torch.testing.assert_close(limits, expected_limits)
    # The real API also projects default positions. Physical measured positions
    # are unchanged until reset/physics, and observations report them honestly.
    assert torch.all(robot.data.default_joint_pos >= limits[..., 0])
    assert torch.all(robot.data.default_joint_pos <= limits[..., 1])
    torch.testing.assert_close(hand.actual_synergy, torch.full((2, 2), 0.5))
    hand.process_actions(torch.ones(2, 2))
    hand.apply_actions()
    torch.testing.assert_close(hand.joint_targets, open_positions.expand(2, -1))
    torch.testing.assert_close(hand.synergy_target, torch.ones(2, 2))
    assert torch.all(robot.data.soft_joint_pos_limits[..., 0] > hand.joint_targets)
    hand.reset(torch.tensor((1,)))
    hand.process_actions(-torch.ones(2, 2))
    assert len(robot.limit_writes) == 1


@pytest.mark.parametrize("initial", [0.5, 0.95])
def test_default_reach_and_contact_do_not_write_physical_joint_limits(backend, initial):
    _, actions, _ = backend
    assert actions.InspireHandSynergyActionCfg.enforce_synergy_joint_limits is False
    _, hand, robot = _environment(backend, [[initial, initial]])
    hand.process_actions(torch.ones(1, 2))
    hand.reset()
    assert robot.limit_writes == []


def test_physical_window_intersects_existing_asset_constraints_without_extending_them(backend):
    _, _, _ = backend
    original = torch.tensor((0.0, 2.0)).expand(1, 12, 2).clone()
    original[:, 0] = torch.tensor((0.3, 0.3))
    _, hand, robot = _environment(
        backend, [[0.9, 0.9]], interval=(0.8, 1.0), enforce=True,
        original_hard_limits=original,
    )
    limits = robot.limit_writes[0][0]
    torch.testing.assert_close(limits[:, 0], original[:, 0])
    assert torch.all(limits[..., 0] >= original[..., 0])
    assert torch.all(limits[..., 1] <= original[..., 1])
    hand.process_actions(torch.ones(1, 2))
    assert hand.joint_targets[0, 0].item() == pytest.approx(0.3)


def test_incompatible_asset_limits_fail_before_installing_a_physical_window(backend):
    original = torch.tensor((0.0, 2.0)).expand(1, 12, 2).clone()
    original[:, 0, 0] = 0.5
    with pytest.raises(ValueError, match="existing physical hand joint limits"):
        _environment(
            backend, [[0.9, 0.9]], interval=(0.8, 1.0), enforce=True,
            original_hard_limits=original,
        )


@pytest.mark.parametrize("interval", [(0.8, 0.8), (1.0, 0.8), (-0.1, 1.0), (0.0, 1.1), (float("nan"), 1.0), (0.8, 1.0, 1.0)])
def test_invalid_hand_ranges_are_rejected_before_control(backend, interval):
    with pytest.raises(ValueError, match="synergy_range"):
        _environment(backend, [[0.9, 0.9]], interval=interval)


def _class_assignments(path, class_name):
    tree = ast.parse(path.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
    return {node.targets[0].id: node.value for node in cls.body if isinstance(node, ast.Assign)}


def test_push_ppo_matches_sweep_reference_through_rsl5_model_api():
    reference = _class_assignments(
        REPOSITORY_ROOT / "example/Sweep-Policy/sweeping_policy/config/ur5e/agents/rsl_rl_ppo_cfg_02.py",
        "UR5eSweepPPORunnerCfg",
    )
    push = _class_assignments(PACKAGE_ROOT / "agents/rsl_rl_push_ppo_cfg.py", "PushPPORunnerCfg")
    for key in ("num_steps_per_env", "max_iterations", "save_interval", "run_name"):
        assert ast.literal_eval(push[key]) == ast.literal_eval(reference[key])
    old_policy = {keyword.arg: ast.literal_eval(keyword.value) for keyword in reference["policy"].keywords}
    actor = {keyword.arg: keyword.value for keyword in push["actor"].keywords}
    critic = {keyword.arg: keyword.value for keyword in push["critic"].keywords}
    assert ast.literal_eval(actor["hidden_dims"]) == old_policy["actor_hidden_dims"]
    assert ast.literal_eval(critic["hidden_dims"]) == old_policy["critic_hidden_dims"]
    assert ast.literal_eval(actor["activation"]) == ast.literal_eval(critic["activation"]) == old_policy["activation"]
    assert ast.literal_eval(actor["obs_normalization"]) == old_policy["actor_obs_normalization"]
    assert ast.literal_eval(critic["obs_normalization"]) == old_policy["critic_obs_normalization"]
    distribution = actor["distribution_cfg"]
    assert ast.unparse(distribution.func) == "RslRlMLPModelCfg.GaussianDistributionCfg"
    assert {keyword.arg: ast.literal_eval(keyword.value) for keyword in distribution.keywords} == {
        "init_std": old_policy["init_noise_std"]
    }
    algorithm = lambda node: {keyword.arg: ast.literal_eval(keyword.value) for keyword in node.keywords}
    assert algorithm(push["algorithm"]) == algorithm(reference["algorithm"])
