"""Bounded accumulated OSC targets through production action methods, without Isaac Sim."""

import importlib.util
import math
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from hand_manipulation_test import action_math
from hand_manipulation_test.accumulation_math import accumulated_translation_target, translation_target_error_c


ROOT = Path(__file__).parents[1] / "hand_manipulation_test"
PACKAGE = "_push_v1_action_fixture"


class _ActionTerm:
    def __init__(self, cfg, env):
        self.cfg, self._env, self._asset = cfg, env, env.scene[cfg.asset_name]

    @property
    def num_envs(self):
        return self._env.num_envs

    @property
    def device(self):
        return self._env.device


class _Robot:
    is_fixed_base = True

    def __init__(self, count):
        identity = torch.tensor([1., 0., 0., 0.]).expand(count, -1).clone()
        self.data = SimpleNamespace(
            root_link_pos_w=torch.zeros(count, 3), root_link_quat_w=identity,
            root_link_vel_w=torch.zeros(count, 6),
            body_link_pos_w=torch.zeros(count, 2, 3),
            body_link_quat_w=identity[:, None].expand(-1, 2, -1).clone(),
            body_link_vel_w=torch.zeros(count, 2, 6), body_com_pos_b=torch.zeros(count, 2, 3),
            joint_pos=torch.zeros(count, 6), joint_vel=torch.zeros(count, 6),
            joint_effort_limits=torch.tensor([150., 150., 150., 28., 28., 28.]).expand(count, -1),
        )
        self.root_physx_view = SimpleNamespace(
            get_jacobians=lambda: torch.eye(6).expand(count, 1, -1, -1),
            get_generalized_mass_matrices=lambda: torch.eye(6).expand(count, -1, -1),
            get_gravity_compensation_forces=lambda: torch.zeros(count, 6),
        )
        self.effort_writes = []

    def find_joints(self, names, preserve_order):
        assert preserve_order
        return list(range(6)), names

    def find_bodies(self, name, preserve_order):
        return [1], [name]

    def set_joint_effort_target(self, values, joint_ids):
        self.effort_writes.append(values.clone())


class _OSC:
    def __init__(self, cfg, count, device):
        self.cfg = cfg
        self.commands = []
        self.computed_efforts = torch.zeros(count, 6, device=device)

    def set_command(self, values):
        self.commands.append(values.clone())

    def compute(self, **kwargs):
        return self.computed_efforts.clone()


def _load(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def backend():
    package, mdp = ModuleType(PACKAGE), ModuleType(PACKAGE + ".mdp")
    package.__path__, mdp.__path__ = [str(ROOT)], [str(ROOT / "mdp")]
    sys.modules[PACKAGE], sys.modules[mdp.__name__] = package, mdp
    sys.modules[PACKAGE + ".action_math"] = action_math
    assets, controllers, managers, utils = (ModuleType("isaaclab." + name) for name in (
        "assets", "controllers", "managers", "utils"
    ))
    assets.Articulation = _Robot
    controllers.OperationalSpaceController = _OSC
    controllers.OperationalSpaceControllerCfg = lambda **kwargs: SimpleNamespace(**kwargs)
    managers.ActionTerm, managers.ActionTermCfg = _ActionTerm, type("ActionTermCfg", (), {})
    utils.configclass = lambda cls: cls
    with patch.dict(sys.modules, {"isaaclab": ModuleType("isaaclab"), **{
        module.__name__: module for module in (assets, controllers, managers, utils)
    }}):
        base = _load(PACKAGE + ".mdp.actions", "mdp/actions.py")
        v1 = _load(PACKAGE + ".mdp.push_v1_actions", "mdp/push_v1_actions.py")
    return base, v1


def _fixture(backend, count=3, cap=.06):
    base, module = backend
    defaults = module.AccumulatedTranslationOscActionCfg
    values = {name: getattr(defaults, name) for name in (
        "joint_names", "body_name", "body_offset_quat", "motion_stiffness", "motion_damping_ratio",
        "gravity_compensation", "inertial_dynamics_decoupling", "partial_inertial_dynamics_decoupling",
        "effort_limits", "effort_limit_scale", "saturation_tolerance",
    )}
    values.update(asset_name="robot", body_offset_pos=(0., 0., 0.), translation_scale=(.012, .012, .008),
                  rotation_scale=(.06,) * 3, position_error_limit_m=cap)
    robot = _Robot(count)
    env = SimpleNamespace(num_envs=count, device="cpu", scene={"robot": robot})
    arm = module.AccumulatedTranslationOscAction(SimpleNamespace(**values), env)
    env.action_manager = SimpleNamespace(get_term=lambda name: arm)
    return env, arm, robot, module


def test_accumulation_reaches_norm_cap_and_reversal_releases_windup(backend):
    _, arm, _, _ = _fixture(backend)
    arm.reset()
    command = torch.tensor([[0., 1., 0., 0., 0., 0.]]).expand(3, -1)
    for _ in range(20):
        arm.process_actions(command)
    torch.testing.assert_close(arm.position_target_error_c_m, command[:, :3] * .06)
    arm.process_actions(-command)
    torch.testing.assert_close(arm.position_target_error_c_m[:, 1], torch.full((3,), .048))
    diagonal = torch.tensor([[1., 1., 1., 0., 0., 0.]]).expand(3, -1)
    for _ in range(10):
        arm.process_actions(diagonal)
    torch.testing.assert_close(arm.position_target_error_c_m.norm(dim=-1), torch.full((3,), .06))


def test_rotating_current_frame_changes_increment_and_observation_without_rotating_old_target(backend):
    _, arm, robot, _ = _fixture(backend)
    command = torch.zeros(3, 6)
    command[:, 0] = 1.
    arm.process_actions(command)
    robot.data.body_link_quat_w[:, 1] = action_math.quaternion_from_rotation_vector(
        torch.tensor([[0., 0., math.pi / 2.]]).expand(3, -1)
    )
    arm.process_actions(command)
    torch.testing.assert_close(arm.desired_c_pose_b[:, :3], torch.tensor([[.012, .012, 0.]]).expand(3, -1), atol=1e-7, rtol=1e-5)
    torch.testing.assert_close(arm.position_target_error_c_m, torch.tensor([[.012, -.012, 0.]]).expand(3, -1), atol=1e-7, rtol=1e-5)


def test_zero_action_holds_absolute_position_and_physics_substeps_never_integrate(backend):
    _, arm, robot, _ = _fixture(backend)
    command = torch.zeros(3, 6)
    command[:, 1] = 1.
    arm.process_actions(command)
    target = arm.desired_c_pose_b.clone()
    for position in (.003, .006):
        robot.data.body_link_pos_w[:, 1, 1] = position
        arm.apply_actions()
        torch.testing.assert_close(arm.desired_c_pose_b, target)
    arm.process_actions(torch.zeros_like(command))
    torch.testing.assert_close(arm.desired_c_pose_b[:, :3], target[:, :3])
    torch.testing.assert_close(arm.position_target_error_c_m[:, 1], torch.full((3,), .006))
    assert len(robot.effort_writes) == 2


def test_partial_reset_relatches_actual_pose_without_discarding_other_targets(backend):
    env, arm, robot, module = _fixture(backend)
    torch.testing.assert_close(module.accumulated_translation_error(env), torch.zeros(3, 3))
    arm.process_actions(torch.ones(3, 6))
    retained = arm.desired_c_pose_b[[0, 2]].clone()
    robot.data.body_link_pos_w[1, 1] = torch.tensor([.2, .3, .4])
    arm.reset(torch.tensor([1]))
    torch.testing.assert_close(arm.desired_c_pose_b[[0, 2]], retained)
    torch.testing.assert_close(arm.desired_c_pose_b[1, :3], robot.data.body_link_pos_w[1, 1])
    torch.testing.assert_close(module.accumulated_translation_error(env)[1], torch.zeros(3))
    torch.testing.assert_close(arm.raw_actions[1], torch.zeros(6))


def test_finite_sanitation_rotation_semantics_and_inherited_effort_limits(backend):
    base, _ = backend
    _, arm, _, _ = _fixture(backend)
    command = torch.tensor([[float("nan"), 3., float("inf"), .2, -.4, .5]]).expand(3, -1)
    arm.process_actions(command)
    assert bool(arm.invalid_action.all()) and bool(arm.torque_saturated.all())
    assert bool(torch.isfinite(arm.desired_c_pose_b).all())
    torch.testing.assert_close(arm.raw_actions[:, :3], torch.tensor([[0., 1., 0.]]).expand(3, -1))
    expected = action_math.quaternion_from_rotation_vector(command[:, 3:] * .06)
    torch.testing.assert_close(arm.desired_c_pose_b[:, 3:], expected)
    arm._osc.computed_efforts.fill_(1000.)
    arm.apply_actions()
    torch.testing.assert_close(arm.joint_efforts, torch.tensor([[135., 135., 135., 25.2, 25.2, 25.2]]).expand(3, -1))
    assert arm.apply_actions.__func__ is base.CurrentFrameOscAction.apply_actions
    assert arm._osc.cfg.motion_stiffness_task == (200.,) * 6


def test_selected_action_processing_changes_only_selected_environment(backend):
    _, arm, _, _ = _fixture(backend)
    arm.reset()
    previous = arm.desired_c_pose_b.clone()
    arm.process_actions_for_envs(torch.ones(1, 6), torch.tensor([2]))
    torch.testing.assert_close(arm.desired_c_pose_b[:2], previous[:2])
    torch.testing.assert_close(arm.raw_actions[:2], torch.zeros(2, 6))


def test_rotation_residual_holds_reset_anchor_under_body_perturbation(backend):
    _, arm, robot, _ = _fixture(backend)
    reference = action_math.quaternion_from_rotation_vector(
        torch.tensor([[.3, -.2, .7], [-.4, .5, .1], [.6, .1, -.2]])
    )
    robot.data.body_link_quat_w[:, 1] = reference
    arm.reset()
    command = torch.zeros(3, 6)
    command[:, 3:] = torch.tensor([.5, -.25, 1.])
    expected = action_math.quaternion_multiply(
        reference, action_math.quaternion_from_rotation_vector(command[:, 3:] * .06)
    )
    arm.process_actions(command)
    torch.testing.assert_close(arm.desired_c_pose_b[:, 3:], expected)
    robot.data.body_link_quat_w[:, 1] = action_math.quaternion_from_rotation_vector(
        torch.tensor([[1., -.3, -.4]]).expand(3, -1)
    )
    for _ in range(25):
        arm.process_actions(command)
        torch.testing.assert_close(arm.desired_c_pose_b[:, 3:], expected)
    # A zero residual restores the original reset orientation under load;
    # it neither follows the deflected body nor accumulates earlier rotation.
    arm.process_actions(torch.zeros_like(command))
    torch.testing.assert_close(arm.desired_c_pose_b[:, 3:], reference)
    for _ in range(2):
        arm.apply_actions()
        torch.testing.assert_close(arm.desired_c_pose_b[:, 3:], reference)


def test_partial_reset_changes_only_selected_orientation_reference(backend):
    _, arm, robot, _ = _fixture(backend)
    reference = action_math.quaternion_from_rotation_vector(
        torch.tensor([[.2, .4, -.1], [-.4, .1, .3], [.5, -.2, .1]])
    )
    robot.data.body_link_quat_w[:, 1] = reference
    arm.reset()
    command = torch.zeros(3, 6)
    command[:, 3:] = torch.tensor([1., .5, -.25])
    arm.process_actions(command)
    retained = arm.desired_c_pose_b[[0, 2]].clone()
    replacement = action_math.quaternion_from_rotation_vector(torch.tensor([[.9, -.3, .7]]))[0]
    robot.data.body_link_quat_w[:, 1] = replacement
    arm.reset(torch.tensor([1]))
    torch.testing.assert_close(arm.desired_c_pose_b[[0, 2]], retained)
    torch.testing.assert_close(arm.desired_c_pose_b[1, 3:], replacement)
    arm.process_actions_for_envs(command[1:2], torch.tensor([1]))
    expected = action_math.quaternion_multiply(
        replacement, action_math.quaternion_from_rotation_vector(command[1, 3:] * .06)
    )
    torch.testing.assert_close(arm.desired_c_pose_b[1, 3:], expected)
    torch.testing.assert_close(arm.desired_c_pose_b[[0, 2]], retained)
    arm.process_actions(torch.zeros_like(command))
    torch.testing.assert_close(arm.desired_c_pose_b[[0, 2], 3:], reference[[0, 2]])
    torch.testing.assert_close(arm.desired_c_pose_b[1, 3:], replacement)


def test_first_selected_command_latches_realized_orientation_once(backend):
    _, arm, robot, _ = _fixture(backend)
    reference = action_math.quaternion_from_rotation_vector(torch.tensor([[.7, .2, -.4]]))[0]
    robot.data.body_link_quat_w[2, 1] = reference
    arm.process_actions_for_envs(torch.zeros(1, 6), torch.tensor([2]))
    torch.testing.assert_close(arm.desired_c_pose_b[2, 3:], reference)
    robot.data.body_link_quat_w[2, 1] = torch.tensor([1., 0., 0., 0.])
    arm.process_actions_for_envs(torch.zeros(1, 6), torch.tensor([2]))
    torch.testing.assert_close(arm.desired_c_pose_b[2, 3:], reference)


@pytest.mark.parametrize("cap", (0., -.01, float("nan"), float("inf")))
def test_invalid_error_cap_fails_before_creating_controller(backend, cap):
    with pytest.raises(ValueError, match="finite and positive"):
        _fixture(backend, cap=cap)


def test_pure_target_geometry_rotates_error_and_preserves_target_at_zero_command():
    q = action_math.quaternion_from_rotation_vector(torch.tensor([[0., 0., math.pi / 2.]], dtype=torch.float64))
    current = torch.tensor([[1., 2., 3.]], dtype=q.dtype)
    target = current + torch.tensor([[.03, .02, 0.]], dtype=q.dtype)
    held = accumulated_translation_target(target, current, q, torch.zeros_like(current), .06)
    torch.testing.assert_close(held, target)
    torch.testing.assert_close(translation_target_error_c(held, current, q), torch.tensor([[.02, -.03, 0.]], dtype=q.dtype))
