"""Exercise the production reset guards with real hand quaternions on CPU."""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path
import sys
from itertools import product
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from hand_manipulation_test import action_math


ROOT = Path(__file__).resolve().parents[1] / "hand_manipulation_test"
PACKAGE = "_push_v1_reset_guard_fixture"


def _load(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def backend():
    # Load the complete production classes; replace only unavailable simulator
    # dependencies. Constructors/IK are unnecessary for the final pose gate.
    package, mdp, assets = (ModuleType(PACKAGE + suffix) for suffix in ("", ".mdp", ".assets"))
    package.__path__, mdp.__path__, assets.__path__ = [str(ROOT)], [str(ROOT / "mdp")], []
    robot = ModuleType(PACKAGE + ".assets.robot")
    robot.ARM_JOINT_NAMES = action_math.UR5E_ARM_JOINT_NAMES
    robot.HAND_JOINT_NAMES = action_math.INSPIRE_HAND_JOINT_NAMES
    bounds = ModuleType(PACKAGE + ".collision_geometry")
    bounds.RobotCollisionBounds = object
    managers = ModuleType("isaaclab.managers")
    managers.ManagerTermBase = object
    sim_utils = ModuleType("isaaclab.sim.utils")
    sim_utils.get_current_stage = lambda: None
    utils, math_utils = ModuleType("isaaclab.utils"), ModuleType("isaaclab.utils.math")
    math_utils.quat_apply = action_math.quaternion_rotate
    math_utils.quat_apply_inverse = lambda q, v: action_math.quaternion_rotate(action_math.quaternion_conjugate(q), v)
    utils.math = math_utils
    modules = {
        PACKAGE: package, mdp.__name__: mdp, assets.__name__: assets,
        PACKAGE + ".action_math": action_math, robot.__name__: robot, bounds.__name__: bounds,
        "isaaclab": ModuleType("isaaclab"), managers.__name__: managers,
        "isaaclab.sim": ModuleType("isaaclab.sim"), sim_utils.__name__: sim_utils,
        utils.__name__: utils, math_utils.__name__: math_utils,
    }
    with patch.dict(sys.modules, modules):
        contact = _load(PACKAGE + ".mdp.contact_events", "mdp/contact_events.py")
        push = _load(PACKAGE + ".mdp.push_events", "mdp/push_events.py")
        v1 = _load(PACKAGE + ".mdp.push_v1_events", "mdp/push_v1_events.py")
    return contact, push, v1


def _quaternions():
    half = math.sqrt(0.5)
    # H columns respectively: [up,+Y,-X], [up,-Y,+X], [down,-Y,-X].
    return torch.tensor(((half, 0., -half, 0.), (0., half, 0., half), (0., half, 0., -half)))


def _reset(cls, quaternion=None):
    quaternion = _quaternions() if quaternion is None else quaternion.clone()
    count = len(quaternion)
    origins = torch.tensor(((-48.75, 37.5, 0.), (12., -8., 0.), (0., 0., 0.)))[:count]
    robot = SimpleNamespace(data=SimpleNamespace(
        root_link_pos_w=origins.clone(), joint_pos=torch.zeros(count, 6),
        soft_joint_pos_limits=torch.tensor((-6., 6.)).expand(count, 6, 2).clone(),
    ))
    robot.data.joint_pos[:, 4] = .8
    task = SimpleNamespace(
        reset_joint_limit_margin_rad=.035, reset_wrist_3_range_rad=(-2.8, 2.8),
        reset_position_tolerance_m=.003, reset_orientation_tolerance_rad=.05,
        minimum_finger_outward_cos=.25, wrist_2_branch_sin_margin=.15,
    )
    scene = {"table": SimpleNamespace(data=SimpleNamespace(
        root_pos_w=origins + torch.tensor((-.75, 0., 1.03)),
    ))}
    term = object.__new__(cls)
    term._env = SimpleNamespace(cfg=SimpleNamespace(task=task), scene=scene)
    term.robot, term.arm_joint_ids, term.wrist_local_index = robot, list(range(6)), 5
    term.position_error_m = torch.full((count,), -1.)
    term.orientation_error_rad = torch.full((count,), -1.)
    term.palm_alignment_cos = torch.full((count,), -1.)
    term.direction_w = action_math.quaternion_rotate(
        quaternion, torch.tensor((0., 1., 0.)).expand(count, -1)
    )
    position_errors, rotation_errors = torch.zeros(count, 3), torch.zeros(count, 3)
    term._pose_errors = lambda ids: (position_errors[ids], rotation_errors[ids])
    term.palm_reference_pose_w = lambda ids: (origins[ids], quaternion[ids])
    return term, position_errors, rotation_errors


def test_v1_inherits_actual_outward_guard_and_accepts_left_thumb_down(backend):
    contact, _, v1 = backend
    assert v1.PushV1SafePoseReset._finger_outward_cos is contact.ContactSafePoseReset._finger_outward_cos
    term, _, _ = _reset(v1.PushV1SafePoseReset)
    ids = torch.arange(3)
    quaternion = _quaternions()
    finger = action_math.quaternion_rotate(quaternion, torch.tensor((0., 0., 1.)).expand(3, -1))
    torch.testing.assert_close(finger[:, 0], torch.tensor((-1., 1., -1.)))
    torch.testing.assert_close(term._finger_outward_cos(ids, quaternion), torch.tensor((1., -1., 1.)))
    assert term._check_pose_and_limits(ids).tolist() == [True, False, True]


def test_v0_keeps_the_actual_base_to_table_outward_finger_guard(backend):
    contact, push, _ = backend
    # The unmodified source inherits Contact's geometry-based guard. This
    # exercises its real root/table positions, including translated envs.
    assert push.PushSafePoseReset._finger_outward_cos is contact.ContactSafePoseReset._finger_outward_cos
    term, _, _ = _reset(push.PushSafePoseReset)
    ids = torch.arange(3)
    torch.testing.assert_close(term._finger_outward_cos(ids, _quaternions()), torch.tensor((1., -1., 1.)))
    assert term._check_pose_and_limits(ids).tolist() == [True, False, True]


@pytest.mark.parametrize("failure", ("position", "rotation", "soft_joint_limit", "wrist_limit", "normal_alignment", "nonfinite_joint"))
def test_v1_guard_retains_the_parent_pose_and_joint_checks(backend, failure):
    _, _, v1 = backend
    term, position_errors, rotation_errors = _reset(v1.PushV1SafePoseReset, _quaternions()[:1])
    if failure == "position":
        position_errors[0, 0] = .00301
    elif failure == "rotation":
        rotation_errors[0, 0] = .05001
    elif failure == "soft_joint_limit":
        term.robot.data.joint_pos[0, 0] = 6. - .035 + .001
    elif failure == "wrist_limit":
        term.robot.data.joint_pos[0, 5] = 2.801
    elif failure == "normal_alignment":
        term.direction_w[0] = torch.tensor((1., 0., 0.))
    else:
        term.robot.data.joint_pos[0, 0] = float("nan")
    ids = torch.tensor((0,))
    torch.testing.assert_close(term._finger_outward_cos(ids, _quaternions()[:1]), torch.ones(1))
    assert not term._check_pose_and_limits(ids).item()


def test_v1_pose_gate_updates_only_selected_environment_diagnostics(backend):
    _, _, v1 = backend
    term, _, _ = _reset(v1.PushV1SafePoseReset)
    ids = torch.tensor((2, 0))
    assert term._check_pose_and_limits(ids).tolist() == [True, True]
    torch.testing.assert_close(term.position_error_m, torch.tensor((0., -1., 0.)))
    torch.testing.assert_close(term.orientation_error_rad, torch.tensor((0., -1., 0.)))
    torch.testing.assert_close(term.palm_alignment_cos, torch.tensor((1., -1., 1.)))


@pytest.mark.parametrize("angle, accepted", (
    (.8, True), (.8+2*math.pi, True), (.8-2*math.pi, True),
    (.01, False), (-.8, False), (math.pi-.01, False),
    (math.asin(.15)-1.e-6, False), (math.asin(.15)+1.e-6, True),
    (float("nan"), False),
))
def test_v1_actual_wrist_branch_accepts_periodic_equivalents_and_rejects_flip(backend, angle, accepted):
    _, _, v1 = backend
    term, _, _ = _reset(v1.PushV1SafePoseReset, _quaternions()[:1])
    term.robot.data.soft_joint_pos_limits[:] = torch.tensor((-10., 10.))
    term.robot.data.joint_pos[0, 4] = angle
    assert term._check_pose_and_limits(torch.tensor((0,))).item() is accepted


def test_v1_wrist_guard_uses_named_arm_mapping_instead_of_global_joint_offset(backend):
    _, _, v1 = backend
    term, _, _ = _reset(v1.PushV1SafePoseReset, _quaternions()[:1].expand(2, -1))
    term.arm_joint_ids = [1, 3, 5, 7, 9, 11]
    term.robot.data.joint_pos = torch.zeros(2, 12)
    term.robot.data.soft_joint_pos_limits = torch.tensor((-6., 6.)).expand(2, 12, 2).clone()
    term.robot.data.joint_pos[:, 9] = torch.tensor((.8, -.8))
    term.robot.data.joint_pos[:, 4] = torch.tensor((-.8, .8))
    assert term._check_pose_and_limits(torch.arange(2)).tolist() == [True, False]


@pytest.mark.parametrize("cosine, accepted", ((.249, False), (.25, False), (.251, True)))
def test_v1_strengthens_parent_positive_outward_guard_to_strict_minimum(backend, cosine, accepted):
    _, push, v1 = backend
    term, _, _ = _reset(v1.PushV1SafePoseReset, _quaternions()[:1])
    term.robot.data.root_link_pos_w[:] = 0.
    term._env.scene["table"].data.root_pos_w[:] = torch.tensor((-cosine, math.sqrt(1-cosine**2), 1.03))
    ids = torch.tensor((0,))
    if cosine == .25:
        # Use the represented physical cosine for exact equality; quaternion
        # rotation and vector normalization can round mathematical .25 upward.
        term._env.cfg.task.minimum_finger_outward_cos = term._finger_outward_cos(ids).item()
    assert push.PushSafePoseReset._check_pose_and_limits(term, ids).item()
    assert term._check_pose_and_limits(ids).item() is accepted


def _contact_reference_fixture(backend):
    _, _, v1 = backend
    term, _, _ = _reset(v1.PushV1SafePoseReset, _quaternions()[[0, 2, 0]])
    term._env.num_envs, term._env.device = 3, "cpu"
    term.hand_body_id, term._thumb_contact_body_id = 0, 1
    term.palm_reference_h = torch.tensor((.00045, .047734, .12023))
    hand_position = term.robot.data.root_link_pos_w + torch.tensor((-.70, -.14, 1.05))
    term.robot.data.body_link_pos_w = torch.stack((hand_position, hand_position + torch.tensor((.012, .03, .06))), 1)
    term.robot.data.body_link_quat_w = torch.stack((_quaternions()[[0, 2, 0]], torch.tensor((1., 0., 0., 0.)).expand(3, -1)), 1)

    def central(ids):
        data = term.robot.data
        quaternion = data.body_link_quat_w[ids, term.hand_body_id]
        point = data.body_link_pos_w[ids, term.hand_body_id] + action_math.quaternion_rotate(
            quaternion, term.palm_reference_h.expand(len(ids), -1)
        )
        return point, quaternion

    term.palm_reference_pose_w = central
    # Explicit geometry tests the cached surface adapter, not USD loading or
    # physical collision. No normal-step mesh access is available in this fixture.
    term._thumb_contact_vertices_b = torch.tensor(list(product((-.01, .01), (-.02, .02), (-.003, .003))))
    term._thumb_contact_face_b = torch.zeros(3, 3)
    term._use_thumb_contact = torch.zeros(3, dtype=torch.bool)
    return term


def test_cached_contact_reference_transforms_real_body_point_and_retains_outward_gate(backend):
    term = _contact_reference_fixture(backend)
    ids = torch.arange(3)
    term._cache_contact_reference(ids)
    assert term._use_thumb_contact.tolist() == [False, True, False]
    torch.testing.assert_close(term._thumb_contact_face_b, torch.tensor(((0., .02, 0.), (0., -.02, 0.), (0., .02, 0.))))
    actual, quaternion = term.contact_reference_pose_w()
    central, _ = term.palm_reference_pose_w(ids)
    expected = central.clone()
    expected[1] = term.robot.data.body_link_pos_w[1, 1] + torch.tensor((0., -.02, 0.))
    torch.testing.assert_close(actual, expected, atol=3.e-6, rtol=0.)
    torch.testing.assert_close(quaternion, term.robot.data.body_link_quat_w[:, 0])
    assert bool(term._check_pose_and_limits(ids).all())
    # A thumb selection never substitutes for the physical outward predicate.
    term.robot.data.body_link_quat_w[1, 0] = _quaternions()[1]
    assert not term._check_pose_and_limits(torch.tensor((1,))).item()


def test_live_thumb_joint_motion_changes_reference_and_control_offset_without_face_resampling(backend):
    term = _contact_reference_fixture(backend)
    ids = torch.arange(3)
    term._cache_contact_reference(ids)
    saved_face = term._thumb_contact_face_b.clone()
    before = term.contact_reference_pose_w()[0]
    term.robot.data.body_link_pos_w[1, 1, 0] += .005
    angle = .3
    term.robot.data.body_link_quat_w[1, 1] = torch.tensor((math.cos(angle/2), math.sin(angle/2), 0., 0.))
    after, hand_quaternion = term.contact_reference_pose_w()
    expected = term.robot.data.body_link_pos_w[1, 1] + torch.tensor((0., -.02*math.cos(angle), -.02*math.sin(angle)))
    torch.testing.assert_close(after[1], expected, atol=2.e-6, rtol=0.)
    assert float((after[1]-before[1]).norm()) > .004
    torch.testing.assert_close(after[[0, 2]], before[[0, 2]], atol=0., rtol=0.)
    torch.testing.assert_close(term._thumb_contact_face_b, saved_face, atol=0., rtol=0.)
    offset = term.contact_reference_offset_h()
    recovered = term.robot.data.body_link_pos_w[:, 0] + action_math.quaternion_rotate(hand_quaternion, offset)
    torch.testing.assert_close(recovered, after, atol=4.e-6, rtol=0.)
    torch.testing.assert_close(offset[[0, 2]], term.palm_reference_h.expand(2, -1), atol=5.e-6, rtol=0.)


def test_partial_contact_face_reset_keeps_other_rows_and_selects_surface_in_actual_pad_axes(backend):
    term = _contact_reference_fixture(backend)
    term._cache_contact_reference(torch.arange(3))
    saved_face, saved_selection = term._thumb_contact_face_b.clone(), term._use_thumb_contact.clone()
    # Rotate the actual thumb pad by pi/2 about world Z. The left H normal
    # then supports its local -X face instead of its former -Y face.
    term.robot.data.body_link_quat_w[1, 1] = torch.tensor((math.sqrt(.5), 0., 0., math.sqrt(.5)))
    term._cache_contact_reference(torch.tensor((1,)))
    torch.testing.assert_close(term._thumb_contact_face_b[1], torch.tensor((-.01, 0., 0.)), atol=2.e-8, rtol=0.)
    torch.testing.assert_close(term._thumb_contact_face_b[[0, 2]], saved_face[[0, 2]], atol=0., rtol=0.)
    torch.testing.assert_close(term._use_thumb_contact, saved_selection, atol=0., rtol=0.)
    point, quaternion = term.contact_reference_pose_w(torch.tensor((1,)))
    expected = term.robot.data.body_link_pos_w[1, 1] + torch.tensor((0., -.01, 0.))
    torch.testing.assert_close(point[0], expected, atol=2.e-6, rtol=0.)
    assert point.shape == (1, 3) and quaternion.shape == (1, 4)


def _branch_seed_fixture(backend):
    _, push, v1 = backend
    count, width = 8, 18
    defaults = torch.arange(count*width, dtype=torch.float64).reshape(count, width)/100.
    actual = defaults + 10.
    robot = SimpleNamespace(data=SimpleNamespace(default_joint_pos=defaults, joint_pos=actual))
    task = SimpleNamespace(
        right_reset_arm_seed=(.179, .154, -1.560, -4.876, 1.750, 1.571),
        left_reset_arm_seed=(-.615, .159, -1.625, -4.818, .955, -1.571),
    )
    command = SimpleNamespace(
        angle_rad=torch.tensor((-10., 0., 10., 170., 180., 190., 0., 180.), dtype=defaults.dtype)*math.pi/180,
        distance_m=torch.full((count,), .25, dtype=defaults.dtype),
        target_pos_w=torch.tensor((-.70, 0., 1.08), dtype=defaults.dtype).expand(count, -1).clone(),
    )
    env = SimpleNamespace(num_envs=count, device="cpu", cfg=SimpleNamespace(task=task),
                          command_manager=SimpleNamespace(get_term=lambda _: command))
    term = object.__new__(v1.PushV1SafePoseReset)
    term._env, term.robot = env, robot
    term.arm_joint_ids = [1, 3, 5, 7, 9, 11]
    cache_calls = []
    term._cache_contact_reference = lambda ids: cache_calls.append((ids.clone(), defaults.clone()))
    return term, push, env, command, cache_calls


@pytest.mark.parametrize("parent_fails", (False, True))
def test_v1_side_seed_defaults_are_temporary_selected_arm_only_and_restore_after_parent(parent_fails, backend):
    term, push, env, command, cache_calls = _branch_seed_fixture(backend)
    ids = torch.tensor((5, 0, 4, 1, 3, 2))
    saved_defaults = term.robot.data.default_joint_pos.clone()
    saved_actual = term.robot.data.joint_pos.clone()
    saved_command = (command.angle_rad.clone(), command.distance_m.clone(), command.target_pos_w.clone())
    expected = saved_defaults.clone()
    expected[ids[:, None], torch.tensor(term.arm_joint_ids)] = torch.tensor(
        (env.cfg.task.left_reset_arm_seed, env.cfg.task.right_reset_arm_seed,
         env.cfg.task.left_reset_arm_seed, env.cfg.task.right_reset_arm_seed,
         env.cfg.task.left_reset_arm_seed, env.cfg.task.right_reset_arm_seed), dtype=saved_defaults.dtype
    )
    parent_calls = []

    def parent_call(self, passed_env, passed_ids):
        assert passed_env is env
        torch.testing.assert_close(passed_ids, ids, atol=0., rtol=0.)
        torch.testing.assert_close(self.robot.data.default_joint_pos, expected, atol=0., rtol=0.)
        # V1 only substitutes defaults; physical state writes belong to the
        # bounded parent reset. Preserve its actual IK result after restoration.
        torch.testing.assert_close(self.robot.data.joint_pos, saved_actual, atol=0., rtol=0.)
        self.robot.data.joint_pos[passed_ids] = expected[passed_ids]+.01
        parent_calls.append(passed_ids.clone())
        if parent_fails:
            raise RuntimeError("bounded IK fixture did not converge")

    with patch.object(push.PushSafePoseReset, "__call__", parent_call):
        if parent_fails:
            with pytest.raises(RuntimeError, match="did not converge"):
                term(env, ids)
        else:
            term(env, ids)
    assert len(parent_calls) == 1
    torch.testing.assert_close(term.robot.data.default_joint_pos, saved_defaults, atol=0., rtol=0.)
    torch.testing.assert_close(term.robot.data.joint_pos[ids], expected[ids]+.01, atol=0., rtol=0.)
    torch.testing.assert_close(term.robot.data.joint_pos[[6, 7]], saved_actual[[6, 7]], atol=0., rtol=0.)
    for current, saved in zip((command.angle_rad, command.distance_m, command.target_pos_w), saved_command):
        torch.testing.assert_close(current, saved, atol=0., rtol=0.)
    assert len(cache_calls) == (0 if parent_fails else 1)
    if cache_calls:
        torch.testing.assert_close(cache_calls[0][0], ids, atol=0., rtol=0.)
        torch.testing.assert_close(cache_calls[0][1], saved_defaults, atol=0., rtol=0.)


def test_v1_all_row_reset_selects_both_branches_and_empty_selection_does_nothing(backend):
    term, push, env, command, cache_calls = _branch_seed_fixture(backend)
    saved = term.robot.data.default_joint_pos.clone()
    captured = []

    def parent_call(self, passed_env, ids):
        captured.append((ids.clone(), self.robot.data.default_joint_pos[:, self.arm_joint_ids].clone()))

    with patch.object(push.PushSafePoseReset, "__call__", parent_call):
        term(env, torch.empty(0, dtype=torch.long))
        assert not captured and not cache_calls
        torch.testing.assert_close(term.robot.data.default_joint_pos, saved, atol=0., rtol=0.)
        term(env, None)
    assert len(captured) == len(cache_calls) == 1
    torch.testing.assert_close(captured[0][0], torch.arange(env.num_envs), atol=0., rtol=0.)
    right, left = (saved.new_tensor(getattr(env.cfg.task, name)) for name in ("right_reset_arm_seed", "left_reset_arm_seed"))
    torch.testing.assert_close(captured[0][1][[0, 1, 2, 6]], right.expand(4, -1), atol=0., rtol=0.)
    torch.testing.assert_close(captured[0][1][[3, 4, 5, 7]], left.expand(4, -1), atol=0., rtol=0.)
    torch.testing.assert_close(term.robot.data.default_joint_pos, saved, atol=0., rtol=0.)
