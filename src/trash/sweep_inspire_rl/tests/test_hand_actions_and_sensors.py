"""Numerical control/sensor checks without loading an Isaac Sim runtime."""

from __future__ import annotations

import ast
import importlib.util
import math
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch


ROOT = Path(__file__).resolve().parents[3]
SOURCE_ACTIONS = ROOT / "src/hand_manipulation_rl/hand_manipulation_rl/mdp/actions.py"
SOURCE_SENSORS = ROOT / "src/hand_manipulation_rl/hand_manipulation_rl/sensors.py"
NEW_MDP = ROOT / "src/sweep_inspire_rl/sweep_inspire_rl/mdp"


def _execute_source(path, namespace):
    """Execute production math/classes, replacing only simulator imports."""

    tree = ast.parse(path.read_text(), filename=str(path))
    nodes = [
        node for node in tree.body
        if not isinstance(node, (ast.Import, ast.ImportFrom))
        or isinstance(node, ast.ImportFrom) and node.module == "__future__"
    ]
    namespace["__file__"] = str(path)
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), str(path), "exec"), namespace)
    return SimpleNamespace(**namespace)


class FakeArticulation:
    def __init__(self, joint_pos, source):
        self.joint_names = source.INSPIRE_HAND_JOINT_NAMES
        self.data = SimpleNamespace(
            joint_pos=joint_pos,
            soft_joint_pos_limits=torch.tensor([[[-10.0, 10.0]] * 12]).expand(len(joint_pos), -1, -1),
        )
        self.targets = joint_pos.clone()

    def find_joints(self, names, preserve_order=False):
        return [self.joint_names.index(name) for name in names], names

    def set_joint_position_target(self, targets, joint_ids, env_ids=None):
        if env_ids is None:
            self.targets[:, joint_ids] = targets
        else:
            self.targets[env_ids[:, None], torch.tensor(joint_ids)[None, :]] = targets


class FakeActionTerm:
    def __init__(self, cfg, env):
        self.cfg = cfg
        self._env = env
        self._asset = env.scene[cfg.asset_name]
        self.num_envs = env.num_envs
        self.device = env.device


class FakeOscAction:
    def _preprocess_actions(self, actions):
        self._raw_actions[:] = actions
        self._processed_actions[:] = actions
        self._processed_actions[:, :3] *= self.cfg.position_scale
        self._processed_actions[:, 3:] *= self.cfg.orientation_scale


source_actions = _execute_source(
    SOURCE_ACTIONS,
    {
        "math": math,
        "torch": torch,
        "TYPE_CHECKING": False,
        "ActionTerm": FakeActionTerm,
        "ActionTermCfg": type("ActionTermCfg", (), {}),
        "Articulation": FakeArticulation,
        "configclass": lambda cls: cls,
    },
)


def _rotation_vector(quaternion):
    quat = quaternion / torch.linalg.vector_norm(quaternion, dim=-1, keepdim=True)
    quat = torch.where(quat[:, :1] < 0, -quat, quat)
    vector = quat[:, 1:]
    sine = torch.linalg.vector_norm(vector, dim=-1, keepdim=True)
    angle = 2.0 * torch.atan2(sine, quat[:, :1])
    return vector * torch.where(sine > 1e-8, angle / sine.clamp_min(1e-8), torch.full_like(sine, 2.0))


math_utils = SimpleNamespace(
    quat_mul=source_actions.quaternion_multiply,
    quat_inv=source_actions.quaternion_conjugate,
    quat_apply=source_actions.quaternion_rotate,
    quat_apply_inverse=lambda quaternion, vector: source_actions.quaternion_rotate(
        source_actions.quaternion_conjugate(quaternion), vector
    ),
    matrix_from_quat=source_actions.quaternion_to_matrix,
    axis_angle_from_quat=_rotation_vector,
    quat_unique=lambda quaternion: torch.where(quaternion[:, :1] < 0, -quaternion, quaternion),
)

actions = _execute_source(
    NEW_MDP / "actions.py",
    {
        "math": math,
        "torch": torch,
        "math_utils": math_utils,
        "configclass": lambda cls: cls,
        "OperationalSpaceControllerAction": FakeOscAction,
        "OperationalSpaceControllerActionCfg": type("OperationalSpaceControllerActionCfg", (), {}),
        "InspireHandSynergyAction": source_actions.InspireHandSynergyAction,
        "InspireHandSynergyActionCfg": source_actions.InspireHandSynergyActionCfg,
        "inspire_joint_positions_to_synergy": source_actions.inspire_joint_positions_to_synergy,
        "inspire_synergy_to_joint_positions": source_actions.inspire_synergy_to_joint_positions,
        "shift_jacobian_to_point": source_actions.shift_jacobian_to_point,
    },
)

sensor_spec = importlib.util.spec_from_file_location("sweep_inspire_test_sensors", SOURCE_SENSORS)
sensors = importlib.util.module_from_spec(sensor_spec)
sys.modules[sensor_spec.name] = sensors
sensor_spec.loader.exec_module(sensors)
observations = _execute_source(
    NEW_MDP / "observations.py",
    {
        "torch": torch,
        "math_utils": math_utils,
        "FixedJointWrenchReader": sensors.FixedJointWrenchReader,
        "PALM_CHANNEL_NAMES": sensors.PALM_CHANNEL_NAMES,
        "contact_magnitudes_and_bits": sensors.contact_magnitudes_and_bits,
    },
)


def _hand_term(initial_openness=(1.0, 1.0), num_envs=2):
    joint_pos = source_actions.inspire_synergy_to_joint_positions(torch.tensor([initial_openness] * num_envs))
    robot = FakeArticulation(joint_pos, source_actions)
    env = SimpleNamespace(scene={"robot": robot}, num_envs=num_envs, device="cpu", step_dt=0.1)
    cfg = SimpleNamespace(
        asset_name="robot",
        min_openness=0.8,
        joint_names=list(source_actions.INSPIRE_HAND_JOINT_NAMES),
        open_positions=source_actions.INSPIRE_OPEN_JOINT_POSITIONS,
        closed_positions=source_actions.INSPIRE_CLOSED_JOINT_POSITIONS,
        max_synergy_rate=(1.0, 1.0),
        max_joint_target_rate=(10.0,) * 12,
        saturation_tolerance=1e-6,
    )
    return actions.NearlyOpenHandSynergyAction(cfg, env), robot


def test_hand_endpoints_common_flexion_and_independent_thumb_followers():
    term, robot = _hand_term()
    action = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    for _ in range(3):
        term.process_actions(action)
    term.apply_actions()
    torch.testing.assert_close(term.raw_actions, action)
    torch.testing.assert_close(term.processed_actions, torch.tensor([[0.8, 1.0], [1.0, 0.8]]))
    # First coordinate bends every finger, including the explicit mimic joints.
    assert robot.targets[0, 4] > 0 and robot.targets[0, 5] > 0
    torch.testing.assert_close(robot.targets[:, 5], 1.0843 * robot.targets[:, 4])
    torch.testing.assert_close(robot.targets[:, 2], 0.8024 * robot.targets[:, 1])
    # Second coordinate acts only on thumb1; common flexion remains fully open.
    assert robot.targets[1, 0] > 0.2
    torch.testing.assert_close(robot.targets[1, 1:], torch.zeros(11))


def test_hand_rate_limit_reset_and_partial_processing():
    term, _ = _hand_term((0.5, 0.5))
    torch.testing.assert_close(term.processed_actions, torch.full((2, 2), 0.8))
    term.process_actions_for_envs(torch.ones(1, 2), torch.tensor([1]))
    torch.testing.assert_close(term.processed_actions[0], torch.full((2,), 0.8))
    torch.testing.assert_close(term.processed_actions[1], torch.full((2,), 0.9))
    assert term.action_saturated[1]
    term.reset([1])
    torch.testing.assert_close(term.processed_actions[1], torch.full((2,), 0.8))
    # Reset only the selected row, leaving another environment's target alone.
    torch.testing.assert_close(term.processed_actions[0], torch.full((2,), 0.8))


def test_hand_input_bounds_and_nonfinite_values_do_not_close_hand():
    term, _ = _hand_term()
    for _ in range(3):
        term.process_actions(torch.tensor([[-20.0, 4.0], [float("nan"), float("inf")]]))
    assert torch.isfinite(term.processed_actions).all()
    assert torch.all(term.processed_actions >= 0.8 - 1e-6)
    assert torch.all(term.processed_actions <= 1.0 + 1e-6)
    torch.testing.assert_close(term.raw_actions, torch.tensor([[0.0, 1.0], [0.0, 1.0]]))
    assert term.action_saturated.all()


def test_fixed_palm_quaternion_faces_right_with_fingers_into_shelf():
    quaternion = torch.tensor([actions.RIGHT_PALM_QUAT_WXYZ])
    axes = source_actions.quaternion_to_matrix(quaternion)[0]
    torch.testing.assert_close(axes[:, 0], torch.tensor([0.0, 0.0, 1.0]), atol=1e-6, rtol=0)
    torch.testing.assert_close(axes[:, 1], torch.tensor([0.0, 1.0, 0.0]), atol=1e-6, rtol=0)
    torch.testing.assert_close(axes[:, 2], torch.tensor([-1.0, 0.0, 0.0]), atol=1e-6, rtol=0)


def test_osc_projection_holds_world_palm_and_preserves_translation():
    term = actions.RightPalmOscAction.__new__(actions.RightPalmOscAction)
    term.num_envs, term.device = 2, "cpu"
    term.cfg = SimpleNamespace(palm_quat_w=actions.RIGHT_PALM_QUAT_WXYZ, position_scale=1.0, orientation_scale=1.0)
    term._raw_actions = torch.zeros(2, 6)
    term._processed_actions = torch.zeros(2, 6)
    root_quat = torch.tensor([[0.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 0.0]])
    term._asset = SimpleNamespace(data=SimpleNamespace(root_quat_w=root_quat))
    term._offset_rot = None
    term._task_frame_pose_b = None
    current_quat = source_actions.quaternion_from_rotation_vector(torch.tensor([[0.2, 0.1, -0.3], [-0.6, 0.7, 0.2]]))
    term._ee_pose_b = torch.cat((torch.zeros(2, 3), current_quat), dim=-1)
    inputs = torch.tensor([[0.1, 0.2, 0.3, 10.0, -9.0, 7.0], [-0.2, 0.4, -0.1, -3.0, 6.0, 2.0]])
    term._preprocess_actions(inputs)
    torch.testing.assert_close(term._processed_actions[:, :3], inputs[:, :3])
    delta = source_actions.quaternion_from_rotation_vector(term._processed_actions[:, 3:])
    target_b = source_actions.quaternion_multiply(delta, current_quat)
    target_w = source_actions.quaternion_multiply(root_quat, target_b)
    palm_normal_w = source_actions.quaternion_rotate(target_w, torch.tensor([[0.0, 1.0, 0.0]]).expand(2, -1))
    torch.testing.assert_close(palm_normal_w, torch.tensor([[0.0, 1.0, 0.0]]).expand(2, -1), atol=1e-6, rtol=0)


def test_osc_com_jacobian_and_link_velocity_use_same_rotated_control_point():
    term = actions.RightPalmOscAction.__new__(actions.RightPalmOscAction)
    term._ee_body_idx = 0
    term._compute_ee_pose = lambda: None
    root_quat = torch.tensor([[0.0, 0.0, 0.0, 1.0]], dtype=torch.float64)
    hand_quat = torch.tensor([actions.RIGHT_PALM_QUAT_WXYZ], dtype=torch.float64)
    hand_pos = torch.tensor([[-0.6, 0.1, 1.2]], dtype=torch.float64)
    hand_linear = torch.tensor([[0.03, -0.02, 0.01]], dtype=torch.float64)
    angular = torch.tensor([[0.1, 0.2, 0.3]], dtype=torch.float64)
    offset_h = torch.tensor([[0.0, 0.05, 0.1]], dtype=torch.float64)
    com_offset_h = torch.tensor([[0.02, -0.01, 0.03]], dtype=torch.float64)
    h_to_c_w = source_actions.quaternion_rotate(hand_quat, offset_h)
    h_to_com_w = source_actions.quaternion_rotate(hand_quat, com_offset_h)
    com_linear_w = hand_linear + torch.linalg.cross(angular, h_to_com_w, dim=-1)
    world_to_b = source_actions.quaternion_to_matrix(source_actions.quaternion_conjugate(root_quat))
    com_twist_b = torch.cat(((world_to_b @ com_linear_w.unsqueeze(-1)).squeeze(-1),
                            (world_to_b @ angular.unsqueeze(-1)).squeeze(-1)), dim=-1)
    term.jacobian_b = com_twist_b.unsqueeze(-1)
    term._jacobian_b = torch.zeros(1, 6, 1, dtype=torch.float64)
    hand_quat_b = source_actions.quaternion_multiply(source_actions.quaternion_conjugate(root_quat), hand_quat)
    c_pos_b = (world_to_b @ (hand_pos + h_to_c_w).unsqueeze(-1)).squeeze(-1)
    term._ee_pose_b = torch.cat((c_pos_b, hand_quat_b), dim=-1)
    h_pos_b = (world_to_b @ hand_pos.unsqueeze(-1)).squeeze(-1)
    term._ee_pose_b_no_offset = torch.cat((h_pos_b, hand_quat_b), dim=-1)
    term._offset_pos = offset_h
    term._ee_vel_w = torch.zeros(1, 6, dtype=torch.float64)
    term._ee_vel_b = torch.zeros(1, 6, dtype=torch.float64)
    data = SimpleNamespace(
        root_quat_w=root_quat, root_pos_w=torch.zeros(1, 3, dtype=torch.float64),
        root_link_vel_w=torch.zeros(1, 6, dtype=torch.float64),
        body_com_pos_w=(hand_pos + h_to_com_w).unsqueeze(1),
        body_link_vel_w=torch.cat((hand_linear, angular), dim=-1).unsqueeze(1),
    )
    term._asset = SimpleNamespace(data=data)
    term._compute_ee_jacobian()
    term._compute_ee_velocity()
    torch.testing.assert_close(term._jacobian_b.squeeze(-1), term._ee_vel_b)
    # Independent pose finite difference checks transport to the physical C.
    dt = 1e-6
    next_quat = source_actions.quaternion_multiply(
        source_actions.quaternion_from_rotation_vector(angular * dt), hand_quat
    )
    next_c_w = hand_pos + hand_linear * dt + source_actions.quaternion_rotate(next_quat, offset_h)
    finite_difference_b = (world_to_b @ ((next_c_w - hand_pos - h_to_c_w) / dt).unsqueeze(-1)).squeeze(-1)
    torch.testing.assert_close(term._ee_vel_b[:, :3], finite_difference_b, atol=1e-7, rtol=0)


def test_palm_channels_reorder_and_inclusive_threshold():
    names = tuple(reversed(sensors.PALM_CHANNEL_NAMES))
    force = torch.zeros(2, 17, 3)
    force[0, names.index("inspire_palm_force_sensor"), 0] = 0.05
    force[1, names.index("inspire_index_force_sensor_1"), 1] = -0.1
    env = SimpleNamespace(
        scene={"palm_tactile": SimpleNamespace(body_names=names, data=SimpleNamespace(net_forces_w=force))},
        num_envs=2, device="cpu",
    )
    result = observations.palm_tactile_bits(env)
    assert result.shape == (2, 17)
    assert result[0, 0] == 1 and result[0].sum() == 1
    assert result[1, 5] == 1 and result[1].sum() == 1
    env.scene["palm_tactile"].data.net_forces_w.zero_()
    assert observations.palm_tactile_bits(env).sum() == 0


def test_palm_channels_reject_missing_or_dorsal_approximation_bodies():
    env = SimpleNamespace(
        scene={"palm_tactile": SimpleNamespace(body_names=sensors.DORSAL_PARENT_BY_CHANNEL)}, device="cpu"
    )
    with pytest.raises(RuntimeError, match="17 physical pads"):
        observations.palm_tactile_bits(env)


def test_wrench_uses_measurement_child_and_transports_force_and_moment():
    names = ("axia80_link", "inspire_base_link", "wrist_3_link")
    wrench = torch.zeros(1, 3, 6)
    wrench[0, 0] = 999.0  # parent-body reaction is intentionally the wrong signal
    wrench[0, 1] = torch.tensor([3.0, 0.0, 0.0, 0.0, 2.0, 0.0])
    # With H rotated 90deg around Z, F's +X force becomes H -Y.
    h_quat = source_actions.quaternion_from_rotation_vector(torch.tensor([[0.0, 0.0, math.pi / 2]]))
    pose_quat = torch.tensor([[[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]])
    pose_quat[:, 1] = h_quat
    positions = torch.zeros(1, 3, 3)
    positions[0, 0, 2] = -0.02
    robot = SimpleNamespace(body_names=names, data=SimpleNamespace(
        body_incoming_joint_wrench_b=wrench, body_pos_w=positions, body_quat_w=pose_quat
    ))
    env = SimpleNamespace(scene={"robot": robot}, num_envs=1, device="cpu")
    result = observations.wrist_wrench_c(env)
    # C->F in H=(0,-.05,-.12), F_H=(0,-3,0), moment shift=(-.36,0,0).
    torch.testing.assert_close(result, torch.tensor([[0.0, -3.0, 0.0, 1.64, 0.0, 0.0]]), atol=1e-6, rtol=0)
    assert observations.wrist_wrench_c(env).abs().sum() > 0  # no tare on repeated reads


def test_hand_observation_reads_actual_joint_state_not_command():
    term, robot = _hand_term()
    robot.data.joint_pos[:] = source_actions.inspire_synergy_to_joint_positions(torch.tensor([[0.82, 0.95], [0.9, 0.85]]))
    env = SimpleNamespace(num_envs=2, action_manager=SimpleNamespace(get_term=lambda name: term))
    torch.testing.assert_close(observations.actual_hand_synergy(env), torch.tensor([[0.82, 0.95], [0.9, 0.85]]))


def test_sensor_reset_mask_zeros_only_reset_rows_and_preserves_live_wrench():
    force = torch.ones(2, 17, 3)
    identity = torch.tensor([1.0, 0.0, 0.0, 0.0]).reshape(1, 1, 4).repeat(2, 2, 1)
    wrench = torch.zeros(2, 2, 6)
    wrench[:, 1, :] = torch.tensor([0.0, 4.0, 0.0, 0.0, 0.0, 2.0])
    robot = SimpleNamespace(body_names=("axia80_link", "inspire_base_link"), data=SimpleNamespace(
        body_incoming_joint_wrench_b=wrench, body_pos_w=torch.zeros(2, 2, 3), body_quat_w=identity
    ))
    env = SimpleNamespace(num_envs=2, device="cpu", sensor_data_fresh=torch.tensor([False, True]), scene={
        "robot": robot,
        "palm_tactile": SimpleNamespace(body_names=sensors.PALM_CHANNEL_NAMES, data=SimpleNamespace(net_forces_w=force)),
    })
    tactile = observations.palm_tactile_bits(env)
    measured = observations.wrist_wrench_c(env)
    torch.testing.assert_close(tactile[0], torch.zeros(17))
    torch.testing.assert_close(tactile[1], torch.ones(17))
    torch.testing.assert_close(measured[0], torch.zeros(6))
    torch.testing.assert_close(measured[1], torch.tensor([0.0, 4.0, 0.0, 0.4, 0.0, 2.0]))
    env.sensor_data_fresh[:] = False
    assert observations.palm_tactile_bits(env).sum() == 0
    assert observations.wrist_wrench_c(env).sum() == 0
    # Once physics refreshes the reset row, the same physical signal is visible.
    env.sensor_data_fresh[0] = True
    torch.testing.assert_close(observations.palm_tactile_bits(env)[0], torch.ones(17))
    torch.testing.assert_close(observations.wrist_wrench_c(env)[0], measured[1])
