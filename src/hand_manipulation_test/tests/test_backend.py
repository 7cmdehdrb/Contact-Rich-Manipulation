"""Pure-torch checks for the physical sensor and control-frame contracts."""

from pathlib import Path
from types import SimpleNamespace
import sys

import pytest
import torch


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hand_manipulation_test.action_math import (
    com_to_target_offset,
    compose_fixed_offset_pose,
    current_frame_delta_target,
    inspire_joint_positions_to_synergy,
    inspire_synergy_to_joint_positions,
    quaternion_from_rotation_vector,
    quaternion_multiply,
    shift_jacobian_to_point,
    shift_twist_to_point,
)
from hand_manipulation_test.geometry import base_link_pose_w, control_point_pose_w, palm_tactile_bits, wrist_wrench_c
from hand_manipulation_test.sensors import (
    FixedJointWrenchReader,
    PALM_CHANNEL_NAMES,
    PalmTactileReader,
    contact_magnitudes_and_bits,
    transform_wrench_f_to_c,
)


def test_current_frame_translation_and_rotation_follow_measured_orientation():
    dtype = torch.float64
    q = quaternion_from_rotation_vector(torch.tensor([[0.0, 0.0, torch.pi / 2]], dtype=dtype))
    delta_rotation = torch.tensor([[0.2, 0.0, 0.0]], dtype=dtype)
    position, quaternion = current_frame_delta_target(
        torch.tensor([[1.0, 2.0, 3.0]], dtype=dtype),
        q,
        torch.tensor([[0.1, 0.0, 0.0]], dtype=dtype),
        delta_rotation,
    )
    torch.testing.assert_close(position, torch.tensor([[1.0, 2.1, 3.0]], dtype=dtype))
    torch.testing.assert_close(quaternion, quaternion_multiply(q, quaternion_from_rotation_vector(delta_rotation)))


def test_zero_rotation_and_half_open_synergy_are_finite_and_exact():
    torch.testing.assert_close(
        quaternion_from_rotation_vector(torch.zeros(4, 3)),
        torch.tensor([[1.0, 0.0, 0.0, 0.0]]).expand(4, -1),
    )
    synergy = torch.tensor([[0.0, 1.0], [0.5, 0.5], [1.0, 0.0]])
    joints = inspire_synergy_to_joint_positions(synergy)
    torch.testing.assert_close(inspire_joint_positions_to_synergy(joints), synergy)
    assert joints.shape == (3, 12)
    torch.testing.assert_close(joints[1, :2], torch.tensor([0.68205, 0.2932]))


def test_com_shift_matches_control_point_velocity_and_finite_difference():
    dtype = torch.float64
    theta = torch.tensor(0.37, dtype=dtype)
    offset = torch.tensor([[0.5, 0.1, 0.0]], dtype=dtype)

    def point(angle):
        q = quaternion_from_rotation_vector(torch.stack((angle * 0, angle * 0, angle)).reshape(1, 3))
        return compose_fixed_offset_pose(torch.zeros(1, 3, dtype=dtype), q, offset, q.new_tensor([[1, 0, 0, 0]]))

    _, _, r_hc = point(theta)
    jacobian_com = torch.zeros(1, 6, 1, dtype=dtype)
    jacobian_com[:, 5, 0] = 1.0
    jacobian_c = shift_jacobian_to_point(jacobian_com, r_hc)
    epsilon = 1.0e-6
    numeric = (point(theta + epsilon)[0] - point(theta - epsilon)[0]) / (2 * epsilon)
    torch.testing.assert_close(jacobian_c[:, :3, 0], numeric, rtol=1e-7, atol=1e-9)
    torch.testing.assert_close(
        shift_twist_to_point(torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 1.0]], dtype=dtype), r_hc),
        jacobian_c[:, :, 0],
    )
    q = quaternion_from_rotation_vector(torch.tensor([[0.0, 0.0, torch.pi / 2]], dtype=dtype))
    shifted = com_to_target_offset(q, torch.tensor([[0.2, 0.0, 0.0]], dtype=dtype), offset)
    torch.testing.assert_close(shifted, torch.tensor([[0.5, -0.1, 0.0]], dtype=dtype))


def test_palm_reader_reorders_all_seventeen_channels_and_includes_threshold():
    names = tuple(reversed(PALM_CHANNEL_NAMES))
    forces = torch.zeros(2, 17, 3)
    for index, name in enumerate(names):
        forces[:, index, 0] = PALM_CHANNEL_NAMES.index(name) / 100.0
    reader = PalmTactileReader(
        SimpleNamespace(body_names=names, data=SimpleNamespace(net_forces_w=forces)),
        threshold_n=0.1,
    )
    sample = reader.read()
    torch.testing.assert_close(sample.magnitudes_n[0], torch.arange(17) / 100.0)
    assert sample.bits.shape == (2, 17)
    assert sample.bits[0].tolist() == [0] * 10 + [1] * 7
    _, bits = contact_magnitudes_and_bits(torch.tensor([[[0.03, 0.04, 0.0]]]), 0.05)
    assert bits.item() == 1


def test_palm_reader_rejects_missing_channels_and_invalid_samples():
    bad_names = (*PALM_CHANNEL_NAMES[:-1], "unknown")
    with pytest.raises(RuntimeError):
        PalmTactileReader(SimpleNamespace(body_names=bad_names))
    with pytest.raises(ValueError):
        contact_magnitudes_and_bits(torch.zeros(1, 17, 3), 0.0)
    with pytest.raises(RuntimeError):
        contact_magnitudes_and_bits(torch.full((1, 17, 3), float("nan")), 0.05)


def test_wrench_rotation_includes_moment_arm_and_preserves_raw_self_weight():
    incoming = torch.zeros(2, 3, 6)
    incoming[:, 2] = torch.tensor([1.0, 2.0, 13.0, 0.1, 0.2, 0.3])
    robot = SimpleNamespace(
        body_names=["base", "axia80_link", "inspire_base_link"],
        data=SimpleNamespace(body_incoming_joint_wrench_b=incoming),
    )
    sample = FixedJointWrenchReader(robot, "inspire_base_link").read(torch.eye(3), torch.zeros(3))
    torch.testing.assert_close(sample.raw_f, incoming[:, 2])
    torch.testing.assert_close(sample.measured_f, incoming[:, 2])
    torch.testing.assert_close(sample.measured_c, incoming[:, 2])
    rotation = torch.tensor([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    shifted = transform_wrench_f_to_c(
        torch.tensor([[1.0, 0.0, 0.0, 0.0, 2.0, 0.0]]), rotation, torch.tensor([1.0, 0.0, 0.0])
    )
    torch.testing.assert_close(shifted, torch.tensor([[0.0, 1.0, 0.0, -2.0, 0.0, 1.0]]))


def test_geometry_composes_configured_frames_and_exposes_unmasked_sensor_data():
    names = ["axia80_link", "inspire_base_link"]
    positions = torch.tensor([[[0.0, 0.0, -0.1], [0.0, 0.0, 0.0]]])
    quaternions = torch.tensor([[[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]])
    incoming = torch.zeros(1, 2, 6)
    incoming[:, 1, 2] = 12.0
    robot = SimpleNamespace(
        body_names=names,
        find_bodies=lambda name, preserve_order=True: ([names.index(name)], [name]),
        data=SimpleNamespace(
            body_link_pos_w=positions,
            body_link_quat_w=quaternions,
            body_incoming_joint_wrench_b=incoming,
        ),
    )
    palm = SimpleNamespace(body_names=PALM_CHANNEL_NAMES, data=SimpleNamespace(net_forces_w=torch.ones(1, 17, 3)))
    env = SimpleNamespace(
        scene={"robot": robot, "palm_tactile": palm},
        cfg=SimpleNamespace(
            actions=SimpleNamespace(arm_action=SimpleNamespace(
                asset_name="robot", body_name="inspire_base_link",
                body_offset_pos=(0.0, 0.05, 0.1), body_offset_quat=(1.0, 0.0, 0.0, 0.0),
            )),
            task=SimpleNamespace(tactile_threshold_n=0.05),
        ),
        episode_length_buf=torch.zeros(1, dtype=torch.long),
    )
    position, _ = control_point_pose_w(env)
    torch.testing.assert_close(position, torch.tensor([[0.0, 0.05, 0.1]]))
    sample = wrist_wrench_c(env)
    torch.testing.assert_close(sample.measured_c, torch.tensor([[0.0, 0.0, 12.0, -0.6, 0.0, 0.0]]))
    torch.testing.assert_close(palm_tactile_bits(env), torch.ones(1, 17))


def test_base_link_pose_reads_actual_named_body_and_tracks_live_pose():
    names = ["inspire_base_link", "base_link", "axia80_link"]
    positions = torch.tensor(
        (((9.0, 8.0, 7.0), (3.0, -2.0, 0.79505), (6.0, 5.0, 4.0)),)
    )
    quaternions = torch.tensor(
        (((1.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0), (1.0, 0.0, 0.0, 0.0)),)
    )
    robot = SimpleNamespace(
        find_bodies=lambda name, preserve_order=True: ([names.index(name)], [name]),
        data=SimpleNamespace(
            body_link_pos_w=positions, body_link_quat_w=quaternions,
            # A root asset pose can differ from the named link's pose.
            root_link_pos_w=torch.tensor(((100.0, 200.0, 300.0),)),
            root_link_quat_w=torch.tensor(((1.0, 0.0, 0.0, 0.0),)),
        ),
    )
    env = SimpleNamespace(
        scene={"robot": robot},
        cfg=SimpleNamespace(actions=SimpleNamespace(arm_action=SimpleNamespace(asset_name="robot"))),
    )
    position, quaternion = base_link_pose_w(env)
    torch.testing.assert_close(position, positions[:, 1], atol=0, rtol=0)
    torch.testing.assert_close(quaternion, quaternions[:, 1], atol=0, rtol=0)
    positions[:, 1, 0] += 0.5
    updated_position, _ = base_link_pose_w(env)
    torch.testing.assert_close(updated_position, torch.tensor(((3.5, -2.0, 0.79505),)), atol=0, rtol=0)
