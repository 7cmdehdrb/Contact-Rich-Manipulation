"""CPU checks for dynamic tool cancellation, frames and reset isolation."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch


@pytest.fixture
def wrench_module(load_mdp, monkeypatch):
    from _source_units import import_unit
    return import_unit("contact_sweep_clutter_v2.wrench")


def make_robot(module, count=2):
    # Shuffle link order and include an upstream housing with deliberately huge
    # mass. Neither ordering nor upstream gravity may affect hand cancellation.
    names = ["axia80_link", *reversed(module.HAND_CONTACT_BODY_NAMES), "wrist_3_link"]
    bodies = len(names)
    masses = torch.full((count, bodies), 0.01)
    masses[:, 0] = 99.0
    masses[:, -1] = 88.0
    inertias = torch.eye(3).expand(count, bodies, 3, 3).clone() * 0.001
    positions = torch.zeros(count, bodies, 3)
    positions[:, :, 0] = 0.1
    quats = torch.zeros(count, bodies, 4)
    quats[:, :, 0] = 1.0
    robot = SimpleNamespace(
        body_names=names,
        device="cpu",
        root_physx_view=SimpleNamespace(
            get_masses=lambda: masses,
            get_inertias=lambda: inertias.flatten(-2),
        ),
        data=SimpleNamespace(
            body_link_pos_w=positions.clone(),
            body_link_quat_w=quats,
            body_com_pos_w=positions.clone(),
            body_com_vel_w=torch.zeros(count, bodies, 6),
            body_com_acc_w=torch.zeros(count, bodies, 6),
            body_incoming_joint_wrench_b=torch.zeros(count, bodies, 6),
        ),
    )
    return robot


def test_newton_euler_includes_gravity_acceleration_gyro_and_moment_arm(wrench_module):
    m = wrench_module
    # One body's link X/Y axes are rotated 90 deg in world: diag(1,2,3)
    # becomes diag(2,1,3). omega=(1,2,3) => omega x Iomega = (12,-3,-2).
    masses = torch.tensor([[2.0]])
    inertia = torch.diag(torch.tensor([1.0, 2.0, 3.0]))[None, None]
    half = 2.0 ** -0.5
    quat = torch.tensor([[[half, 0.0, 0.0, half]]])
    pos = torch.tensor([[[1.0, 0.0, 0.0]]])
    vel = torch.tensor([[[0.0, 0.0, 0.0, 1.0, 2.0, 3.0]]])
    acc = torch.tensor([[[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]]])
    result = m.support_wrench_world(
        masses, inertia, quat, pos, vel, acc, torch.zeros(1, 3), torch.tensor([0.0, 0.0, -10.0]),
    )
    # F=(2,4,26), Ialpha=(8,5,18), r x F=(0,-26,4).
    assert torch.allclose(result, torch.tensor([[2.0, 4.0, 26.0, 20.0, -24.0, 20.0]]), atol=1e-5)


def test_static_downstream_gravity_cancels_without_tare(wrench_module):
    robot = make_robot(wrench_module)
    reader = wrench_module.PureWrenchReader(robot)
    hand_mass = 0.01 * len(wrench_module.HAND_CONTACT_BODY_NAMES)
    robot.data.body_incoming_joint_wrench_b[:, reader.child_body_index, 2] = hand_mass * 9.81
    sample = reader.read(torch.zeros(2, 3), robot.data.body_link_quat_w[:, 0])
    assert sample.raw_world[0, 2].item() == pytest.approx(hand_mass * 9.81)
    assert sample.raw_world[0, 4].item() == pytest.approx(-0.1 * hand_mass * 9.81)
    assert torch.allclose(sample.pure_eef, torch.zeros(2, 6), atol=1e-6)
    assert len(reader.body_ids) == 30  # thirteen links plus seventeen tactile pads


def test_external_load_is_preserved_with_joint_origin_transport(wrench_module):
    robot = make_robot(wrench_module)
    reader = wrench_module.PureWrenchReader(robot)
    child = reader.child_body_index
    # The housing origin differs from the joint/child origin. Using the housing
    # would silently introduce a 0.0127 m moment-transport error.
    robot.data.body_link_pos_w[:, 0, 2] = -0.0127
    hand_weight = 0.01 * len(reader.body_ids) * 9.81
    robot.data.body_incoming_joint_wrench_b[:, child, 2] = hand_weight
    # +5 N external Y load on the hand yields -5 N sensor reaction, together
    # with a -0.3 Nm applied Z torque.
    robot.data.body_incoming_joint_wrench_b[:, child, 1] = -5.0
    robot.data.body_incoming_joint_wrench_b[:, child, 5] = -0.3
    half = 2.0 ** -0.5
    eef_quat = torch.tensor([[half, 0.0, 0.0, half]]).expand(2, -1)
    sample = reader.read(torch.zeros(2, 3), eef_quat)
    assert torch.allclose(sample.pure_world, torch.tensor([[0., -5., 0., 0., 0., -0.8]]).expand(2, -1), atol=1e-6)
    assert torch.allclose(sample.pure_eef, torch.tensor([[-5., 0., 0., 0., 0., -0.8]]).expand(2, -1), atol=1e-6)


def test_accelerating_and_rotating_hand_cancels_same_step_load(wrench_module):
    robot = make_robot(wrench_module)
    reader = wrench_module.PureWrenchReader(robot, gravity_w=(0.0, 0.0, -10.0))
    count = len(reader.body_ids)
    # Per-body F = 0.01*(2,3,14); isotropic I makes gyro term zero.
    # Alpha=(5,6,7), hence per-body COM moment=(.005,.006,.007).
    robot.data.body_com_acc_w[:, :, :] = torch.tensor([2., 3., 4., 5., 6., 7.])
    robot.data.body_com_vel_w[:, :, 3:] = torch.tensor([8., 9., 10.])
    raw_at_joint = count * torch.tensor([0.02, 0.03, 0.14, 0.005, 0.006, 0.007])
    robot.data.body_incoming_joint_wrench_b[:, reader.child_body_index] = raw_at_joint
    sample = reader.read(torch.zeros(2, 3), robot.data.body_link_quat_w[:, 0])
    assert torch.allclose(sample.pure_world, torch.zeros(2, 6), atol=1e-6)


def test_partial_reset_masks_bad_state_without_affecting_other_environments(wrench_module):
    robot = make_robot(wrench_module)
    reader = wrench_module.PureWrenchReader(robot)
    hand_weight = 0.01 * len(reader.body_ids) * 9.81
    robot.data.body_incoming_joint_wrench_b[:, reader.child_body_index, 2] = hand_weight
    robot.data.body_incoming_joint_wrench_b[0] = torch.nan
    robot.data.body_com_acc_w[0] = torch.nan
    robot.data.body_com_pos_w[0] = torch.nan
    robot.data.body_link_quat_w[0] = torch.nan
    sample = reader.read(
        torch.zeros(2, 3), robot.data.body_link_quat_w[:, 0],
        valid_env_mask=torch.tensor([False, True]),
    )
    for value in vars(sample).values():
        assert torch.isfinite(value).all()
        assert value[0].count_nonzero() == 0
    assert torch.allclose(sample.pure_world[1], torch.zeros(6), atol=1e-6)
    with pytest.raises(RuntimeError, match="non-finite"):
        reader.read(torch.zeros(2, 3), robot.data.body_link_quat_w[:, 0])


def test_all_reset_rows_skip_simulation_state_reads(wrench_module):
    robot = make_robot(wrench_module)
    reader = wrench_module.PureWrenchReader(robot)
    robot.data = SimpleNamespace()
    sample = reader.read(
        torch.full((2, 3), torch.nan), torch.full((2, 4), torch.nan),
        valid_env_mask=torch.zeros(2, dtype=torch.bool),
    )
    assert all(torch.equal(value, torch.zeros(2, 6)) for value in vars(sample).values())


def test_missing_downstream_pad_is_rejected(wrench_module):
    robot = make_robot(wrench_module)
    robot.body_names.remove("inspire_palm_force_sensor")
    with pytest.raises(RuntimeError, match="downstream"):
        wrench_module.PureWrenchReader(robot)
