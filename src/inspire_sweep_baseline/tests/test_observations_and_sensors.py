"""Actual palm channel, mount wrench transport, and Actor 41D assembly."""

from types import SimpleNamespace

import pytest
import torch


def test_actor_observation_is_41d_and_uses_live_object_not_sensor_rewards(load_mdp, make_env):
    observations = load_mdp("observations")
    env = make_env(3)
    arm = SimpleNamespace(name="robot", joint_ids=[0, 1, 2, 3, 4, 5])
    robot = env.scene["robot"]
    robot.body_names = ["base_link", "axia80_link", "inspire_base_link"]
    robot.data.body_pos_w = torch.zeros(3, 3, 3)
    robot.data.body_quat_w = torch.tensor([1.0, 0.0, 0.0, 0.0]).expand(3, 3, 4).clone()
    robot.data.body_incoming_joint_wrench_b = torch.zeros(3, 3, 6)
    robot.data.body_incoming_joint_wrench_b[:, 2, 0] = 2.0
    env.scene["palm_tactile"] = SimpleNamespace(
        body_names=["inspire_palm_force_sensor"],
        data=SimpleNamespace(net_forces_w=torch.tensor([[[0.05, 0.0, 0.0]], [[0.0, 0.0, 0.0]], [[0.08, 0.0, 0.0]]])),
    )
    env.sensor_data_fresh[2] = False
    robot.data.body_incoming_joint_wrench_b[2, 2] = torch.nan
    prefix = [
        observations.arm_joint_pos_rel(env, arm),
        observations.arm_joint_vel_rel(env, arm),
        observations.last_action(env),
        observations.object_position_in_root(env),
        observations.object_width(env),
        observations.eef_pose_in_root(env),
        observations.goal_position_in_root(env),
    ]
    assert [term.shape[1] for term in prefix] == [6, 6, 8, 3, 1, 7, 3]
    palm = observations.palm_contact_bit(env)
    wrench = observations.wrist_wrench_eef(env)
    assert palm[:, 0].tolist() == [1.0, 0.0, 0.0]
    assert wrench[2].count_nonzero() == 0
    assert wrench[:2, 0].tolist() == [2.0, 2.0]
    actor = torch.cat([*prefix, palm, wrench], dim=-1)
    assert actor.shape == (3, 41)
    assert torch.isfinite(actor).all()
    before = observations.object_position_in_root(env).clone()
    env.scene["object_collection"].data.object_pos_w[0, 0, 1] += 0.02
    assert observations.object_position_in_root(env)[0, 1].item() == pytest.approx(before[0, 1].item() + 0.02)


def test_stale_sensor_rows_are_zero_without_reading_sim_handles(load_mdp, make_env):
    observations = load_mdp("observations")
    env = make_env(3)
    env.sensor_data_fresh[:] = False
    assert torch.equal(observations.palm_contact_bit(env), torch.zeros(3, 1))
    assert torch.equal(observations.wrist_wrench_eef(env), torch.zeros(3, 6))


def test_force_rotation_and_moment_transport_at_eef(load_mdp):
    sensors = load_mdp("sensors")
    wrench = torch.tensor([[2.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    # Sensor X points along EEF Y; sensor lies 10 cm along EEF Z.
    rotation = torch.tensor([[[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]])
    arm = torch.tensor([[0.0, 0.0, 0.1]])
    result = sensors.transform_wrench_f_to_c(wrench, rotation, arm)
    assert torch.allclose(result, torch.tensor([[0.0, 2.0, 0.0, -0.2, 0.0, 0.0]]))


def test_wrench_reader_selects_mount_child_by_name_not_asset_order(load_mdp):
    sensors = load_mdp("sensors")
    robot = SimpleNamespace(
        body_names=["inspire_base_link", "wrist_3_link", "axia80_link"],
        data=SimpleNamespace(body_incoming_joint_wrench_b=torch.zeros(2, 3, 6)),
    )
    robot.data.body_incoming_joint_wrench_b[0, 0] = torch.tensor([1., 2., 3., 4., 5., 6.])
    robot.data.body_incoming_joint_wrench_b[1, 0] = torch.nan
    reader = sensors.FixedJointWrenchReader(robot)
    result = reader.read(valid_env_mask=torch.tensor([True, False]))
    assert torch.equal(result.measured_f[0], torch.tensor([1., 2., 3., 4., 5., 6.]))
    assert result.measured_f[1].count_nonzero() == 0
    with pytest.raises(RuntimeError, match="non-finite"):
        reader.read()


def test_palm_sensor_rejects_multi_pad_aggregation(load_mdp, make_env):
    observations = load_mdp("observations")
    env = make_env(1)
    env.scene["palm_tactile"] = SimpleNamespace(body_names=["inspire_palm_force_sensor", "index_pad"])
    with pytest.raises(RuntimeError, match="only the palm"):
        observations.palm_contact_bit(env)
