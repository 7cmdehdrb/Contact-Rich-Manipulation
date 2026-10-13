"""Behavioral checks for the independently owned baseline task."""

from types import SimpleNamespace

import pytest
import torch


@pytest.mark.parametrize("direction, expected", [("left", -0.18), ("right", 0.18)])
def test_single_object_event_preserves_z_and_only_selected_rows(load_mdp, make_env, direction, expected):
    events = load_mdp("events")
    env = make_env(3)
    env.scene.env_origins[2] = torch.tensor([2.0, 3.0, 4.0])
    before = env.scene["object_collection"].data.object_link_state_w.clone()
    events.randomize_single_object_scene(env, torch.tensor([2]), sweep_direction=direction)
    state = env.scene["object_collection"].data.object_link_state_w
    assert torch.equal(state[:2], before[:2])
    assert state[2, 0, 2].item() == pytest.approx(5.05)
    assert abs((state[2, 0, 0] - 2.0).item() + 0.75) <= 0.020001
    assert abs((state[2, 0, 1] - 3.0).item()) <= 0.020001
    assert state[2, 0, 7:].count_nonzero() == 0
    assert torch.linalg.vector_norm(state[2, 0, 3:7]).item() == pytest.approx(1.0)
    assert env.sweep_dir[2].tolist() == pytest.approx([0.0, expected, 0.0])
    assert env.target_id[2, 0] == 0
    assert env.target_width[2, 0].item() == pytest.approx(0.04)


def test_default_event_samples_both_directions_without_extra_objects(load_mdp, make_env):
    events = load_mdp("events")
    torch.manual_seed(42)
    env = make_env(512)
    events.randomize_single_object_scene(env, torch.arange(512))
    assert env.scene["object_collection"].data.object_link_state_w.shape == (512, 1, 13)
    assert set(torch.sign(env.sweep_dir[:, 1]).tolist()) == {-1.0, 1.0}
    with pytest.raises(ValueError, match="sweep_direction"):
        events.randomize_single_object_scene(env, [0], sweep_direction="unknown")


def test_reach_and_push_use_bottom_plus_nine_cm_and_same_approach(load_mdp, make_env):
    rewards = load_mdp("rewards")
    env = make_env(2)
    env.sweep_dir[1, 1] *= -1.0
    object_bottom = env.scene["object_collection"].data.object_pos_w[:, 0].clone()
    expected = object_bottom + torch.tensor([-0.02, 0.0, 0.09])
    expected[:, 1] = torch.tensor([-0.04, 0.04])
    env.scene["ee_frame"].data.target_pos_w[:, 0] = expected
    env.scene["wrist_frame"].data.target_pos_w[:, 0] = expected
    assert torch.allclose(rewards.reaching_target(env), torch.ones(2))
    # At half of the required sweep distance, the matching EEF gate permits shaping.
    env.command_manager = SimpleNamespace(get_command=lambda name: object_bottom + torch.tensor([0.0, 0.09, 0.0]))
    assert torch.allclose(rewards.pushing_target(env), torch.full((2,), 0.5), atol=1e-6)
    env.scene["ee_frame"].data.target_pos_w[:, 0, 2] -= 0.05
    assert torch.allclose(rewards.reaching_target(env), torch.full((2,), torch.exp(torch.tensor(-0.5))), atol=1e-6)
    assert torch.equal(rewards.pushing_target(env), torch.zeros(2))


def test_one_rad_cutoff_checks_arm_and_excludes_fast_fingers(load_mdp, make_env):
    terms = load_mdp("terminations")
    env = make_env(4)
    arm_ids = [1, 4, 7, 10, 13, 16]
    cfg = SimpleNamespace(name="robot", joint_ids=arm_ids)
    env.scene["robot"].data.joint_vel[:, 0] = 100.0
    env.scene["robot"].data.joint_vel[1, 7] = 1.0
    env.scene["robot"].data.joint_vel[2, 10] = 1.001
    env.scene["robot"].data.joint_vel[3, 16] = -1.001
    assert terms.arm_velocity(env, asset_cfg=cfg).tolist() == [False, False, True, True]


def test_hand_channels_independent_clamp_and_subset_open_reset(load_mdp, make_env):
    actions = load_mdp("actions")
    env = make_env(3)

    class Robot:
        def find_joints(self, names, preserve_order):
            assert preserve_order is True
            return list(range(12)), names

        def set_joint_position_target(self, targets, joint_ids, env_ids=None):
            self.last = (targets.clone(), joint_ids, env_ids)

    env.scene["robot"] = Robot()
    term = actions.HandOpennessAction(SimpleNamespace(asset_name="robot"), env)
    assert term.action_dim == 2
    assert torch.equal(term.raw_actions, torch.ones(3, 2))
    term.process_actions(torch.tensor([[1.5, -0.5], [0.0, 1.0], [0.0, 0.0]]))
    term.apply_actions()
    targets = env.scene["robot"].last[0]
    assert targets[0, 0].item() == pytest.approx(1.1641)
    assert targets[0, 1:].count_nonzero() == 0
    assert targets[1, 0] == 0.0
    assert targets[1, 1:].count_nonzero() == 11
    term.reset(torch.tensor([2]))
    assert torch.equal(term.raw_actions[2], torch.ones(2))
    assert torch.equal(term.processed_actions[2], torch.ones(2))
    assert torch.equal(term.raw_actions[0], torch.tensor([1.5, -0.5]))
    assert env.scene["robot"].last[0].count_nonzero() == 0


def test_goal_stays_fixed_when_object_moves_and_resamples_only_reset_rows(load_mdp, make_env):
    commands = load_mdp("commands")
    env = make_env(3)
    command = commands.SweepGoalCommand(SimpleNamespace(asset_name="object_collection"), env)
    original = command.command.clone()
    env.scene["object_collection"].data.object_pos_w[:, 0, 1] += 0.07
    command._update_command()
    assert torch.equal(command.command, original)
    command._resample_command(torch.tensor([1]))
    assert torch.equal(command.command[[0, 2]], original[[0, 2]])
    assert command.command[1, 1].item() == pytest.approx(original[1, 1].item() + 0.07)


def test_alignment_uses_physical_columns_and_penalizes_inverted_axis(load_mdp, make_env, tensor_math):
    rewards = load_mdp("rewards")
    env = make_env(3)
    zeros = torch.zeros(3)
    angles = torch.tensor([torch.pi / 2, -torch.pi / 2, 0.0])
    env.scene["ee_frame"].data.target_quat_w[:, 0] = tensor_math.quat_from_euler_xyz(angles, zeros, zeros)
    assert torch.allclose(rewards.align_eef_y_with_shelf_z(env), torch.tensor([1.0, -1.0, 0.0]), atol=1e-6)
