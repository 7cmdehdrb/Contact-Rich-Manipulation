"""Numeric trajectory checks for DexTouch-style sweep distance records."""

from types import SimpleNamespace

import pytest
import torch


def _reach(rewards, env, near_radius=0.10):
    return rewards.DexReachProgress(
        SimpleNamespace(params={"near_radius": near_radius, "frame_name": "ee_frame"}), env
    )


@pytest.mark.parametrize("step_dt", [0.01, 0.02, 0.05])
def test_reach_pays_only_extra_record_distance_in_meters(load_mdp, make_env, step_dt):
    rewards = load_mdp("dex_rewards")
    env = make_env(2)
    env.step_dt = step_dt
    position = env.scene["object_collection"].data.object_pos_w[:, 0]
    reference = env.scene["ee_frame"].data.target_pos_w[:, 0]
    # Different reset distances must not change reward for equal improvements.
    reference[:] = position
    reference[:, 0] += torch.tensor([0.6, 1.1])
    term = _reach(rewards, env)
    total = torch.zeros(2)
    for distances, expected in [
        ([0.6, 1.1], [0.0, 0.0]),
        ([0.4, 0.9], [0.2, 0.2]),
        ([0.5, 1.0], [0.0, 0.0]),
        ([0.4, 0.9], [0.0, 0.0]),
        ([0.2, 0.7], [0.2, 0.2]),
        ([0.1, 0.1], [0.1, 0.6]),
        ([0.0, 0.0], [0.0, 0.0]),
    ]:
        reference[:, 0] = position[:, 0] + torch.tensor(distances)
        reward = term(env) * step_dt
        assert reward.tolist() == pytest.approx(expected, abs=2e-7)
        total += reward
    assert total.tolist() == pytest.approx([0.5, 1.0], abs=2e-7)


def test_reach_uses_live_object_and_fixed_eef_not_finger_closure(load_mdp, make_env):
    rewards = load_mdp("dex_rewards")
    env = make_env(1)
    env.step_dt = 0.02
    position = env.scene["object_collection"].data.object_pos_w[:, 0]
    reference = env.scene["ee_frame"].data.target_pos_w[:, 0]
    reference[:] = position + torch.tensor([0.6, 0.0, 0.0])
    term = _reach(rewards, env)
    env.scene["robot"].data.joint_pos[:, 6:] = 1.0
    assert term(env).item() == 0.0
    position[:, 0] += 0.2
    assert (term(env) * env.step_dt).item() == pytest.approx(0.2)
    # Sweeping together at fixed proximity cannot collect approach reward again.
    position[:, 1] += 0.2
    reference[:, 1] += 0.2
    assert term(env).item() == 0.0


@pytest.mark.parametrize("step_dt", [0.01, 0.02, 0.05])
def test_goal_records_are_metric_isotropic_and_ignore_height(load_mdp, make_env, step_dt):
    rewards = load_mdp("dex_rewards")
    env = make_env(3)
    env.step_dt = step_dt
    position = env.scene["object_collection"].data.object_pos_w[:, 0]
    initial = position.clone()
    directions = torch.tensor([[1.0, 0.0], [0.0, -1.0], [-0.6, 0.8]])
    goal = initial.clone()
    goal[:, :2] += directions * torch.tensor([0.15, 0.25, 0.35])[:, None]
    env.command_manager.get_command = lambda name: goal
    term = rewards.DexPlanarGoalProgress(SimpleNamespace(params={}), env)
    for distance, expected in [(0.0, 0.0), (0.03, 0.03), (0.01, 0.0), (0.03, 0.0), (0.07, 0.04)]:
        position[:, :2] = initial[:, :2] + directions * distance
        position[:, 2] += 0.1
        assert (term(env) * step_dt).tolist() == pytest.approx([expected] * 3, abs=2e-7)
    position[:, :2] = goal[:, :2]
    assert (term(env) * step_dt).tolist() == pytest.approx([0.07, 0.17, 0.27], abs=2e-7)
    assert torch.equal(term(env), torch.zeros(3))
    position[:, :2] += directions * 0.10  # Overshoot, then return: no new record.
    assert torch.equal(term(env), torch.zeros(3))
    position[:, :2] = goal[:, :2]
    assert torch.equal(term(env), torch.zeros(3))


def test_partial_reset_clears_only_selected_records_and_pays_first_transition(load_mdp, make_env):
    rewards = load_mdp("dex_rewards")
    env = make_env(3)
    env.step_dt = 0.02
    position = env.scene["object_collection"].data.object_pos_w[:, 0]
    reference = env.scene["ee_frame"].data.target_pos_w[:, 0]
    reference[:] = position + torch.tensor([0.6, 0.0, 0.0])
    goal = position.clone() + torch.tensor([0.15, 0.0, 0.0])
    env.command_manager.get_command = lambda name: goal
    reach = _reach(rewards, env)
    sweep = rewards.DexPlanarGoalProgress(SimpleNamespace(params={}), env)
    position[:, 0] += 0.03
    assert (reach(env) * env.step_dt).tolist() == pytest.approx([0.03] * 3, abs=1e-7)
    assert (sweep(env) * env.step_dt).tolist() == pytest.approx([0.03] * 3, abs=1e-7)

    position[1] = torch.tensor([3.0, 4.0, 1.05])
    reference[1] = position[1] + torch.tensor([0.6, 0.0, 0.0])
    # Reward reset runs before CommandManager.reset: reading it here is a bug.
    def stale_command(name):
        raise AssertionError("reset must not read a previous episode command")

    env.command_manager.get_command = stale_command
    reach.reset(torch.tensor([1]))
    sweep.reset(torch.tensor([1]))
    goal[1] = position[1] + torch.tensor([0.15, 0.0, 0.0])
    env.command_manager.get_command = lambda name: goal
    position[:, 0] += 0.02
    assert (reach(env) * env.step_dt).tolist() == pytest.approx([0.02] * 3, abs=3e-7)
    assert (sweep(env) * env.step_dt).tolist() == pytest.approx([0.02] * 3, abs=3e-7)
    # Unselected rows retain their old record even after a retreat/revisit.
    for offset in (-0.02, 0.02):
        position[[0, 2], 0] += offset
        assert torch.allclose(reach(env), torch.zeros(3), atol=1e-5)
        assert torch.allclose(sweep(env), torch.zeros(3), atol=1e-5)


@pytest.mark.parametrize("reset_ids", [None, slice(None), torch.tensor([], dtype=torch.long)])
def test_reset_selections_and_initially_satisfied_gaps(load_mdp, make_env, reset_ids):
    rewards = load_mdp("dex_rewards")
    env = make_env(2)
    env.step_dt = 0.02
    position = env.scene["object_collection"].data.object_pos_w[:, 0]
    initial = position.clone()
    reference = env.scene["ee_frame"].data.target_pos_w[:, 0]
    reference[:] = initial
    goal = initial.clone()
    env.command_manager.get_command = lambda name: goal
    reach = _reach(rewards, env)
    sweep = rewards.DexPlanarGoalProgress(SimpleNamespace(params={}), env)
    for term in (reach, sweep):
        term.reset(reset_ids)
    for offset in (0.0, 0.2, 0.0):
        position[:, 0] = initial[:, 0] + offset
        for term in (reach, sweep):
            assert torch.equal(term(env), torch.zeros(2))


def test_reach_reset_refreshes_kinematics_and_keeps_environment_records_independent(load_mdp, make_env):
    rewards = load_mdp("dex_rewards")
    env = make_env(2)
    env.step_dt = 0.02
    position = env.scene["object_collection"].data.object_pos_w[:, 0]
    reference = env.scene["ee_frame"].data.target_pos_w[:, 0]
    reference[:] = 99.0

    class RobotData:
        @property
        def body_link_pos_w(self):
            reference[:] = position + torch.tensor([0.6, 0.0, 0.0])
            return reference

    env.scene["robot"].data = RobotData()
    term = _reach(rewards, env)
    reference[0, 0] -= 0.2
    assert (term(env) * env.step_dt).tolist() == pytest.approx([0.2, 0.0])
    reference[1, 0] -= 0.1
    assert (term(env) * env.step_dt).tolist() == pytest.approx([0.0, 0.1])


@pytest.mark.parametrize("invalid", [-0.01, float("inf"), float("nan")])
def test_invalid_distance_thresholds_fail_explicitly(load_mdp, make_env, invalid):
    rewards = load_mdp("dex_rewards")
    env = make_env(1)
    with pytest.raises(ValueError, match="near_radius"):
        _reach(rewards, env, near_radius=invalid)
    term = rewards.DexPlanarGoalProgress(SimpleNamespace(params={}), env)
    with pytest.raises(ValueError, match="position_tolerance"):
        term(env, position_tolerance=invalid)
