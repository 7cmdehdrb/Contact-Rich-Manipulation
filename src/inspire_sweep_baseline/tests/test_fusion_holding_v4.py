"""Uncapped center reward and goal-triggered, non-repeatable reach credit."""

import importlib
from types import SimpleNamespace as NS

import pytest
import torch

from test_fusion_holding_v0 import environment


@pytest.fixture
def rewards(load_mdp):
    return importlib.import_module(
        "_baseline_cpu_pkg.mdp.reward_functions.contact_sweep_clutter_fusion_holding_scheduled"
    )


@pytest.mark.parametrize("radius", [.02, .01, .005])
def test_center_doubles_inner_boundary_and_no_time_cap(rewards, radius):
    env = environment()
    pos = env.scene["object_collection"].data.object_pos_w
    pos[:, 0, 0] = torch.tensor([0., radius / 2, radius, 2 * radius])
    old = rewards.goal_settling(env, position_tolerance=radius)
    new = rewards.goal_settling_centered(env, position_tolerance=radius)
    assert new.tolist() == pytest.approx([2., old[1].item() * 1.25, .5, .0625])
    for _ in range(200):  # Beyond v3's 150 paid steps, with no state or reset.
        torch.testing.assert_close(rewards.goal_settling_centered(env, radius), new)
    env.scene["object_collection"].data.object_lin_vel_w[:, 0, 2] = .05
    torch.testing.assert_close(rewards.goal_settling_centered(env, radius), new * .5)
    pos[:, 0, 2] = 1.2
    assert not rewards.goal_settling_centered(env, radius).any()


@pytest.mark.parametrize("radius", [.02, .01, .005])
def test_outer_reward_reaches_inclusive_2r_without_expanding_old_reward(rewards, radius):
    env = environment()
    pos = env.scene["object_collection"].data.object_pos_w
    distances = torch.tensor([radius + 1e-5, 1.5 * radius, 2 * radius, 2 * radius + 1e-5])
    pos[:, 0, 0] = distances
    expected = torch.exp(-torch.log(torch.tensor(2.)) * (distances / radius).square())
    expected[-1] = 0.
    torch.testing.assert_close(rewards.goal_settling_centered(env, radius), expected)
    assert not rewards.goal_settling(env, radius).any()  # v2/v3 keep their r cutoff.
    pos[:, 0, 2] = 1.2
    assert not rewards.goal_settling_centered(env, radius).any()


def reach_env():
    env = environment()
    data = env.scene["object_collection"].data
    state = torch.zeros(4, 1, 13)
    state[..., :3] = data.object_pos_w
    state[..., 3] = 1.
    data.object_link_state_w = state
    data.object_pos_w = state[..., :3]
    data.object_pos_w[:, 0, 0] = .03  # Initially outside goal.
    env.scene["robot"] = NS(data=NS())
    env.scene["ee_frame"] = NS(data=NS(target_pos_w=data.object_pos_w.clone()))
    env.scene["ee_frame"].data.target_pos_w[:, 0, 0] += .6
    goal = NS(params={"position_tolerance": .02})
    env.reward_manager = NS(get_term_cfg=lambda _: goal)
    env.scene["whole_hand_contact"] = NS(data=NS(net_forces_w=torch.zeros(4, 1, 3)))
    env.sensor_data_fresh = torch.ones(4, dtype=torch.bool)
    return env, goal


def test_reach_matches_old_until_goal_then_accepts_distant_hand_without_repeat(rewards):
    env, goal = reach_env()
    old = rewards.DexReachProgress(NS(params={}), env)
    new = rewards.ReachingOrGoal(NS(params={}), env)
    frame = env.scene["ee_frame"].data.target_pos_w
    pos = env.scene["object_collection"].data.object_pos_w
    for distance in [.6, .4, .5, .4]:
        frame[:, 0] = pos[:, 0]
        frame[:, 0, 0] += distance
        torch.testing.assert_close(new(env), old(env))
    # Enter goal without changing hand/object distance. Trigger consumes the
    # remaining approach record once, despite the hand being 40 cm away.
    pos[:, 0, 0] = .01
    frame[:, 0] = pos[:, 0]
    frame[:, 0, 0] += .4
    assert (new(env) * env.step_dt).tolist() == pytest.approx([.3] * 4)
    assert not new(env).any()
    for x in [.03, .01, .03, 0.]:
        pos[:, 0, 0] = x
        frame[:, 0] = pos[:, 0]
        frame[:, 0, 0] += .4
        assert not new(env).any()
    # Only reset environments regain a distance record.
    pos[:, 0, 0] = .03
    new.reset([1, 3])
    pos[:, 0, 0] = 0.
    assert (new(env) > 0).tolist() == [False, True, False, True]


def test_reach_trigger_uses_live_curriculum_validity_and_contact_goal_boundary(rewards):
    env, goal = reach_env()
    pos = env.scene["object_collection"].data.object_pos_w
    pos[:, 0, 0] = torch.tensor([.02, .02001, .01, 0.])
    pos[3, 0, 2] = 1.2
    frame = env.scene["ee_frame"].data.target_pos_w
    frame[:, 0] = pos[:, 0]
    frame[:, 0, 0] += .6
    term = rewards.ReachingOrGoal(NS(params={}), env)
    assert (term(env) > 0).tolist() == [True, False, True, False]
    assert rewards.hand_contact_or_goal(env).tolist() == [1., 0., 1., 0.]
    # Rewind/tighten the radius without rebuilding the reward term.
    goal.params["position_tolerance"] = .005
    term.reset()
    assert not term(env).any()
    assert not rewards.hand_contact_or_goal(env).any()
    goal.params["position_tolerance"] = .02
    assert (term(env) > 0).tolist() == [True, False, True, False]


@pytest.mark.parametrize("radius", [0., -1., float("nan"), float("inf")])
def test_invalid_radius_rejected(rewards, radius):
    env, goal = reach_env()
    with pytest.raises(ValueError):
        rewards.goal_settling_centered(env, position_tolerance=radius)
    term = rewards.ReachingOrGoal(NS(params={}), env)
    with pytest.raises(ValueError):
        term(env, goal_near_radius=radius)
