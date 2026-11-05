"""Wrench v0_3/v0_4 reward boundaries, state resets and uniform bank coverage."""

import math
from types import SimpleNamespace as NS

import pytest
import torch

from conftest import MODULE_ROOT
from test_final_configuration import configs
from test_omni_configuration import omni_config, _execute_definitions
from test_fusion_holding_osc import osc
from test_revision_layout import bounds
from test_wrench_v0_1 import reward_env


def test_uniform_counts_independent_seeded_partial_and_empty_reset(configs, monkeypatch):
    events = configs('event_functions', 'wrench')
    n = 10000
    counts = torch.full((n, 2), -1)
    calls = []

    def capture(env, ids, gray_obstacles, white_obstacles, **kw):
        counts[ids] = torch.tensor([gray_obstacles, white_obstacles])
        calls.append(kw)

    monkeypatch.setattr(events, 'reset_wrench_scene', capture)
    env = NS(num_envs=n, device='cpu', _revision_layout_rng=torch.Generator().manual_seed(42))
    ids = torch.arange(1, n, 2)
    events.reset_wrench_uniform_scene(env, ids, excluded_x_half_angle=.2)
    expected = torch.randint(0, 5, (len(ids), 2), generator=torch.Generator().manual_seed(42))
    assert torch.equal(counts[ids], expected)
    assert (counts[::2] == -1).all()
    assert set(map(tuple, counts[ids].tolist())) == {(g,w) for g in range(5) for w in range(5)}
    assert all(kw['excluded_x_half_angle'] == .2 for kw in calls)
    snapshot = counts.clone()
    rng = env._revision_layout_rng.get_state().clone()
    events.reset_wrench_uniform_scene(env, [])
    assert torch.equal(counts, snapshot)
    assert torch.equal(env._revision_layout_rng.get_state(), rng)
    env._revision_layout_rng.manual_seed(42)
    events.reset_wrench_uniform_scene(env, ids)
    assert torch.equal(counts, snapshot)
    events.reset_wrench_uniform_scene(env, None)
    assert (counts >= 0).all() and (counts <= 4).all()


def test_uniform_bank_covers_every_count_at_every_inherited_angle(configs):
    events = configs('event_functions', 'wrench')
    stages = tuple((a,g,w) for a in range(0,31,5) for g in range(5) for w in range(5))
    spawn = configs('events', 'wrench').WrenchEventsCfg().object_spawn
    env = NS(num_envs=2, _revision_bounds_t=bounds(2),
             _revision_layout_rng=torch.Generator().manual_seed(9),
             cfg=NS(events=NS(object_spawn=spawn), layout_bank_size=1,
                    layout_build_batch_size=16, red_margin=.023, rear_offset=.02))
    events.initialize_wrench_layouts(env, None, stages=stages)
    for angle, gray, white in stages:
        sample = env._wrench_layout_bank.sample(
            torch.arange(2), env._revision_layout_rng, gray, white, math.radians(angle))
        active, actual = sample[2], sample[4]
        assert (actual <= torch.tensor([gray, white])).all()
        assert torch.equal(active.sum(-1), actual.sum(-1) + 1)


def test_osc_output_rate_uses_joint_bounds_not_policy_inputs_and_resets(configs):
    # Deliberately permuted joint IDs and environment-specific actuator bounds.
    limits = torch.tensor([[1.,2.,4.,8.,16.,32.], [2.,4.,8.,16.,32.,64.]])
    ids = [3,1,5,0,4,2]
    arm = NS(_asset=NS(data=NS(joint_effort_limits=limits)), _joint_ids=ids,
             cfg=NS(effort_limit_scale=.5, minimum_effort_limit=1e-6),
             joint_efforts=limits[:,ids] * .25)
    env = NS(num_envs=2, device='cpu', common_step_counter=1,
             action_manager=NS(action=torch.full((2,8), 100.), prev_action=torch.zeros(2,8),
                               get_term=lambda _: arm))
    env.action_manager.action[:,6:] = torch.tensor([.5, -.5])
    term = configs('reward_functions', 'wrench').WrenchV0_3ActionRatePenalty(NS(params={}), env)
    assert term(env).tolist() == pytest.approx([.03*(6*.25+.5)]*2)
    assert term(env).tolist() == pytest.approx([.06]*2)  # duplicate call cannot advance history
    env.action_manager.prev_action[:] = env.action_manager.action
    env.action_manager.action[:,:6] = -100  # raw Arm jump has no output cost
    env.common_step_counter += 1
    assert not term(env).any()
    arm.joint_efforts *= -1
    env.common_step_counter += 1
    assert term(env).tolist() == pytest.approx([.18]*2)
    term.reset(torch.tensor([0]))
    assert term(env).tolist() == pytest.approx([0.,.18])
    env.common_step_counter += 1
    assert term(env).tolist() == pytest.approx([.045,0.])
    # Relative output cost is independent of physical torque units/ranges.
    limits *= 10
    arm.joint_efforts *= 10
    env.common_step_counter += 1
    assert not term(env).any()
    term.reset()
    arm.joint_efforts = limits[:,ids] * .5
    env.common_step_counter += 1
    assert term(env).tolist() == pytest.approx([.18]*2)
    arm.joint_efforts *= -1
    env.action_manager.action[:,6:] += 100
    env.common_step_counter += 1
    assert term(env).tolist() == pytest.approx([.03*24 + .03*100]*2)


@pytest.mark.parametrize('height,valid', [(.14-1e-8, True), (.14, False), (.14+1e-8, False)])
def test_relaxed_height_boundary_distance_contact(configs, height, valid):
    rev = configs('reward_functions', 'revision')
    env, pos, _ = reward_env()
    env.cfg.goal_reward_height_limit = env.cfg.contact_height_limit = .14
    env.scene['ee_frame'].data.target_pos_w[...,2] = rev.SHELF_SURFACE_HEIGHT + height
    assert bool(rev.goal_reward_height_valid(env).item()) == valid
    assert (configs('reward_functions', 'wrench').goal_distance_cosine(env).item() > 0) == valid
    assert (rev.target_contact_below_height(env).item() > 0) == valid


@pytest.mark.parametrize('dt', [.01, .02, .05])
def test_episode_best_progress_height_gate_first_step_reset_and_no_repayment(configs, dt):
    rev = configs('reward_functions', 'revision')
    env, pos, _ = reward_env(2)
    # Isaac tensors and inherited progress buffers both use float32.
    data = env.scene['object_collection'].data
    data.object_link_state_w = data.object_link_state_w.float()
    pos = data.object_link_state_w[..., :3]
    data.object_pos_w = pos
    env.command_manager.get_command = lambda _: torch.zeros(2,3)
    env.step_dt = dt
    env.cfg.goal_reward_height_limit = .14
    pos[:,0,0] = .1
    term = configs('reward_functions', 'wrench').WrenchV0_4GoalProgress(NS(params={}), env)
    pos[:,0,0] = .08
    assert (term(env)*dt).tolist() == pytest.approx([.02,.02], abs=1e-7)
    assert not term(env).any()
    pos[:,0,0] = .09
    assert not term(env).any()
    pos[:,0,0] = .08
    assert not term(env).any()
    env.scene['ee_frame'].data.target_pos_w[0,0,2] = rev.SHELF_SURFACE_HEIGHT + .14
    pos[:,0,0] = .06
    assert (term(env)*dt).tolist() == pytest.approx([0,.02], abs=1e-7)
    env.scene['ee_frame'].data.target_pos_w[0,0,2] -= .01
    assert not term(env).any()  # cannot cash in the previous high-EEF improvement
    pos[:,0,0] = .04
    term.reset(torch.tensor([0]))
    # Reset snapshot precedes the new command; first transition is included.
    goal = torch.zeros(2,3)
    goal[0,0] = -.02
    env.command_manager.get_command = lambda _: goal
    pos[:,0,0] = .03
    assert (term(env)*dt).tolist() == pytest.approx([.01,.03], abs=1e-7)
    pos[0,0,0] = -.02
    pos[1,0,0] = 0
    assert (term(env)*dt).tolist() == pytest.approx([.05,.03], abs=1e-7)
    assert not term(env).any()
    term.reset()
    assert not term(env).any()


@pytest.mark.parametrize('speed,damping', [(.01,1),(.04-1e-8,1),(.04,1),(.06,math.exp(-1))])
def test_first_entry_free_speed_four_cm(configs, speed, damping):
    env, pos, _ = reward_env()
    env.step_dt = .02
    env._sim_step_counter = 1
    env.cfg.goal_success_radius = .03
    pos[:,0,0] = .02
    env.scene['object_collection'].data.object_lin_vel_w = torch.tensor([[[0.,0.,speed]]])
    cfg = configs('rewards', 'wrench').WrenchV0_3RewardsCfg().target_goal_first_entry_bonus
    term = cfg.func(cfg, env)
    assert (term(env, **cfg.params)*env.step_dt*cfg.weight).item() == pytest.approx(25*damping)
    env.common_step_counter += 1
    assert term(env, **cfg.params).item() == 0


def test_derived_runners_only_change_experiment_name(omni_config):
    cc = omni_config.baseline.configclass

    @cc
    class Parent:
        experiment_name = 'inspire_wrench_v0_2'
        settings = {'gamma': .98, 'steps': 36}

    m = _execute_definitions(MODULE_ROOT/'agent/wrench_ppo_cfg.py',
                            dict(configclass=cc, WrenchV0_2PPORunnerCfg=Parent),
                            {'WrenchV0_3PPORunnerCfg', 'WrenchV0_4PPORunnerCfg'})
    parent = Parent()
    for cls, base, name in [(m.WrenchV0_3PPORunnerCfg, Parent, 'inspire_wrench_v0_3'),
                             (m.WrenchV0_4PPORunnerCfg, m.WrenchV0_3PPORunnerCfg, 'inspire_wrench_v0_4')]:
        child = cls()
        assert cls.__bases__ == (base,)
        assert child.experiment_name == name
        child.experiment_name = parent.experiment_name
        assert child.to_dict() == parent.to_dict()
