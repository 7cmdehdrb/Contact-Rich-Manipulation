"""Revision-v1 contract checks using production tensor functions and manager cfgs."""

import importlib
import math
from types import SimpleNamespace as NS

import pytest
import torch

from test_final_configuration import configs
from test_omni_configuration import omni_config
from test_fusion_holding_osc import osc


@pytest.fixture
def functions(load_mdp):
    return importlib.import_module('_baseline_cpu_pkg.mdp.reward_functions.revision')


def test_reward_and_curriculum_contract(configs):
    module = configs('rewards', 'revision')
    parent = module.RevisionRewardsCfg()
    before = parent.to_dict()
    child = module.RevisionV1RewardsCfg()
    names = [k for k, v in vars(child).items() if hasattr(v, 'weight')]
    assert names == [
        'target_reaching', 'target_goal_distance', 'target_goal_hold', 'target_contact',
        'robot_action_rate_penalty', 'robot_joint_velocity_penalty', 'object_tilt_penalty',
        'obstacle_displacement_penalty', 'object_acceleration_penalty', 'shelf_penalty',
        'obstacle_contact_penalty', 'target_goal_first_entry_bonus', 'target_goal_timeout_bonus',
        'wrist_wrench_change_penalty', 'target_speed_penalty',
    ]
    assert [getattr(child, n).weight for n in names] == [30, 3.7, 8, 1.45, -1, -1, -30, -150, -.05, -1, -.05, 30, 40, -.1, -5]
    unchanged = ('target_reaching', 'target_goal_distance', 'target_contact',
                 'robot_action_rate_penalty', 'robot_joint_velocity_penalty', 'shelf_penalty', 'target_speed_penalty')
    for name in unchanged:
        assert getattr(child, name).to_dict() == getattr(parent, name).to_dict()
    assert module.RevisionRewardsCfg().to_dict() == before
    curriculum = configs('curricula', 'revision')
    c = curriculum.RevisionV1CurriculumCfg()
    assert [k for k, v in vars(c).items() if hasattr(v, 'params')] == [
        'obstacle_population', 'target_free_speed', 'target_goal_direction_deadzone']
    for name in ('target_free_speed', 'target_goal_direction_deadzone'):
        assert getattr(c, name).to_dict() == getattr(curriculum.RevisionCurriculumCfg(), name).to_dict()
    term = NS(params={})
    env = NS(cfg=NS(max_gray_obstacles=4, max_white_obstacles=5), event_manager=NS(
        get_term_cfg=lambda _: term, set_term_cfg=lambda *args: None))
    for iteration, expected in [(0,(3,3)), (9999,(3,3)), (10000,(3,4)), (11999,(3,4)),
                                (12000,(4,4)), (13999,(4,4)), (14000,(4,5)), (90000,(4,5)), (0,(3,3))]:
        env.learning_iteration = iteration
        c.obstacle_population.func(env, None, **c.obstacle_population.params)
        assert tuple(term.params.values()) == expected


def test_hold_distance_speed_boundaries_and_validity(functions, monkeypatch):
    d = torch.tensor([0, .015, .03-1e-9, .03, .03+1e-9, .08, float('nan')], dtype=torch.float64)
    v = torch.zeros_like(d)
    valid = torch.isfinite(d)
    monkeypatch.setattr(functions, 'target_measurements', lambda *args: (d, v, valid))
    env = NS(cfg=NS(goal_success_radius=.03))
    out = functions.goal_hold_v1(env) * 8
    assert out.tolist() == pytest.approx([16, 8, 1, 0, 0, 0, 0], abs=1e-6)
    d[:] = 0
    v[:] = torch.tensor([0, .05-1e-9, .05, .05+1e-9, .10, .15, 1], dtype=v.dtype)
    valid[:] = True
    assert (functions.goal_hold_v1(env)*8).tolist() == pytest.approx([16,16,16,16,8,1,0], abs=1e-6)
    valid[0] = False
    assert functions.goal_hold_v1(env)[0] == 0
    with pytest.raises(ValueError, match='synchronized'):
        functions.goal_hold_v1(NS(cfg=NS(goal_success_radius=.02)))


@pytest.mark.parametrize('dt', [.01, .02, .1])
def test_entry_latch_boundary_reentry_partial_reset_and_timeout(functions, dt):
    positions = torch.tensor([[.02,0,100], [.03,0,float('nan')], [.031,0,0], [float('nan'),0,0]], dtype=torch.float64)
    env = NS(num_envs=4, device='cpu', cfg=NS(goal_success_radius=.03), step_dt=dt,
             common_step_counter=0, _sim_step_counter=0, target_id=torch.zeros(4,1,dtype=torch.long),
             scene={'object_collection': NS(data=NS(object_pos_w=positions[:,None]))},
             command_manager=NS(get_command=lambda _: torch.zeros(4,3)))
    bonus = functions.FirstGoalEntryBonus(NS(params={}), env)
    def amount():
        return bonus(env)*dt*30
    assert amount().tolist() == pytest.approx([30,0,0,0])
    assert amount().tolist() == pytest.approx([30,0,0,0])
    for distance in (.04, .01):
        env.common_step_counter += 1
        positions[0,0] = distance
        assert amount()[0] == 0
    env.common_step_counter += 1
    positions[2,0] = .01
    assert amount().tolist() == pytest.approx([0,0,30,0])
    bonus.reset(torch.tensor([0]))
    assert amount().tolist() == pytest.approx([0,0,30,0])  # preserve other rows' same-step cache
    env.common_step_counter += 1
    paid = amount()
    assert paid.tolist() == pytest.approx([30,0,0,0])
    env.termination_manager = NS(get_term=lambda _: torch.tensor([True,True,True,False]))
    terminal = functions.successful_timeout_bonus(env)*dt*40
    assert (paid + terminal).tolist() == pytest.approx([70,40,40,0])


def test_wrench_raw_matches_parent_without_any_reward_manager(functions, load_mdp):
    old = importlib.import_module('_baseline_cpu_pkg.mdp.reward_functions.contact_sweep_clutter_fusion')
    env = NS(num_envs=3, device='cpu', common_step_counter=0, _sim_step_counter=0,
             sensor_data_fresh=torch.ones(3,dtype=torch.bool), wrench=torch.zeros(3,6))
    env.observation_manager = NS(cfg=NS(policy=NS(wrist_wrench=NS(func=lambda env: env.wrench, params={}))))
    child = functions.IndependentWrenchMagnitudeChangePenalty(NS(params={}), env)
    parent = old.WrenchMagnitudeChangePenalty(NS(params={}), env)
    # Compare the parent's raw path with an unlimited budget, including stale/NaN history.
    parent._reward_budget = lambda *args: torch.full((3,), float('inf'))
    samples = [torch.zeros(3,6), torch.tensor([[35.,0,0,0,0,0],[0,0,0,3.5,0,0],[100,0,0,10,0,0]]),
               torch.full((3,6),float('nan')), torch.zeros(3,6), torch.full((3,6),1000.)]
    for i, sample in enumerate(samples):
        env.common_step_counter += 1
        env.wrench[:] = sample
        env.sensor_data_fresh[1] = i != 2
        actual = child(env)
        torch.testing.assert_close(actual, parent(env, free_force_change=20, free_moment_change=2))
        assert torch.isfinite(actual).all() and (actual >= 0).all() and (actual <= 1).all()
        assert (actual*.1*.02 <= .002+1e-9).all()
        if i == 1:
            assert actual.tolist() == pytest.approx([math.expm1(1.5)/math.expm1(3)]*2+[1])
        if i >= 1:
            child.reset(torch.tensor([0])); parent.reset(torch.tensor([0]))
        torch.testing.assert_close(child(env), parent(env, free_force_change=20, free_moment_change=2))


def test_split_cost_equivalence_masks_caps_acceleration_history(load_mdp):
    old = importlib.import_module('_baseline_cpu_pkg.mdp.reward_functions.final')
    funcs = importlib.import_module('_baseline_cpu_pkg.mdp.reward_functions.contact_sweep_clutter_fusion')
    n, m = 3, 4
    position = torch.zeros(n,m,3)
    position[:,:,0] = torch.tensor([100.,.1,.2,100.])
    quat = torch.zeros(n,m,4); quat[...,0] = math.cos(.2); quat[...,1] = math.sin(.2)
    data = NS(object_pos_w=position, object_quat_w=quat, object_com_lin_acc_w=torch.full((n,m,3),1000.))
    env = NS(num_envs=n,device='cpu', common_step_counter=0,_sim_step_counter=0,
             scene={'object_collection':NS(data=data),
                    'shelf':NS(data=NS(root_quat_w=torch.tensor([[1.,0,0,0]]*n)))}, target_id=torch.zeros(n,1,dtype=torch.long),
             _initial_object_positions_w=torch.zeros_like(position),
             _active_object_mask=torch.tensor([[True,True,True,False]]*n))
    cfg = NS(params={})
    composite, accel = old.ObjectPenalty(cfg,env), funcs.ObjectAccelerationPenalty(cfg,env)
    for i in range(4):
        env.common_step_counter += 1
        if i == 2:
            composite.reset(torch.tensor([0])); accel.reset(torch.tensor([0]))
            env._active_object_mask[1,3] = True
        expected = -composite(env, free_acceleration=20, acceleration_weight=-.05)
        actual = (-30*funcs.active_object_tilt_cost(env)-150*funcs.active_obstacle_displacement_cost(env)
                  -.05*accel(env,free_acceleration=20,max_cost=100))
        torch.testing.assert_close(actual,expected)
        assert funcs.active_obstacle_displacement_cost(env).tolist() == pytest.approx([.04]*n)
        if i == 0:
            assert accel(env,free_acceleration=20).tolist() == [0]*n
        elif i == 1:
            assert accel(env,free_acceleration=20).tolist() == pytest.approx([300]*n)
        elif i == 2:
            assert accel(env,free_acceleration=20).tolist() == pytest.approx([0,300,300])
