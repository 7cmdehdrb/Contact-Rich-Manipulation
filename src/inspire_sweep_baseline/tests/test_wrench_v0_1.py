"""Consecutive target contact, signed goal cosine, and fixed easy layout banks."""

import importlib
import math
from types import SimpleNamespace as NS

import pytest
import torch

from conftest import MODULE_ROOT
from test_final_configuration import configs
from test_omni_configuration import omni_config, _execute_definitions
from test_fusion_holding_osc import osc
from test_revision_layout import bounds


def reward_env(count=1):
    state = torch.zeros(count, 1, 13, dtype=torch.float64)
    position = state[..., :3]
    position[:, 0, 0] = .05
    initial = position.clone()
    initial[:, 0, 0] = .1
    # Two opposed contact contributions must be summed as vectors first.
    force = torch.zeros(count, 1, 2, 3, dtype=torch.float64)
    force[:, 0, 0, 0] = .02
    force[:, 0, 1, 0] = -.01
    env = NS(num_envs=count, device='cpu', common_step_counter=1,
             target_id=torch.zeros(count, 1, dtype=torch.long),
             sensor_data_fresh=torch.ones(count, dtype=torch.bool),
             _initial_object_positions_w=initial,
             _revision_initial_shelf_pos=torch.zeros(count, 3, dtype=torch.float64),
             cfg=NS(goal_reward_height_limit=.12),
             command_manager=NS(get_command=lambda _: torch.zeros(count, 3, dtype=torch.float64)),
             scene={'object_collection': NS(object_names=['target'], data=NS(object_pos_w=position, object_link_state_w=state)),
                    'hand_contact_target': NS(data=NS(force_matrix_w=force)),
                    'ee_frame': NS(data=NS(target_pos_w=torch.tensor([[[0.,0.,1.16]]], dtype=torch.float64).repeat(count,1,1)))})
    return env, position, force


def test_contact_sequence_duplicate_calls_partial_reset_and_skipped_step(configs):
    cls = configs('reward_functions', 'wrench').WrenchV0_1GoalDistance
    env, _, force = reward_env(2)
    term = cls(NS(params={}), env)
    assert term(env).tolist() == [0, 0]
    assert term(env).tolist() == [0, 0]
    env.common_step_counter += 1
    torch.testing.assert_close(term(env), torch.full((2,), math.exp(-1)))
    term.reset(torch.tensor([0]))
    assert term(env).tolist() == pytest.approx([0, math.exp(-1)])
    env.common_step_counter += 1
    assert term(env).tolist() == pytest.approx([0, math.exp(-1)])
    env.common_step_counter += 1
    assert (term(env) > 0).all()
    force[0, 0, 1, 0] = -.02  # zero resultant despite two nonzero contacts
    env.common_step_counter += 1
    assert term(env).tolist() == pytest.approx([0, math.exp(-1)])
    force[0, 0, 1, 0] = -.01
    env.common_step_counter += 1
    assert term(env).tolist() == pytest.approx([0, math.exp(-1)])
    env.common_step_counter += 1
    assert (term(env) > 0).all()
    env.common_step_counter += 2
    assert not term(env).any()
    env.common_step_counter += 1
    assert (term(env) > 0).all()
    term.reset()
    assert not term(env).any()


@pytest.mark.parametrize('net,expected',[(.009999,False),(.01,True),(.010001,True),(float('nan'),False),(float('inf'),False)])
def test_force_threshold_and_finite_readings(configs, net, expected):
    cls = configs('reward_functions', 'wrench').WrenchV0_1GoalDistance
    env, _, force = reward_env()
    force.zero_(); force[0,0,0,0] = net
    term = cls(NS(params={}), env)
    term(env); env.common_step_counter += 1
    assert bool(term(env).item() > 0) == expected


@pytest.mark.parametrize('xy,alpha',[( (.05,0),1), ((.03,.04),.6), ((0,.05),0), ((-.05,0),-1), ((0,0),1)])
def test_angle_distance_and_zero_vector(configs, xy, alpha):
    cls = configs('reward_functions', 'wrench').WrenchV0_1GoalDistance
    env, pos, _ = reward_env()
    pos[0,0,:2] = torch.tensor(xy)
    term = cls(NS(params={}), env)
    term(env); env.common_step_counter += 1
    expected = math.exp(-math.hypot(*xy)/.05) * alpha
    assert term(env).item() == pytest.approx(expected, abs=1e-7)
    # Cosine and distance remain planar even if the object root Z changes.
    pos[0,0,2] = 100
    env.common_step_counter += 1
    assert term(env).item() == pytest.approx(expected, abs=1e-7)


def test_height_freshness_nonfinite_geometry_and_initial_position(configs):
    cls = configs('reward_functions', 'wrench').WrenchV0_1GoalDistance
    env, pos, _ = reward_env(7)
    env.scene['ee_frame'].data.target_pos_w[:,0,2] = torch.tensor([1.169999,1.17,1.170001,1.16,1.16,1.16,1.16], dtype=torch.float64)
    env.sensor_data_fresh[3] = False
    pos[4,0,0] = float('nan')
    env._initial_object_positions_w[5] = 0  # zero initial direction
    env._initial_object_positions_w[6,0,0] = float('inf')
    term = cls(NS(params={}), env)
    term(env); env.common_step_counter += 1
    assert term(env).tolist() == pytest.approx([math.exp(-1),0,0,0,0,0,0])
    env.sensor_data_fresh[3] = True
    env.common_step_counter += 1
    assert term(env)[3] == 0
    env.common_step_counter += 1
    assert term(env)[3] > 0


def test_shorter_distance_increases_reward_with_same_angle_and_contact(configs):
    cls = configs('reward_functions', 'wrench').WrenchV0_1GoalDistance
    env, pos, _ = reward_env()
    term = cls(NS(params={}), env)
    term(env); env.common_step_counter += 1
    distant = term(env).item()
    pos[0,0,0] = .025; env.common_step_counter += 1
    assert term(env).item() > distant


@pytest.mark.parametrize('distance,alpha',[(.03-1e-9,1),(.03,-1),(.03+1e-9,-1)])
def test_near_goal_angle_override_strict_boundary_keeps_contact_and_height(configs, distance, alpha):
    cls = configs('reward_functions', 'wrench').WrenchV0_1GoalDistance
    env, pos, force = reward_env()
    pos[0,0,0] = -distance
    term = cls(NS(params={}), env)
    assert term(env).item() == 0  # proximity does not bypass contact history
    env.common_step_counter += 1
    assert term(env).item() == pytest.approx(alpha * math.exp(-distance/.05))
    env.scene['ee_frame'].data.target_pos_w[0,0,2] = 1.18
    env.common_step_counter += 1
    assert term(env).item() == 0
    env.scene['ee_frame'].data.target_pos_w[0,0,2] = 1.16
    force.zero_(); env.common_step_counter += 1
    assert term(env).item() == 0


def test_fixed_angle_bank_startup_and_all_population_transitions(configs, load_mdp):
    events = configs('event_functions', 'wrench')
    curriculum = configs('curricula', 'wrench').WrenchV0_1CurriculumCfg()
    spawn = configs('events', 'wrench').WrenchEventsCfg().object_spawn
    spawn.params.update(excluded_x_half_angle=math.radians(40), goal_distance=.075)
    stages = ((40,3,3), (40,3,4), (40,4,4), (40,4,5))
    env = NS(num_envs=4, _revision_bounds_t=bounds(4),
             _revision_layout_rng=torch.Generator().manual_seed(17),
             cfg=NS(events=NS(object_spawn=spawn),layout_bank_size=2,
                    layout_build_batch_size=8,red_margin=.023,rear_offset=.02,
                    max_gray_obstacles=4,max_white_obstacles=5),
             event_manager=NS(get_term_cfg=lambda _:spawn,set_term_cfg=lambda *a:None))
    events.initialize_wrench_layouts(env, None, stages=stages, min_goal_distance=.04)
    bank = env._wrench_layout_bank
    assert bank.stages == stages
    assert curriculum.target_goal_direction_deadzone is None
    for iteration in (0,9999,10000,11999,12000,13999,14000,25000,90000):
        env.learning_iteration = iteration
        pop = curriculum.obstacle_population
        pop.func(env, None, **pop.params)
        assert spawn.params['excluded_x_half_angle'] == math.radians(40)
        sampled = bank.sample(torch.arange(4), env._revision_layout_rng,
                              spawn.params['gray_obstacles'], spawn.params['white_obstacles'],
                              spawn.params['excluded_x_half_angle'])
        delta = sampled[3]; distance = delta.norm(dim=-1)
        assert ((distance >= .04-1e-6) & (distance <= .075+1e-6)).all()
        assert (distance > .03).all()  # no initial sample within the success radius
        assert (delta[:,1].abs()/distance >= math.sin(math.radians(40))-1e-6).all()


def test_sampler_minimum_outside_success_and_parent_default_unchanged(load_mdp):
    m = importlib.import_module('_baseline_cpu_pkg.mdp.common.revision_layout')
    b = bounds(256)[:,0]
    for limit in (.04, .075):
        start, goal = m.sample_revision_commands(b, torch.Generator().manual_seed(9),
            goal_distance=limit, min_goal_distance=.04, excluded_x_half_angle=math.radians(40))
        torch.testing.assert_close((goal-start).norm(dim=-1), torch.full((256,),limit))
    with pytest.raises(ValueError):
        m.sample_revision_commands(b, torch.Generator().manual_seed(9),
                                   goal_distance=.04-1e-9, min_goal_distance=.04)
    with pytest.raises(ValueError):
        m.sample_revision_commands(b, torch.Generator().manual_seed(9), goal_distance=.025)
    start, goal = m.sample_revision_commands(b, torch.Generator().manual_seed(9))
    assert ((goal-start).norm(dim=-1) >= .05-1e-6).all()
    assert ((goal-start).norm(dim=-1) <= .15+1e-6).all()


def test_runner_only_changes_experiment_name(omni_config):
    cc = omni_config.baseline.configclass
    @cc
    class Parent:
        experiment_name = 'inspire_wrench_v0'
        settings = {'gamma': .98, 'steps':36}
    m = _execute_definitions(MODULE_ROOT/'agent/wrench_ppo_cfg.py',
        dict(configclass=cc,WrenchPPORunnerCfg=Parent), {'WrenchV0_1PPORunnerCfg'})
    parent = Parent(); child = m.WrenchV0_1PPORunnerCfg()
    assert type(child).__bases__ == (Parent,)
    assert child.experiment_name == 'inspire_wrench_v0_1'
    child.experiment_name = parent.experiment_name
    assert child.to_dict() == parent.to_dict()
