"""Feasibility reward state, curriculum boundaries and randomized bank coverage."""
import importlib
import math
from types import SimpleNamespace as NS

import pytest
import torch

from conftest import MODULE_ROOT
from test_final_configuration import configs
from test_omni_configuration import omni_config, _execute_definitions
from test_fusion_holding_osc import osc
from test_wrench_v0_1 import reward_env
from test_revision_layout import bounds


def contact_env(n=1):
    env, pos, force = reward_env(n)
    env.cfg.contact_height_limit = .12
    return env, pos, force


@pytest.mark.parametrize('xy,alpha', [((.05,0),1), ((.03,.04),.6), ((0,.05),0),
    ((-.05,0),-1), ((-.03,0),-1), ((-.03+1e-9,0),1), ((-.03-1e-9,0),-1), ((0,0),1)])
def test_distance_no_contact_and_strict_near_goal_override(configs, xy, alpha):
    m = configs('reward_functions', 'wrench')
    env, pos, force = contact_env()
    pos[0,0,:2] = torch.tensor(xy, dtype=torch.float64)
    force.zero_(); env.sensor_data_fresh[:] = False
    assert m.goal_distance_cosine(env).item() == pytest.approx(math.exp(-math.hypot(*xy)/.05)*alpha)
    env.scene['ee_frame'].data.target_pos_w[...,2] = 1.17
    assert m.goal_distance_cosine(env).item() == 0


def test_window_accumulation_five_rejections_recovery_cache_and_partial_reset(configs):
    cls = configs('reward_functions', 'wrench').WrenchV0_2ContactMaintenance
    env, _, force = contact_env(2)
    term = cls(NS(params={}), env)
    # 1,0,1 qualifies; a rejected window retains the count, but pays zero.
    signals = [1,0,1,0,1,1,0,0,0,0,0,0,1,1]
    expected = [0,0,1,0,2,3,4,0,0,0,0,0,0,1]
    for step, (signal, payout) in enumerate(zip(signals,expected), 1):
        env.common_step_counter = step
        force.zero_(); force[:,:,0,0] = .01 * signal
        assert term(env).tolist() == [payout,payout]
        assert term(env).tolist() == [payout,payout]
        if step == 11:
            assert term._count.tolist() == [4,4]  # fourth rejected window
        if step == 12:
            assert term._count.tolist() == [0,0]  # fifth rejected window
    term.reset(torch.tensor([0]))
    assert term(env).tolist() == [0,1]
    env.common_step_counter += 1
    assert term(env).tolist() == [0,2]
    env.common_step_counter += 1
    assert term(env).tolist() == [1,3]
    env.common_step_counter += 2
    assert term(env).tolist() == [0,0]
    term.reset()
    assert not term._history.any() and not term._count.any() and not term._misses.any()


@pytest.mark.parametrize('net,expected', [(.009999,0),(.01,1),(.010001,1),(float('nan'),0),(float('inf'),0)])
def test_filtered_force_threshold(configs, net, expected):
    env, _, force = contact_env()
    force.zero_(); force[0,0,0,0] = net
    term = configs('reward_functions','wrench').WrenchV0_2ContactMaintenance(NS(params={}),env)
    assert term(env).item() == 0
    env.common_step_counter += 1
    assert term(env).item() == expected


def test_goal_immediately_qualifies_but_exact_boundary_does_not(configs):
    env, pos, force = contact_env(5)
    pos[:,0,0] = torch.tensor([.03-1e-9,.03,.03+1e-9,0,float('nan')],dtype=torch.float64)
    force.zero_(); env.sensor_data_fresh[:] = False
    env.scene['ee_frame'].data.target_pos_w[...,2] = 2  # goal override also bypasses height
    term = configs('reward_functions','wrench').WrenchV0_2ContactMaintenance(NS(params={}),env)
    assert term(env).tolist() == [1,0,0,1,0]
    env.common_step_counter += 1
    assert term(env).tolist() == [2,0,0,2,0]
    assert (term(env)*1.45*.02).tolist() == pytest.approx([.058,0,0,.058,0])


def test_stale_high_and_cancelled_force_do_not_count(configs):
    env, _, force = contact_env(4)
    env.sensor_data_fresh[0] = False
    env.scene['ee_frame'].data.target_pos_w[1,0,2] = 1.17
    force[2,0,1,0] = -.02
    term = configs('reward_functions','wrench').WrenchV0_2ContactMaintenance(NS(params={}),env)
    term(env); env.common_step_counter += 1
    assert term(env).tolist() == [0,0,0,1]


def test_all_schedule_boundaries_randomization_bank_coverage_and_rewind(configs):
    m = configs('curriculum_functions','wrench')
    stages = set(m.feasibility_layout_stages())
    term = NS(params={})
    env = NS(event_manager=NS(get_term_cfg=lambda _:term,set_term_cfg=lambda *a:None))
    expected = [(0,(-1,-1)),(1999,(-1,-1)),(2000,(-1,0)),(4999,(-1,0)),
                (5000,(0,0)),(8000,(0,1)),(11000,(1,1)),(14000,(1,2)),
                (17000,(2,2)),(20000,(2,3)),(23000,(3,3)),(26000,(3,4)),
                (29000,(4,4)),(31999,(4,4)),(32000,(4,5)),(90000,(4,5)),(0,(-1,-1))]
    for iteration, counts in expected:
        env.learning_iteration = iteration
        m.feasibility_obstacle_population_schedule(env,None)
        assert (term.params['gray_obstacles'],term.params['white_obstacles']) == counts
    for iteration in range(90001):
        g,w = m.feasibility_population(iteration)
        angle = max(0,30-5*(0 if iteration<10000 else 1+(iteration-10000)//3000))
        for dg in (-1,0,1):
            for dw in (-1,0,1):
                assert (angle,max(0,min(4,g+dg)),max(0,min(5,w+dw))) in stages


def test_reset_randomization_before_clamp_independent_seeded_and_selected_rows(configs, monkeypatch):
    m = configs('event_functions','wrench')
    captured = torch.full((2000,2),-99)
    def capture(env, ids, gray_obstacles, white_obstacles, **kw):
        captured[ids] = torch.tensor([gray_obstacles,white_obstacles])
    monkeypatch.setattr(m,'reset_wrench_scene',capture)
    env = NS(num_envs=2000,device='cpu',_revision_layout_rng=torch.Generator().manual_seed(42))
    ids = torch.arange(1,2000,2)
    m.reset_wrench_feasibility_scene(env,ids)
    assert (captured[ids]==0).all() and (captured[::2]==-99).all()
    m.reset_wrench_feasibility_scene(env,ids,gray_obstacles=2,white_obstacles=3)
    pairs = set(map(tuple,captured[ids].tolist()))
    assert pairs == {(g,w) for g in (1,2,3) for w in (2,3,4)}
    env._revision_layout_rng.manual_seed(9)
    m.reset_wrench_feasibility_scene(env,ids,gray_obstacles=4,white_obstacles=5)
    snapshot = captured.clone()
    assert set(captured[ids,0].tolist())=={3,4} and set(captured[ids,1].tolist())=={4,5}
    env._revision_layout_rng.manual_seed(9)
    m.reset_wrench_feasibility_scene(env,ids,gray_obstacles=4,white_obstacles=5)
    assert torch.equal(snapshot,captured)
    m.reset_wrench_feasibility_scene(env,torch.tensor([],dtype=torch.long))
    assert torch.equal(snapshot,captured)


def test_actual_layout_bank_builds_and_samples_all_reachable_stages(configs):
    m = configs('event_functions','wrench')
    stages = configs('curriculum_functions','wrench').feasibility_layout_stages()
    spawn = configs('events','wrench').WrenchEventsCfg().object_spawn
    env = NS(num_envs=2,_revision_bounds_t=bounds(2),_revision_layout_rng=torch.Generator().manual_seed(9),
             cfg=NS(events=NS(object_spawn=spawn),layout_bank_size=1,layout_build_batch_size=16,
                    red_margin=.023,rear_offset=.02))
    m.initialize_wrench_layouts(env,None,stages=stages)
    for angle,g,w in stages:
        values = env._wrench_layout_bank.sample(torch.arange(2),env._revision_layout_rng,g,w,math.radians(angle))
        active,actual = values[2],values[4]
        assert (actual <= torch.tensor([g,w])).all()
        assert torch.equal(active.sum(-1),actual.sum(-1)+1)
        if (g,w)==(0,0):
            assert (active.sum(-1)==1).all()


def test_runner_preserves_all_parent_settings(omni_config):
    cc = omni_config.baseline.configclass
    @cc
    class Parent:
        experiment_name = 'inspire_wrench_v0'
        settings = {'gamma':.98,'steps':36}
    m = _execute_definitions(MODULE_ROOT/'agent/wrench_ppo_cfg.py',
        dict(configclass=cc,WrenchPPORunnerCfg=Parent), {'WrenchV0_2PPORunnerCfg'})
    child = m.WrenchV0_2PPORunnerCfg()
    assert type(child).__bases__ == (Parent,)
    assert child.experiment_name == 'inspire_wrench_v0_2'
    child.experiment_name = Parent().experiment_name
    assert child.to_dict() == Parent().to_dict()
