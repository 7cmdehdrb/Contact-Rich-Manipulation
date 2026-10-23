"""Holding reward geometry, strict success, warmup and rewind contracts."""
import importlib
import runpy
from types import SimpleNamespace as NS
import pytest
import torch
from conftest import MODULE_ROOT
from test_omni_configuration import _execute_definitions, omni_config

ROOT=MODULE_ROOT/'contact_sweep_clutter_fusion_holding_v0'
schedule=runpy.run_path(str(ROOT/'curriculum.py'))


@pytest.fixture
def holding(load_mdp):
    load_mdp('goal_rewards')
    return importlib.import_module('_baseline_cpu_pkg.contact_sweep_clutter_fusion_holding_v0.mdp')


def environment():
    pos=torch.tensor([[[0.,0.,1.05]]]*4)
    quat=torch.tensor([[[1.,0.,0.,0.]]]*4)
    vel=torch.zeros(4,1,3)
    fail=torch.zeros(4,dtype=torch.bool)
    return NS(num_envs=4,device='cpu',step_dt=.02,target_id=torch.zeros(4,1,dtype=torch.long),
        scene={'object_collection':NS(data=NS(object_pos_w=pos,object_quat_w=quat,object_lin_vel_w=vel)),
               'shelf':NS(data=NS(root_pos_w=torch.zeros(4,3),root_quat_w=quat[:,0]))},
        command_manager=NS(get_command=lambda _:torch.tensor([[0.,0.,1.05]]*4)),
        termination_manager=NS(get_term=lambda _:fail),fail=fail)


def test_settling_far_still_fast_and_invalid(holding):
    e=environment();p=e.scene['object_collection'].data.object_pos_w
    p[1,0,0]=.02;p[2,0,0]=.12
    e.scene['object_collection'].data.object_lin_vel_w[3,0,0]=.05
    r=holding.goal_settling(e,position_scale=.02,speed_scale=.05)
    assert r.tolist()==pytest.approx([1.,.5,0.,.5])
    p[0,0,2]=1.10;p[1,0,1]=.47;p[3,0,0]=float('nan')
    assert holding.goal_settling(e).tolist()==[0.,0.,0.,0.]


def test_success_not_dense_reward_exit_failure_and_partial_reset(holding):
    e=environment();p=e.scene['object_collection'].data.object_pos_w
    p[1,0,0]=.008 # nonzero dense reward, never strict success
    e.scene['object_collection'].data.object_lin_vel_w[2,0,0]=.02
    assert (holding.goal_settling(e)>0).all()
    term=holding.HoldingSuccess(NS(),e)
    for _ in range(149):assert not term(e).any()
    assert term(e).tolist()==[True,False,False,True]
    term.reset([0]);assert term(e).tolist()==[False,False,False,True]
    e.fail[3]=True;assert not term(e,failure_terms=('push_fast',)).any()
    e.fail[3]=False;assert not term(e,failure_terms=('push_fast',))[3]
    term.reset();assert not term._steps.any()


@pytest.fixture
def curricula(omni_config):
    ns=dict(configclass=omni_config.baseline.configclass,CurriculumTermCfg=omni_config.terms.RewardTermCfg,
            delayed_reward_schedule=lambda:None,goal_deadzone_schedule=lambda:None,
            obstacle_population_schedule=lambda:None,reward_schedule=schedule['reward_schedule'])
    ns.update(vars(_execute_definitions(MODULE_ROOT/'contact_sweep_clutter_fusion_v1_0/curriculum_cfg.py',ns)))
    return _execute_definitions(ROOT/'curriculum_cfg.py',ns).FusionHoldingV0CurriculumCfg()


@pytest.mark.parametrize('name,end', [('acceleration_weight',-.05),('wrench_weight',-.1),('obstacle_weight',-.1)])
def test_zero_warmup_and_rewind(curricula,name,end):
    term=getattr(curricula,name);resets=[];cfg=NS(params={},weight=0.,func=NS(reset=lambda:resets.append(1)))
    env=NS(reward_manager=NS(get_term_cfg=lambda _:cfg,set_term_cfg=lambda *_:None))
    for it in [0,999,1000,1249]:
        env.learning_iteration=it;assert term.func(env,None,**term.params)==0.
    env.learning_iteration=1250;assert term.func(env,None,**term.params)<0;assert len(resets)==1
    env.learning_iteration=16000;assert term.func(env,None,**term.params)==pytest.approx(end)
    env.learning_iteration=0;assert term.func(env,None,**term.params)==0.;assert len(resets)==2


def test_speed_schedule_matches_unchanged_v14_and_all_points(curricula):
    old=runpy.run_path(str(MODULE_ROOT/'contact_sweep_clutter_fusion_v1_4/curriculum.py'))['staged_reward_threshold']
    for name,initial,stage in [('arm_free_speed',1.,(500,3300,.98,.7)),('eef_free_speed',1.5,(500,13300,1.48,.2)),('pushing_free_speed',.1,(500,1900,.09,.02))]:
        term=getattr(curricula,name);cfg=NS(params={},weight=-1.)
        env=NS(reward_manager=NS(get_term_cfg=lambda _:cfg,set_term_cfg=lambda *_:None))
        for it in range(17001):
            env.learning_iteration=it
            expected=old(env,None,term.params['term_name'],term.params['parameter'],initial,(stage,))
            assert term.func(env,None,**term.params)==pytest.approx(expected)
    for term in vars(curricula).values():
        if hasattr(term,'params') and 'points' in term.params:
            for it,value in term.params['points']:
                assert schedule['scheduled_value'](it,term.params['points'])==pytest.approx(value)
