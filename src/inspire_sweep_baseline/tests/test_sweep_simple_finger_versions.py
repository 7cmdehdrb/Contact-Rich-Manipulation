"""Finger-v1/v2 contracts, bounded PPO and latched retreat on CPU."""
import ast
import copy
import importlib
import math
from types import SimpleNamespace as NS

import pytest
import torch
from tensordict import TensorDict

from conftest import MODULE_ROOT, PROJECT_ROOT
from test_sweep_simple import family, configs, omni_config
from test_final_configuration import assert_ordered_equal
from test_omni_configuration import _execute_definitions


def module(name):
    return importlib.import_module('_baseline_cpu_pkg.' + name)


def test_parent_and_complete_config_diff(family):
    base = family.SweepSimpleFingerEnvCfg
    v1, v2 = family.SweepSimpleFingerV1EnvCfg, family.SweepSimpleFingerV2EnvCfg
    assert v1.__bases__ == (base,)
    assert v2.__bases__ == (v1,)
    snapshot = base().to_dict()
    a, b = v1(), v2()
    expected = copy.deepcopy(snapshot)
    expected['observations']['policy']['actions'] = a.to_dict()['observations']['policy']['actions']
    expected['rewards']['arm_joint_velocity']['weight'] = -.06
    expected['terminations']['arm_joint_velocity']['params']['threshold'] = 2.
    expected['curriculum']['arm_joint_velocity_threshold']['params'] = dict(
        initial=2.,minimum=1.,start_iteration=2000,interval=250,decrement=1/12)
    # This EnvCfg fixture replaces action construction with doubles. The action
    # child is checked separately, including pre-scale raw bounds.
    assert_ordered_equal(a.to_dict(),expected)
    expected = a.to_dict()
    for term in ('reaching','goal_distance','hand_contact'):
        expected['rewards'][term] = b.to_dict()['rewards'][term]
    expected['terminations']['success'] = b.to_dict()['terminations']['success']
    assert_ordered_equal(b.to_dict(),expected)
    assert b.rewards.target_inverse_distance.func is a.rewards.target_inverse_distance.func
    b.curriculum.arm_joint_velocity_threshold.params['initial'] = 99
    a.rewards.arm_joint_velocity.weight = -99
    assert_ordered_equal(base().to_dict(),snapshot)
    assert v2().curriculum.arm_joint_velocity_threshold.params['initial'] == 2.


@pytest.mark.parametrize('iteration,value',[(0,2),(1999,2),(2000,2),(2249,2),(2250,2-1/12),
    (4750,1+1/12),(4999,1+1/12),(5000,1),(99999,1)])
def test_velocity_schedule(configs,iteration,value):
    cfg = configs('curricula','sweep_simple').SweepSimpleFingerV1CurriculumCfg().arm_joint_velocity_threshold
    term = NS(params={'threshold':99.})
    env = NS(learning_iteration=iteration,termination_manager=NS(
        get_term_cfg=lambda _:term,set_term_cfg=lambda name,new:None))
    assert cfg.func(env,None,**cfg.params) == pytest.approx(value)
    assert term.params['threshold'] == pytest.approx(value)


def gate_env(load_mdp,n=4):
    pos = torch.zeros(n,1,3)
    goal = torch.zeros(n,3)
    initial = torch.zeros(n,1,3);initial[:,0,0]=-.09
    env=NS(num_envs=n,device='cpu',step_dt=.02,common_step_counter=0,
           scene={'object_collection':NS(data=NS(object_pos_w=pos))},
           command_manager=NS(get_command=lambda _:goal),
           _initial_object_positions_w=initial,_revision_initial_shelf_pos=torch.zeros(n,3))
    env._finger_retreat=module('mdp.common.sweep_simple').FingerRetreatState(env)
    env._finger_retreat.reset(env,torch.arange(n))
    return env,pos,goal


def test_continuous_timer_boundary_latch_partial_reset_and_target(load_mdp):
    env,pos,goal=gate_env(load_mdp)
    state=env._finger_retreat
    torch.testing.assert_close(state.target,torch.tensor([[-.05,0,1.13]]).expand(4,-1))
    pos[:,0,0]=torch.tensor([.019999,.02,.020001,float('nan')])
    state.update(env);assert not state.steps.any()  # reset same-step no time credit
    for step in range(1,150):
        env.common_step_counter=step
        state.update(env);state.update(env)
    assert state.steps.tolist()==[149,0,0,0]
    assert not state.active.any()
    env.common_step_counter=150
    assert state.update(env).tolist()==[True,False,False,False]
    pos[0,0,0]=.5
    pos[1,0,0]=.01
    for step in range(151,251):
        env.common_step_counter=step;state.update(env)
    assert state.active.tolist()==[True,False,False,False]
    assert state.steps[1]==100
    pos[1,0,0]=.02
    env.common_step_counter+=1;state.update(env);assert state.steps[1]==0
    pos[1,0,0]=.01
    for _ in range(149):
        env.common_step_counter+=1;state.update(env)
    assert not state.active[1]
    env.common_step_counter+=1;state.update(env);assert state.active[1]
    old_target=state.target.clone()
    goal[0]=torch.tensor([2.,3.,0.])
    env._initial_object_positions_w[0,0,:2]=torch.tensor([2.,4.])
    env._revision_initial_shelf_pos[0,2]=5
    state.reset(env,torch.tensor([0]))
    torch.testing.assert_close(state.target[0],torch.tensor([2.,3.05,6.13]))
    assert not state.active[0] and state.active[1]
    torch.testing.assert_close(state.target[1:],old_target[1:])


def test_success_only_at_timeout_and_failure_precedence(configs,load_mdp,monkeypatch):
    env,pos,goal=gate_env(load_mdp,3)
    fn=configs('terminations','sweep_simple').SweepSimpleFingerV2TerminationsCfg().success.func
    tree=ast.parse((PROJECT_ROOT/'IsaacLab/source/isaaclab/isaaclab/envs/mdp/terminations.py').read_text())
    tree.body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='time_out']
    namespace=dict(torch=torch,ManagerBasedRLEnv=object)
    exec(compile(tree,'official_time_out','exec'),namespace)
    official=namespace['time_out']
    monkeypatch.setattr(module('mdp.termination_functions.sweep_simple'), 'time_out', official)
    # Official time_out uses the environment episode length.
    env.episode_length_buf=torch.tensor([499,500,500]);env.max_episode_length=500
    env.termination_manager=NS(get_term=lambda name:torch.tensor([False,False,name=='object_fall_or_tilt']))
    env._finger_retreat.active[:]=True
    pos[:]=100.  # achievement stays latched even outside the goal
    assert fn(env).tolist()==[False,True,False]
    env._finger_retreat.active[1]=False
    assert not fn(env).any()


@pytest.mark.parametrize('initial',[.06,.09,.12])
def test_log_area_normalization_and_no_repay(configs,initial):
    from test_wrench_v0_1 import reward_env
    env,pos,_=reward_env(1)
    env.step_dt=.02
    objects=env.scene['object_collection'].data
    objects.object_link_state_w=objects.object_link_state_w.float()
    pos=objects.object_link_state_w[...,:3]
    goal=pos.new_zeros(1,3)
    env.command_manager=NS(get_command=lambda _:goal)
    pos[:]=0;pos[0,0,0]=initial
    cfg=configs('rewards','sweep_simple').SweepSimpleFingerV2RewardsCfg().goal_distance
    term=cfg.func(cfg,env)
    distances=torch.linspace(initial,0,1001)
    rewards=[]
    for distance in distances:
        pos[0,0,0]=distance
        rewards.append((term(env,**cfg.params)*env.step_dt*cfg.weight).item())
    cumulative=torch.tensor(rewards).cumsum(0)
    area=torch.trapezoid(cumulative.flip(0),distances.flip(0))
    assert area.item()==pytest.approx(100*initial**2/2,rel=2e-5)
    if initial==.09:assert cumulative[-1].item()==pytest.approx(13.92398374,rel=1e-5)
    assert rewards[-1]>rewards[1]>0
    pos[0,0,0]=.03
    assert term(env)==0
    pos[0,0,0]=0
    assert term(env)==0
    term.reset(torch.tensor([0]))
    assert term(env)==0  # zero initial distance is finite and earns zero


def test_retreat_reach_switch_and_two_cm_plateau(configs):
    from test_wrench_refactor_relative_ik_test import inverse_reach_fixture
    env,pos,eef,old_cfg,old=inverse_reach_fixture(configs,4)
    env._finger_retreat=NS(active=torch.tensor([False,True,True,True]),
        target=torch.tensor([[-.05,0,.08]]).expand(4,-1),update=lambda _:env._finger_retreat.active)
    cfg=configs('rewards','sweep_simple').SweepSimpleFingerV2RewardsCfg().reaching
    term=cfg.func(cfg,env)
    eef[:,0]=env._finger_retreat.target
    eef[2,0,0]+=.0199;eef[3,0,0]+=.0201
    value=term(env,**cfg.params)
    torch.testing.assert_close(value[0],old(env,**old_cfg.params)[0])
    assert value[1]==value[2] and value[3]<value[2]
    # Gate latch is independent of the live object's position after activation.
    before=value.clone();pos[:,0,0]=.5
    after=term(env,**cfg.params)
    torch.testing.assert_close(after[1:],before[1:])


def test_contact_bypass_only_after_gate(configs):
    from test_wrench_refactor_relative_ik_test import contact_count_fixture
    env,force,_=contact_count_fixture(configs,n=2)
    env._finger_retreat=NS(active=torch.tensor([False,True]),update=lambda _:env._finger_retreat.active)
    cfg=configs('rewards','sweep_simple').SweepSimpleFingerV2RewardsCfg().hand_contact
    term=cfg.func(cfg,env)
    force.zero_();env.sensor_data_fresh[:]=False
    env.scene['ee_frame'].data.target_pos_w[:,0,2]=2.
    assert term(env).tolist()==[0,20]
    assert term._count.tolist()==[0,20]
    assert not term._history.any()
    env.common_step_counter+=1
    assert term(env).tolist()==[0,20]
    term.reset(torch.tensor([1]));env._finger_retreat.active[1]=False
    env.common_step_counter+=1
    assert term(env).tolist()==[0,0]


def test_policy_density_bounds_kl_and_export(load_mdp):
    cls=module('agent.sweep_simple_distributions').FingerBoundedGaussian
    bounds=module('assets.hand_kinematics')
    lo,hi=bounds.finger_action_bounds('cpu')
    distribution=cls(12)
    mu=torch.randn(1024,12)*.4
    distribution.update(mu)
    actions=distribution.sample()
    assert ((actions>=lo)&(actions<=hi)).all()
    assert torch.isfinite(distribution.log_prob(actions)).all()
    # Verify the change of variables against independent torch transforms.
    from torch.distributions import TransformedDistribution,Normal,TanhTransform,AffineTransform
    reference=TransformedDistribution(Normal(mu,torch.ones_like(mu)),[
        TanhTransform(),AffineTransform((hi+lo)/2,(hi-lo)/2)])
    torch.testing.assert_close(distribution.log_prob(actions),reference.log_prob(actions).sum(-1),atol=2e-4,rtol=1e-4)
    old=tuple(x.clone() for x in distribution.params)
    torch.testing.assert_close(distribution.kl_divergence(old,old),torch.zeros(1024))
    distribution.update(mu+.2)
    torch.testing.assert_close(distribution.kl_divergence(old,distribution.params),torch.full((1024,),.24),atol=2e-6,rtol=1e-5)
    extreme=distribution.deterministic_output(torch.full((1,12),1000.))
    assert ((extreme>=lo)&(extreme<=hi)).all()
    assert torch.isfinite(distribution.log_prob(extreme)).all()
    export=torch.jit.script(distribution.as_deterministic_output_module())
    torch.testing.assert_close(export(mu),distribution.deterministic_output(mu))
    assert torch.isfinite(distribution.entropy).all()


def test_raw_last_action_and_input_clipping_before_scale(load_mdp):
    bounds=module('assets.hand_kinematics');lo,hi=bounds.finger_action_bounds('cpu')
    raw=torch.tensor([[100.,-100.,float('nan'),100.,-100.,float('inf'),-1.,100.,100.,100.,100.,100.]])
    env=NS(action_manager=NS(action=raw),_finger_action_lower=lo,_finger_action_upper=hi)
    observed=module('mdp.observation_functions.sweep_simple').bounded_previous_action(env)
    assert observed[0,0]==.5 and observed[0,1]==-.5 and observed[0,2]==0
    torch.testing.assert_close(observed[0,3:6],torch.tensor([math.pi/2,-math.pi/2,math.pi/2]))
    torch.testing.assert_close(observed[0,7:],hi[7:])
    assert observed[0,6]==0
    # Production OSC override must bound commands BEFORE parent processing/scales.
    class Parent:
        def process_actions(self,actions):
            self.raw=actions
            self.processed=actions*torch.tensor([.03]*3+[.15]*3)
    ns=_execute_definitions(MODULE_ROOT/'mdp/action_functions/sweep_simple_osc.py',dict(
        WrenchRefactorOSCAction=Parent,clip_finger_actions=bounds.clip_finger_actions))
    action=ns.BoundedFingerOSCAction.__new__(ns.BoundedFingerOSCAction)
    action._finger_lower,action._finger_upper=lo[:6],hi[:6]
    action.process_actions(raw[:,:6])
    torch.testing.assert_close(action.raw,observed[:,:6])
    assert action.processed[:,:3].abs().max()==pytest.approx(.015)
    assert action.processed[:,3:].abs().max()==pytest.approx(math.radians(13.5))
    # Runtime feeds the bounded action to ActionManager, so rate penalties and
    # next-step Last Action see the executed command rather than a rejected one.
    class EnvParent:
        def step(self,action):return action
    runtime=_execute_definitions(MODULE_ROOT/'runtime/sweep_simple.py',dict(
        SweepSimpleFingerEnv=EnvParent,__package__='_baseline_cpu_pkg.runtime'),{'SweepSimpleFingerV1Env'})
    live=runtime.SweepSimpleFingerV1Env()
    live._finger_action_lower,live._finger_action_upper=lo,hi
    torch.testing.assert_close(live.step(raw),observed)


def test_bounded_hand_raw_and_urdf_limits(load_mdp):
    import re
    hand=module('assets.hand_kinematics')
    asset=(MODULE_ROOT/'assets/data/wrench_right_hand/ur5e_inspire_right.usda').read_text()
    for name,upper in zip(hand.HAND_DRIVE_JOINT_NAMES,hand.HAND_DRIVE_UPPER_LIMITS):
        # USD angular limit attributes are degrees, original URDF values radians.
        start=asset.index('"'+name+'"')
        region=asset[start:start+2200]
        lower_usd=float(re.search(r'physics:lowerLimit = ([-\d.]+)',region)[1])
        upper_usd=float(re.search(r'physics:upperLimit = ([-\d.]+)',region)[1])
        assert lower_usd==0
        assert math.radians(upper_usd)==pytest.approx(upper,abs=2e-7)
    cls=module('mdp.action_functions.sweep_simple').BoundedFingerJointPositionAction
    term=cls.__new__(cls)
    term._raw_actions=torch.zeros(1,6);term._processed_actions=torch.zeros(1,6)
    term._joint_targets=torch.zeros(1,12)
    term._upper_limits=torch.tensor(hand.HAND_DRIVE_UPPER_LIMITS)
    term._drive_indices=torch.tensor(hand.HAND_MIMIC_DRIVE_INDICES)
    term._multipliers=torch.tensor(hand.HAND_MIMIC_MULTIPLIERS)
    term.process_actions(torch.tensor([[-100,100,100,100,100,100.]]))
    torch.testing.assert_close(term.raw_actions,term.processed_actions)
    assert term.raw_actions[0,0]==0
    torch.testing.assert_close(term.raw_actions[0,1:],term._upper_limits[1:])


def test_real_bounded_ppo_update_checkpoint_and_full_export(load_mdp,tmp_path):
    from rsl_rl.algorithms import PPO
    model=module('agent.sweep_simple_models').SweepSimpleFingerModel
    dist=module('agent.sweep_simple_distributions').FingerBoundedGaussian
    obs=TensorDict({'policy':torch.randn(8,66),'obstacles':torch.randn(8,64)},batch_size=[8])
    obs['obstacles'][:,::4]=1
    cfg=dict(actor=dict(class_name=model,distribution_cfg=dict(class_name=dist,init_std=1.)),
             critic=dict(class_name=model),algorithm=dict(class_name='PPO',num_learning_epochs=2,num_mini_batches=2,
             schedule='adaptive',desired_kl=.02,entropy_coef=.005),
             obs_groups={'actor':['policy','obstacles'],'critic':['policy','obstacles']},
             num_steps_per_env=4,multi_gpu=None)
    def create():return PPO.construct_algorithm(obs,NS(num_envs=8,num_actions=12),copy.deepcopy(cfg),'cpu')
    alg=create();lo,hi=module('assets.hand_kinematics').finger_action_bounds('cpu')
    before=copy.deepcopy(alg.actor.state_dict())
    with torch.inference_mode():
        for _ in range(4):
            actions=alg.act(obs)
            assert ((actions>=lo)&(actions<=hi)).all()
            alg.process_env_step(obs,-(actions-.2).square().sum(-1),torch.zeros(8,dtype=torch.long),{})
        alg.compute_returns(obs)
    losses=alg.update()
    assert all(math.isfinite(v) for v in losses.values())
    assert any(not torch.equal(v,alg.actor.state_dict()[k]) for k,v in before.items())
    assert all(torch.isfinite(p).all() for p in alg.actor.parameters())
    path=tmp_path/'bounded.pt';torch.save(alg.save(),path)
    other=create();other.load(torch.load(path,weights_only=True),None,strict=True)
    torch.testing.assert_close(alg.actor(obs),other.actor(obs))
    exporter=torch.jit.script(alg.actor.as_jit())
    torch.testing.assert_close(exporter(torch.cat([obs['policy'],obs['obstacles']],-1)),alg.actor(obs))
    assert alg.actor.as_onnx().get_dummy_inputs()[0].shape==(1,130)


def test_new_registry_and_ppo_inheritance(omni_config):
    from dataclasses import MISSING
    from test_omni_configuration import _AssetNode
    cc=omni_config.baseline.configclass
    @cc
    class Base:
        distribution_cfg:object=None
        policy:object=None
    class Distribution(_AssetNode):
        def __init__(self,**kwargs):super().__init__(**kwargs)
    Base.GaussianDistributionCfg=Distribution
    runner=_execute_definitions(MODULE_ROOT/'agent/sweep_simple_ppo_cfg.py',dict(
        configclass=cc,MISSING=MISSING,RslRlMLPModelCfg=Base,RslRlOnPolicyRunnerCfg=Base,RslRlPpoAlgorithmCfg=_AssetNode))
    parent=runner.SweepSimpleFingerPPORunnerCfg().to_dict()
    actual=runner.SweepSimpleFingerV1PPORunnerCfg().to_dict()
    expected=copy.deepcopy(parent);expected['experiment_name']='inspire_sweep_simple_finger_v1'
    expected['actor']['distribution_cfg']=dict(class_name='inspire_sweep_baseline.agent.sweep_simple_distributions:FingerBoundedGaussian',init_std=1.)
    assert_ordered_equal(actual,expected)
    expected['experiment_name']='inspire_sweep_simple_finger_v2'
    assert_ordered_equal(runner.SweepSimpleFingerV2PPORunnerCfg().to_dict(),expected)
    entries={}
    _execute_definitions(MODULE_ROOT/'__init__.py',{'gym':NS(registry=entries,register=lambda **kw:entries.update({kw['id']:kw}))})
    for v in (1,2):
        entry=entries[f'Inspire-Sweep-Simple-Finger-v{v}']
        assert entry['entry_point'].endswith(f':SweepSimpleFingerV{v}Env')
        assert entry['kwargs']['env_cfg_entry_point'].endswith(f':SweepSimpleFingerV{v}EnvCfg')
        assert entry['kwargs']['rsl_rl_cfg_entry_point'].endswith(f':SweepSimpleFingerV{v}PPORunnerCfg')


def test_runtime_retreat_reset_records_old_outcome_first(load_mdp):
    env,pos,goal=gate_env(load_mdp,2)
    env._finger_retreat.active[:]=True
    class Parent:
        def _reset_idx(self,ids):
            self.recorded=self._finger_retreat.active[ids].clone()
            self._initial_object_positions_w[ids,0,0]=-.1
    ns=_execute_definitions(MODULE_ROOT/'runtime/sweep_simple.py',dict(
        SweepSimpleFingerV1Env=Parent),{'SweepSimpleFingerV2Env'})
    live=ns.SweepSimpleFingerV2Env()
    live.__dict__.update(vars(env))
    live._reset_idx(torch.tensor([0]))
    assert live.recorded.tolist()==[True]
    assert live._finger_retreat.active.tolist()==[False,True]
    assert live._finger_retreat.steps[0]==0


def test_log_distance_gt_xy_and_first_transition_after_partial_reset(configs):
    from test_wrench_refactor_relative_ik_test import inverse_reach_fixture
    env,pos,eef,_,_=inverse_reach_fixture(configs,2)
    goal=pos.new_tensor([[0.,0.,0.],[0.,0.,0.]])
    env.command_manager=NS(get_command=lambda _:goal)
    pos[:,0,0]=.09
    cfg=configs('rewards','sweep_simple').SweepSimpleFingerV2RewardsCfg().goal_distance
    term=cfg.func(cfg,env)
    pos[:,0,0]=.03
    pos[0,0,2]=1000  # Z cannot alter planar progress.
    first=term(env)*env.step_dt
    torch.testing.assert_close(first[0],first[1])
    saved=term._best_progress[1].clone()
    pos[0,0,0]=.12
    term.reset(torch.tensor([0]))
    goal[0,0]=.02  # Command reset follows reward reset in the manager lifecycle.
    pos[0,0,0]=.11  # First action's 1cm improvement must be included.
    value=term(env)
    assert value[0]>0 and value[1]==0
    assert term._best_progress[1]==saved
