"""Sensor-frame six-axis feedback, real action lifecycle and v0 isolation on CPU."""
import ast
from copy import deepcopy
import importlib
import math
from types import SimpleNamespace as NS

import pytest
import torch

from conftest import MODULE_ROOT
from test_final_configuration import configs, assert_ordered_equal
from test_omni_configuration import omni_config, _execute_definitions, _AssetNode
from test_fusion_holding_osc import osc


@pytest.fixture
def control(load_mdp):
    return importlib.import_module('_baseline_cpu_pkg.mdp.action_functions.wrench')


@pytest.fixture
def action_configs(osc,omni_config,control):
    parent=osc.v0().actions.arm_action
    ns=dict(configclass=omni_config.baseline.configclass,
            OperationalSpaceControllerCfg=type(parent.controller_cfg),
            OperationalSpaceControllerActionCfg=type(parent),ActionTermCfg=type(parent).__bases__[0],
            ARM_JOINT_NAMES=omni_config.baseline.ARM_JOINT_NAMES,
            UR10OperationalSpaceControllerAction=osc.action,HandOpennessAction=object,
            SensorFrameWrenchAction=control.SensorFrameWrenchAction)
    return _execute_definitions(MODULE_ROOT/'mdp/actions/wrench.py',ns)


def test_cfg_parameters_and_only_arm_changes(action_configs):
    old=action_configs.WrenchActionsCfg()
    before=old.to_dict()
    new=action_configs.WrenchV1ActionsCfg()
    assert type(new).__bases__==(type(old),)
    a=new.arm_action;c=a.controller_cfg
    assert c.target_types==['wrench_abs'] and c.impedance_mode=='fixed'
    assert c.motion_control_axes_task==(0,)*6 and c.contact_wrench_control_axes_task==(1,)*6
    assert c.contact_wrench_stiffness_task==(.2,)*6
    assert c.gravity_compensation and not c.inertial_dynamics_decoupling
    assert c.nullspace_control=='none'
    assert a.wrench_scale==(100.,100.,100.,5.,5.,5.) and a.normalized_wrench_limit==1
    assert a.body_name=='axia80_link' and a.body_offset is None and a.task_frame_rel_path is None
    new.arm_action=deepcopy(old.arm_action)
    assert_ordered_equal(new.to_dict(),before)
    assert action_configs.WrenchActionsCfg().to_dict()==before


def test_force_loop_matches_native_isaac_and_moment_loop_is_extended(control,osc):
    cfg=osc.v0().actions.arm_action.controller_cfg.copy()
    cfg.target_types=['wrench_abs'];cfg.motion_control_axes_task=(0,)*6
    cfg.contact_wrench_control_axes_task=(1,)*6
    cfg.contact_wrench_stiffness_task=(.2,.2,.2,0,0,0)
    cfg.inertial_dynamics_decoupling=False
    native=osc.controller(cfg,2,'cpu')
    target=torch.tensor([[4.,-3.,2.,.3,-.4,.5],[0.,0.,0.,1.,1.,1.]])
    measured=torch.tensor([[1.,-2.,.5,.1,.2,-.1],[1.,1.,1.,.5,.6,.7]])
    jacobian=torch.randn(2,6,6,generator=torch.Generator().manual_seed(42))
    gravity=torch.randn(2,6,generator=torch.Generator().manual_seed(43))
    native.set_command(target)
    actual=control.sensor_wrench_efforts(target,measured,jacobian,torch.tensor(cfg.contact_wrench_stiffness_task),gravity)
    expected=native.compute(jacobian_b=jacobian,current_ee_force_b=measured[:,:3],gravity=gravity)
    torch.testing.assert_close(actual,expected)
    full=control.sensor_wrench_efforts(target,measured,jacobian,torch.full((6,),.2),gravity)
    correction=(jacobian[:,3:].mT @ (.2*(target[:,3:]-measured[:,3:])).unsqueeze(-1)).squeeze(-1)
    torch.testing.assert_close(full,expected+correction)


@pytest.mark.parametrize('axis',range(6))
def test_feedback_sign_equilibrium_and_error_each_axis(control,axis):
    desired=torch.zeros(1,6);desired[0,axis]=2
    jacobian=torch.eye(6)[None]
    gains=torch.full((6,),.2)
    equal=control.sensor_wrench_efforts(desired,desired,jacobian,gains)
    torch.testing.assert_close(equal,desired)
    under=control.sensor_wrench_efforts(desired,desired*.5,jacobian,gains)
    over=control.sensor_wrench_efforts(desired,desired*1.5,jacobian,gains)
    assert under[0,axis]==pytest.approx(2.2) and over[0,axis]==pytest.approx(1.8)
    assert under.count_nonzero()==over.count_nonzero()==1


def make_action(control,action_configs,count=2):
    common=importlib.import_module('_baseline_cpu_pkg.mdp.common.compensated_wrench')
    names=['base_link','axia80_link',*common.HAND_CONTACT_BODY_NAMES]
    n=len(names);nq=18;ids=[3,1,5,0,4,2]
    positions=torch.zeros(count,n,3);positions[:]=torch.tensor([2.,3.,4.])
    positions[:,2:,2]+=.0127
    quats=torch.zeros(count,n,4);quats[...,0]=1
    masses=torch.full((count,n),.01)
    inertia=torch.eye(3).repeat(count,n,1,1)*.001
    jacobian=torch.zeros(count,n-1,6,nq)
    jacobian[:,0,:,ids]=torch.eye(6)[None]
    gravity=torch.zeros(count,nq)
    raw=torch.zeros(count,n,6)
    child=names.index('inspire_base_link')
    raw[:,child,2]=len(common.HAND_CONTACT_BODY_NAMES)*.01*9.81
    captured=[]
    data=NS(body_link_pos_w=positions,body_link_quat_w=quats,
            body_com_pos_w=positions.clone(),body_com_vel_w=torch.zeros(count,n,6),
            body_com_acc_w=torch.zeros(count,n,6),body_incoming_joint_wrench_b=raw,
            joint_effort_limits=torch.full((count,nq),100.))
    view=NS(get_masses=lambda:masses,get_inertias=lambda:inertia.flatten(-2),
            get_jacobians=lambda:jacobian,get_gravity_compensation_forces=lambda:gravity)
    robot=NS(body_names=names,device='cpu',is_fixed_base=True,data=data,root_physx_view=view,
             find_joints=lambda names,preserve_order:(ids,list(names)),
             find_bodies=lambda name:([names.index(name)],[name]),
             set_joint_effort_target=lambda efforts,joint_ids:captured.append((efforts.clone(),list(joint_ids))))
    env=NS(num_envs=count,device='cpu',_sim_step_counter=0,scene={'robot':robot},
           cfg=NS(sim=NS(gravity=(0.,0.,-9.81))))
    action=control.SensorFrameWrenchAction(action_configs.WrenchV1ActionsCfg().arm_action,env)
    return action,env,robot,jacobian,gravity,captured


def test_real_action_sensor_origin_compensation_mapping_and_bounds(control,action_configs):
    a,env,robot,jacobian,gravity,captured=make_action(control,action_configs)
    child=robot.body_names.index('inspire_base_link')
    # Parent-on-child +5N X reaction, at joint plane 12.7mm past the sensor.
    robot.data.body_incoming_joint_wrench_b[:,child,0]=5.
    a.process_actions(torch.zeros(2,6))
    env._sim_step_counter=1;a.apply_actions()
    assert not a.current_wrench_s.any()  # first substep after reset is stale
    env._sim_step_counter=2;a.apply_actions()
    assert a.current_wrench_s[0].tolist()==pytest.approx([5,0,0,0,.0635,0],abs=2e-6)
    torch.testing.assert_close(a.joint_efforts,-.2*a.current_wrench_s)
    assert captured[-1][1]==[3,1,5,0,4,2]
    # World position (2,3,4) creates NO fictitious base-origin moment.
    assert a.joint_efforts[0,4]==pytest.approx(-.0127,abs=1e-6)
    raw=torch.tensor([[2.,-.5,0.,-3.,.25,1.],[float('inf'),0,0,0,0,0]])
    a.process_actions(raw)
    torch.testing.assert_close(a.raw_actions,raw)
    assert a.processed_actions[0].tolist()==[100.,-50.,0.,-5.,1.25,5.]
    robot.data.joint_effort_limits[:]=.5
    env._sim_step_counter=3;a.apply_actions()
    assert (a.joint_efforts.abs()<=.5).all() and a.torque_saturated.all()
    assert not a.joint_efforts[1].any()  # nonfinite policy input rejected, not promoted to a maximum command


def test_action_uses_current_sensor_axes_each_substep_and_no_pose_target(control,action_configs):
    a,env,robot,jacobian,gravity,captured=make_action(control,action_configs)
    # Gravity-free reading, feedforward + P when no contact exists.
    a.process_actions(torch.tensor([[.1,0,0,0,0,0]]).repeat(2,1))
    env._sim_step_counter=2;a.apply_actions()
    assert a.joint_efforts[0,0]==pytest.approx(12.)
    h=math.sqrt(.5)
    # Rotate the sensor only; current reading reference uses this quaternion.
    robot.data.body_link_quat_w[:,1]=torch.tensor([h,0,0,h])
    env._sim_step_counter=3;a.apply_actions()
    assert a.joint_efforts[0,:3].tolist()==pytest.approx([0,12.,0],abs=1e-5)
    # No set_command task translation / p x F term, and no EEF state is supplied.
    assert not hasattr(a,'_ee_pose_b')


def test_partial_reset_freshness_and_feedback_refresh_at_physics_rate(control,action_configs):
    a,env,robot,jacobian,gravity,captured=make_action(control,action_configs)
    a.process_actions(torch.ones(2,6)*.1)
    child=robot.body_names.index('inspire_base_link')
    env._sim_step_counter=2;a.apply_actions()
    old=a.current_wrench_s
    robot.data.body_incoming_joint_wrench_b[:,child,3]=.5
    env._sim_step_counter=3;a.apply_actions()
    assert not torch.equal(a.current_wrench_s,old)  # no policy-rate cache
    second=a.processed_actions[1].clone()
    a.reset(torch.tensor([0]))
    assert not a.current_wrench_s[0].any() and a.current_wrench_s[1,3]==pytest.approx(.5)
    env._sim_step_counter=4;a.apply_actions()
    assert not a.current_wrench_s[0].any() and a.current_wrench_s[1,3]==pytest.approx(.5)
    env._sim_step_counter=5;a.apply_actions()
    assert a.current_wrench_s[0,3]==pytest.approx(.5)
    torch.testing.assert_close(a.processed_actions[1],second)
    a.reset()
    assert not a.raw_actions.any() and not a.joint_efforts.any()


def test_env_inherits_v0_and_only_arm_config_changes(configs,omni_config,action_configs):
    class Scene(_AssetNode):
        def __init__(self,**kw):
            super().__init__(**kw)
            self.robot=_AssetNode(actuators={'arm':_AssetNode()},spawn=_AssetNode())
            self.shelf=_AssetNode(spawn=_AssetNode())
            self.object_collection=_AssetNode(rigid_objects={str(i):_AssetNode(spawn=_AssetNode()) for i in range(10)})
            self.ee_frame=_AssetNode(target_frames=[_AssetNode(prim_path='Robot/axia80_link',offset=_AssetNode(pos=(0,0,.1427),rot=(1,0,0,0)))])
        def configure_population_capacity(self,*args): assert args==(4,5,16)
    ns=dict(vars(omni_config.baseline));ns.update(vars(action_configs))
    ns.update(math=math,WrenchSceneCfg=Scene,WrenchObservationsCfg=_AssetNode,
              WrenchCommandsCfg=lambda:_AssetNode(target_goal_pos=_AssetNode()))
    for role,name in [('rewards','WrenchRewardsCfg'),('terminations','WrenchTerminationsCfg'),
                      ('events','WrenchEventsCfg'),('curricula','WrenchCurriculumCfg')]:
        ns[name]=getattr(configs(role,'wrench'),name)
    ns["WrenchV1RewardsCfg"] = configs("rewards", "wrench").WrenchV1RewardsCfg
    ns["WrenchV0_1RewardsCfg"] = configs("rewards", "wrench").WrenchV0_1RewardsCfg
    ns["WrenchV0_1CurriculumCfg"] = configs("curricula", "wrench").WrenchV0_1CurriculumCfg
    ns["WrenchV0_2RewardsCfg"] = configs("rewards", "wrench").WrenchV0_2RewardsCfg
    ns["WrenchV0_2CurriculumCfg"] = configs("curricula", "wrench").WrenchV0_2CurriculumCfg
    ns["feasibility_layout_stages"] = configs("curriculum_functions", "wrench").feasibility_layout_stages
    ns["reset_wrench_feasibility_scene"] = configs("event_functions", "wrench").reset_wrench_feasibility_scene
    m=_execute_definitions(MODULE_ROOT/'envs/wrench_envs.py',ns)
    parent=m.WrenchEnvCfg();before=parent.to_dict();child=m.WrenchV1EnvCfg()
    assert type(child).__bases__==(type(parent),)
    assert child.actions.arm_action.body_offset is None
    assert parent.actions.arm_action.body_offset.pos==(0,0,.1427)
    child.actions.arm_action=deepcopy(parent.actions.arm_action)
    assert child.rewards.robot_action_rate_penalty.func.__name__ == "desired_wrench_action_rate_penalty"
    child.rewards.robot_action_rate_penalty=deepcopy(parent.rewards.robot_action_rate_penalty)
    assert_ordered_equal(child.to_dict(),before)
    assert m.WrenchEnvCfg().to_dict()==before
    v1 = m.WrenchV1EnvCfg()
    v1_before = v1.to_dict()
    v2 = m.WrenchV2EnvCfg()
    assert type(v2).__bases__ == (type(v1),)
    assert v2.actions.arm_action.wrench_scale == (10., 10., 10., .5, .5, .5)
    assert v2.actions.arm_action.commanded_wrench_limit == (12., 12., 12., .6, .6, .6)
    v2.actions.arm_action.wrench_scale = v1.actions.arm_action.wrench_scale
    v2.actions.arm_action.commanded_wrench_limit = v1.actions.arm_action.commanded_wrench_limit
    assert_ordered_equal(v2.to_dict(), v1_before)
    assert m.WrenchV1EnvCfg().to_dict() == v1_before
    assert m.WrenchV2EnvCfg().actions.arm_action.wrench_scale == (10., 10., 10., .5, .5, .5)

    feasibility = m.WrenchV0_1EnvCfg()
    assert type(feasibility).__bases__ == (type(parent),)
    assert feasibility.rewards.target_contact is None
    term = feasibility.rewards.target_goal_distance
    assert term.func.__name__ == 'WrenchV0_1GoalDistance'
    assert term.weight == 3.7 and term.params == {'distance_scale': .05, 'threshold_n': .01}
    assert feasibility.curriculum.target_goal_direction_deadzone is None
    assert feasibility.events.object_spawn.params == {
        **parent.events.object_spawn.params, 'goal_distance': .075,
        'excluded_x_half_angle': math.radians(40),
    }
    assert feasibility.events.layout_bank.params == {
        'stages': ((40,3,3), (40,3,4), (40,4,4), (40,4,5)), 'min_goal_distance': .04,
    }
    # Restore only requested differences, then compare the entire ordered config.
    feasibility.rewards.target_contact = deepcopy(parent.rewards.target_contact)
    feasibility.rewards.target_goal_distance = deepcopy(parent.rewards.target_goal_distance)
    feasibility.curriculum.target_goal_direction_deadzone = deepcopy(parent.curriculum.target_goal_direction_deadzone)
    feasibility.events.object_spawn.params = deepcopy(parent.events.object_spawn.params)
    feasibility.events.layout_bank.params = deepcopy(parent.events.layout_bank.params)
    assert_ordered_equal(feasibility.to_dict(), before)
    assert m.WrenchEnvCfg().to_dict() == before

    sparse = m.WrenchV0_2EnvCfg()
    assert type(sparse).__bases__ == (type(parent),)
    assert sparse.rewards.target_goal_distance.func.__name__ == 'goal_distance_cosine'
    assert sparse.rewards.target_contact.func.__name__ == 'WrenchV0_2ContactMaintenance'
    assert sparse.events.object_spawn.params == {**parent.events.object_spawn.params,
                                                'gray_obstacles': -1, 'white_obstacles': -1}
    assert sparse.events.object_spawn.func is ns['reset_wrench_feasibility_scene']
    assert sparse.events.layout_bank.params == {'stages': ns['feasibility_layout_stages']()}
    assert sparse.curriculum.obstacle_population.func.__name__ == 'feasibility_obstacle_population_schedule'
    # Every other manager, numerical parameter, scene and action stays identical.
    sparse.rewards.target_goal_distance.func = parent.rewards.target_goal_distance.func
    sparse.rewards.target_contact.func = parent.rewards.target_contact.func
    sparse.events.object_spawn = deepcopy(parent.events.object_spawn)
    sparse.events.layout_bank = deepcopy(parent.events.layout_bank)
    sparse.curriculum.obstacle_population = deepcopy(parent.curriculum.obstacle_population)
    assert_ordered_equal(sparse.to_dict(), before)
    assert m.WrenchEnvCfg().to_dict() == before

    registry={};gym=NS(registry=registry,register=lambda **kw:registry.update({kw['id']:kw}))
    _execute_definitions(MODULE_ROOT/'__init__.py',{'gym':gym})
    assert registry['Inspire-Wrench-v0_2']['kwargs']['env_cfg_entry_point'].endswith(':WrenchV0_2EnvCfg')
    assert registry['Inspire-Wrench-v0_2']['kwargs']['rsl_rl_cfg_entry_point'].endswith(':WrenchV0_2PPORunnerCfg')
    entry = registry['Inspire-Wrench-v0_1']
    assert entry['entry_point'] == registry['Inspire-Wrench-v0']['entry_point']
    assert entry['kwargs']['env_cfg_entry_point'].endswith(':WrenchV0_1EnvCfg')
    assert entry['kwargs']['rsl_rl_cfg_entry_point'].endswith(':WrenchV0_1PPORunnerCfg')
    assert registry['Inspire-Wrench-v1']['entry_point']==registry['Inspire-Wrench-v0']['entry_point']
    assert registry['Inspire-Wrench-v1']['kwargs']['env_cfg_entry_point'].endswith(':WrenchV1EnvCfg')
    assert registry['Inspire-Wrench-v2']['entry_point'] == registry['Inspire-Wrench-v1']['entry_point']
    assert registry['Inspire-Wrench-v2']['kwargs']['env_cfg_entry_point'].endswith(':WrenchV2EnvCfg')
    assert registry['Inspire-Wrench-v2']['kwargs']['rsl_rl_cfg_entry_point'].endswith(':WrenchV2PPORunnerCfg')



def test_ppo_only_changes_experiment_name(omni_config):
    cc=omni_config.baseline.configclass
    @cc
    class Parent:
        experiment_name='inspire_wrench_v0'
        settings={'gamma':.98,'steps':36}
    m=_execute_definitions(MODULE_ROOT/'agent/wrench_ppo_cfg.py',
                          dict(configclass=cc,WrenchPPORunnerCfg=Parent),{'WrenchV1PPORunnerCfg'})
    a=Parent();b=m.WrenchV1PPORunnerCfg()
    assert type(b).__bases__==(Parent,)
    assert b.experiment_name=='inspire_wrench_v1'
    b.experiment_name=a.experiment_name
    assert b.to_dict()==a.to_dict()


@pytest.mark.parametrize('field,value',[
    ('target_types',['pose_rel']),('motion_control_axes_task',(1,)*6),
    ('contact_wrench_control_axes_task',(1,1,1,0,0,0)),
    ('contact_wrench_stiffness_task',None),('inertial_dynamics_decoupling',True),
])
def test_wrong_control_contract_rejected_before_simulation(control,action_configs,field,value):
    cfg=action_configs.WrenchV1ActionsCfg().arm_action
    setattr(cfg.controller_cfg,field,value)
    with pytest.raises(ValueError): control.SensorFrameWrenchAction(cfg,None)


def test_sensor_jacobian_virtual_power_and_gravity_is_added_once(control):
    generator=torch.Generator().manual_seed(17)
    jacobian=torch.randn(5,6,6,generator=generator)
    velocity=torch.randn(5,6,generator=generator)
    wrench=torch.randn(5,6,generator=generator)
    gravity=torch.randn(5,6,generator=generator)
    efforts=control.sensor_wrench_efforts(wrench,wrench,jacobian,torch.full((6,),.2),gravity)
    joint_power=((efforts-gravity)*velocity).sum(-1)
    sensor_velocity=(jacobian@velocity.unsqueeze(-1)).squeeze(-1)
    torch.testing.assert_close(joint_power,(sensor_velocity*wrench).sum(-1))


def test_corrected_command_bounds_before_gravity_and_joint_limits(control, action_configs):
    a, env, robot, jacobian, gravity, captured = make_action(control, action_configs)
    robot.data.joint_effort_limits[:] = 1000.
    gravity[:] = 2.
    # Opposing reaction makes FF+P exceed both force and moment limits.
    child = robot.body_names.index('inspire_base_link')
    robot.data.body_incoming_joint_wrench_b[:, child, 0] = -100.
    robot.data.body_incoming_joint_wrench_b[:, child, 3] = -5.
    a.process_actions(torch.ones(2, 6))
    env._sim_step_counter = 2
    a.apply_actions()
    assert a.joint_efforts[0, 0].item() == pytest.approx(122.)
    assert a.joint_efforts[0, 3].item() == pytest.approx(8.)
    # Negative commands are bounded symmetrically even with opposing feedback.
    robot.data.body_incoming_joint_wrench_b[:, child, 0] = 100.
    robot.data.body_incoming_joint_wrench_b[:, child, 3] = 5.
    a.process_actions(-torch.ones(2, 6))
    a.apply_actions()
    assert a.joint_efforts[0, 0].item() == pytest.approx(-118.)
    assert a.joint_efforts[0, 3].item() == pytest.approx(-4.)


def test_normalized_desired_wrench_rate_clips_targets_and_preserves_hand(configs):
    fn = configs('reward_functions', 'wrench').desired_wrench_action_rate_penalty
    # Same saturated desired target has zero rate despite different raw outputs.
    current = torch.tensor([[3.,0,0,0,0,0,2,0], [-1.,0,0,0,0,0,0,0],
                            [.5,0,0,.5,0,0,0,0]])
    previous = torch.tensor([[2.,0,0,0,0,0,0,0], [1.,0,0,0,0,0,0,0],
                             [0.,0,0,0,0,0,0,0]])
    cfg = NS(normalized_wrench_limit=1., wrench_scale=(100,100,100,5,5,5))
    env = NS(action_manager=NS(action=current, prev_action=previous,
                              get_term=lambda name: NS(cfg=cfg)))
    torch.testing.assert_close(fn(env), torch.tensor([.12,.12,.015]))
    # Target-range normalization remains correct with a non-unit raw limit.
    cfg.normalized_wrench_limit = 2.
    env.action_manager.action = current * 2
    env.action_manager.prev_action = previous * 2
    # Hand stays raw, so its cost scales by four.
    torch.testing.assert_close(fn(env), torch.tensor([.48,.12,.015]))


def test_v2_scaled_execution_and_normalized_action_rate(control, action_configs, configs):
    action, env, robot, jacobian, gravity, captured = make_action(control, action_configs)
    cfg = deepcopy(action.cfg)
    cfg.wrench_scale = tuple(v / 10 for v in cfg.wrench_scale)
    cfg.commanded_wrench_limit = tuple(v / 10 for v in cfg.commanded_wrench_limit)
    small = control.SensorFrameWrenchAction(cfg, env)
    raw = torch.tensor([[.5, 0, 0, .5, 0, 0, 0, 0]]).repeat(2, 1)
    rate = configs('reward_functions', 'wrench').desired_wrench_action_rate_penalty
    env.action_manager = NS(action=raw, prev_action=torch.zeros_like(raw), get_term=lambda name:small)
    small.process_actions(raw[:, :6])
    assert small.processed_actions[0].tolist() == [5., 0, 0, .25, 0, 0]
    torch.testing.assert_close(rate(env), torch.full((2,), .015))
    # Same relative v1 target: 50N / 2.5Nm, equal normalized cost.
    env.action_manager.get_term = lambda name:action
    action.process_actions(raw[:, :6])
    assert action.processed_actions[0].tolist() == [50., 0, 0, 2.5, 0, 0]
    torch.testing.assert_close(rate(env), torch.full((2,), .015))
    # Opposing measured contact cannot escape v2's corrected command limits.
    robot.data.joint_effort_limits[:] = 1000.
    child = robot.body_names.index('inspire_base_link')
    robot.data.body_incoming_joint_wrench_b[:, child, 0] = -100.
    robot.data.body_incoming_joint_wrench_b[:, child, 3] = -10.
    small.process_actions(torch.ones(2, 6))
    env._sim_step_counter = 2
    small.apply_actions()
    assert small.joint_efforts[0, 0].item() == pytest.approx(12.)
    assert small.joint_efforts[0, 3].item() == pytest.approx(.6)


def test_v2_runner_only_changes_experiment_name(omni_config):
    cc = omni_config.baseline.configclass
    @cc
    class Parent:
        experiment_name = 'inspire_wrench_v1'
        settings = {'gamma': .98, 'steps': 36}
    m = _execute_definitions(MODULE_ROOT/'agent/wrench_ppo_cfg.py',
                            dict(configclass=cc, WrenchV1PPORunnerCfg=Parent), {'WrenchV2PPORunnerCfg'})
    parent = Parent(); child = m.WrenchV2PPORunnerCfg()
    assert type(child).__bases__ == (Parent,)
    assert child.experiment_name == 'inspire_wrench_v2'
    child.experiment_name = parent.experiment_name
    assert child.to_dict() == parent.to_dict()
