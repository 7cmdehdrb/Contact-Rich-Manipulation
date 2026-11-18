"""Stabilize/Sweep public registration and sensor-to-task contracts."""
from types import SimpleNamespace as NS
from conftest import MODULE_ROOT
from test_omni_configuration import _execute_definitions


def test_registered_independent_environment_and_runner():
    entries = {}
    _execute_definitions(MODULE_ROOT / '__init__.py', {'gym': NS(
        registry=entries, register=lambda **kw: entries.update({kw['id']: kw}))})
    entry = entries['Inspire-Stabilize-Sweep-v0']
    assert entry['entry_point'] == 'inspire_sweep_baseline.runtime.stabilize_sweep:StabilizeSweepEnv'
    assert entry['kwargs']['env_cfg_entry_point'].endswith(':StabilizeSweepEnvCfg')
    assert entry['kwargs']['rsl_rl_cfg_entry_point'].endswith(':StabilizeSweepPPORunnerCfg')


def test_hand_starts_opposite_goal_with_surface_gap(load_mdp):
    import importlib, torch
    fn = importlib.import_module('_baseline_cpu_pkg.mdp.common.stabilize_sweep_reset').hand_pose
    # EEF -Z points along +X; hand's facing surface is 2cm ahead of EEF.
    q = torch.tensor([[.70710678, 0., -.70710678, 0.]])
    vertices = torch.tensor([[-.01,-.01,-.02],[.01,.01,0.]])
    p, rot = fn(torch.tensor([[0.,0.,1.10]]), torch.tensor([[0.,1.,0.]]),
                torch.tensor([.03]), torch.tensor([[0.,0.,1.13]]), q,
                vertices, torch.full((1,3),.5))
    torch.testing.assert_close(p,torch.tensor([[0.,-.10,1.13]]),atol=1e-6,rtol=0)
    from isaaclab.utils.math import quat_apply
    torch.testing.assert_close(quat_apply(rot,torch.tensor([[0.,0.,-1.]])),
                               torch.tensor([[0.,1.,0.]]),atol=1e-6,rtol=0)


def test_collision_boxes_reject_overlap_without_rejecting_separation(load_mdp):
    import importlib, torch
    fn=importlib.import_module('_baseline_cpu_pkg.mdp.common.stabilize_sweep_geometry').overlap
    center=torch.tensor([[[0.,0.,0.]]]);half=torch.tensor([[[.1,.02,.03]]])
    rotation=torch.eye(3).reshape(1,1,3,3)
    other=torch.tensor([[[.15,0.,0.],[0.,.05,0.],[.21,0.,0.]]])
    assert fn((center,half,rotation),(other,half.expand(1,3,3),rotation.expand(1,3,3,3))).tolist()==[[[True,False,False]]]


def test_surface_snapshot_keeps_disconnected_obstacles_and_real_hits(load_mdp):
    import importlib, numpy as np
    module=importlib.import_module('_baseline_cpu_pkg.mdp.common.stabilize_sweep_surface')
    # A narrow foreground object divides a wider rear obstacle into two masks.
    vertices=[];faces=[];owners=[]
    for owner,(center,size) in enumerate([((0.,.2,0.),(.1,.06,.1)),
                                        ((0.,0.,0.),(.1,.12,.1)),
                                        ((.12,0.,0.),(.02,.02,.12))]):
        v,f=module.box_mesh(center,size)
        faces.extend(f+len(vertices));vertices.extend(v);owners.extend([owner]*len(f))
    target,obstacles,_=module.observe_mesh(vertices,faces,owners,device='cpu')
    assert target is not None and len(obstacles)==3
    np.testing.assert_allclose(target[0],.05,atol=1e-6)
    rear=obstacles[np.isclose(obstacles[:,0],.05)]
    assert len(rear)==2 and rear[:,1].min()<-.01 and rear[:,1].max()>.01


def test_actor_critic_67d_gradients_and_export(load_mdp):
    import importlib, torch
    from tensordict import TensorDict
    cls=importlib.import_module('_baseline_cpu_pkg.agent.stabilize_sweep_models').StabilizeSweepModel
    obs=TensorDict({'policy':torch.randn(8,67),'obstacles':torch.randn(8,64)},batch_size=[8])
    obs['obstacles'][:,::4]=1
    groups={'actor':['policy','obstacles'],'critic':['policy','obstacles']}
    actor=cls(obs,groups,'actor',12,distribution_cfg={'class_name':'GaussianDistribution','init_std':1.})
    critic=cls(obs,groups,'critic',1)
    for model,width in ((actor,12),(critic,1)):
        result=model(obs)
        assert result.shape==(8,width) and model.get_latent(obs).shape==(8,83)
        result.square().mean().backward()
        assert model.obstacle_encoder.mlp[0].weight.grad.abs().sum()>0
    export=torch.jit.script(actor.as_jit())
    torch.testing.assert_close(export(torch.cat((obs['policy'],obs['obstacles']),-1)),actor(obs))
    assert actor.as_onnx().get_dummy_inputs()[0].shape==(1,131)


def test_contact_points_keep_couple_and_squeeze_when_net_force_cancels(load_mdp):
    import importlib,torch
    fn=importlib.import_module('_baseline_cpu_pkg.mdp.common.stabilize_sweep_contact').reduce_contacts
    force=torch.tensor([[0.,2.,0.],[0.,-2.,0.],[float('nan'),0.,0.]])
    point=torch.tensor([[.1,0.,0.],[-.1,0.,0.],[0.,0.,0.]])
    counts=torch.tensor([[2]]);starts=torch.tensor([[0]])
    total,moment,load,valid=fn(force,point,counts,starts,torch.zeros(1,3))
    torch.testing.assert_close(total,torch.zeros(1,3))
    torch.testing.assert_close(moment,torch.tensor([[0.,0.,.4]]))
    torch.testing.assert_close(load,torch.tensor([4.]))
    assert valid.tolist()==[True]
    *_,valid=fn(force,point,torch.tensor([[4]]),starts,torch.zeros(1,3))
    assert valid.tolist()==[False]


def test_layout_skips_large_candidate_and_keeps_smaller_candidate(load_mdp):
    import importlib,torch
    layout_cls=importlib.import_module('_baseline_cpu_pkg.mdp.common.wrench_refactor_layout').CircleResetLayout
    place=importlib.import_module('_baseline_cpu_pkg.mdp.common.stabilize_sweep_layout').place_obstacles
    bounds=torch.tensor([[-.02,-.02,0.],[.02,.02,.06]]).expand(1,17,2,3).clone()
    bounds[:,1,0,:2]=-.3;bounds[:,1,1,:2]=.3
    layout=layout_cls(bounds,torch.full((1,17),1.05))
    local=torch.zeros(1,17,3);local[...,2]=50.;local[:,0,2]=1.05
    yaw=torch.zeros(1,17);active=torch.zeros(1,17,dtype=torch.bool);active[:,0]=True
    robot=(torch.tensor([[[5.,5.,5.]]]),torch.full((1,1,3),.01),torch.eye(3).reshape(1,1,3,3))
    place(layout,torch.tensor([0]),local,yaw,active,torch.tensor([[0.,.08]]),robot,
          torch.tensor([[1,2,3,4,5,6]]),torch.tensor([2]),torch.Generator().manual_seed(1))
    assert active[0,1].item() is False
    assert active[0,2].item() is True and active.sum()==2


def test_gate_requires_30_samples_and_changes_reward_phase_next_step(load_mdp):
    import importlib,torch
    state=importlib.import_module('_baseline_cpu_pkg.mdp.common.stabilize_sweep_state').StabilizeState(1,'cpu')
    center=torch.zeros(1,3);q=torch.tensor([[1.,0.,0.,0.]]);goal=torch.tensor([[1.,0.,0.]])
    state.reset(torch.tensor([0]),center,q,goal)
    def sample(moment=None):
        state.sample(center,q,goal,torch.tensor([[.5,0.,0.]]),torch.zeros(1,3) if moment is None else moment,
                     torch.tensor([.5]),torch.tensor([True]),torch.zeros(1),torch.zeros(1))
    state.begin()
    for _ in range(29):sample()
    assert not state.achieved.item()
    sample(torch.tensor([[0.,0.,.006]]))
    for _ in range(29):sample()
    assert not state.achieved.item()
    sample()
    assert state.achieved.item() and not state.reward_phase.item()
    state.begin();sample(torch.tensor([[0.,0.,100.]]))
    assert state.achieved.item() and state.stage_score.item()==1.
    state.reset(torch.tensor([0]),center,q,goal)
    assert not state.achieved.item()


def test_setup_scores_each_sample_before_averaging(load_mdp):
    import importlib,torch,math
    state=importlib.import_module('_baseline_cpu_pkg.mdp.common.stabilize_sweep_state').StabilizeState(1,'cpu')
    p=torch.zeros(1,3);q=torch.tensor([[1.,0.,0.,0.]]);g=torch.tensor([[1.,0.,0.]])
    state.reset(torch.tensor([0]),p,q,g);state.begin()
    for sign in (-1.,1.):
        state.sample(p,q,g,torch.tensor([[.5,0.,0.]]),torch.tensor([[0.,0.,sign*.005]]),
                     torch.tensor([.5]),torch.tensor([True]),torch.zeros(1),torch.zeros(1))
    torch.testing.assert_close(state.stage_score,torch.tensor([math.exp(-1)]))


def test_sweep_uses_previous_distance_and_normalized_contact_without_retroactive_credit(load_mdp):
    import importlib,torch
    cls=importlib.import_module('_baseline_cpu_pkg.mdp.common.stabilize_sweep_state').StabilizeState
    state=cls(1,'cpu');p=torch.zeros(1,3);q=torch.tensor([[1.,0.,0.,0.]]);goal=torch.tensor([[1.,0.,0.]])
    state.reset(torch.tensor([0]),p,q,goal)
    def finish(x,contact=True):
        state.finish(torch.tensor([[x,0.,0.]]),goal,torch.tensor([[2.,0.,0.]]),torch.tensor([contact]),
                     torch.tensor([True]),torch.tensor([1.10]),torch.tensor([1.05]))
    state.begin();state.achieved[:]=True;finish(.5)
    assert state.positive.item()==0 and state.retreat.item()==0
    state.begin();finish(.51)
    assert state.positive.item()==0
    state.begin();finish(.52)
    torch.testing.assert_close(state.positive,torch.tensor([.0005]),atol=1e-7,rtol=0)
    state.begin();finish(.50,False)
    torch.testing.assert_close(state.retreat,torch.tensor([.02]),atol=1e-7,rtol=0)
    assert state.positive.item()==0


from test_sweep_simple import family,configs,omni_config


def test_existing_costs_curricula_and_independence(family,configs):
    old=family.SweepSimpleFingerForceDirectionEnvCfg()
    new= configs('rewards','stabilize_sweep').StabilizeSweepRewardsCfg()
    for name in ('arm_action_rate','hand_action_rate','arm_joint_velocity','hand_joint_velocity',
                 'shelf_collision','shelf_failure','obstacle_contact','object_tilt','obstacle_displacement'):
        assert getattr(new,name).to_dict()==getattr(old.rewards,name).to_dict(),name
    expected=old.rewards.wrench_change.to_dict()
    expected['params']['max_positive_fraction']=None
    assert new.wrench_change.to_dict()==expected
    curr=configs('curricula','stabilize_sweep').StabilizeSweepCurriculumCfg()
    for name in ('wrench_free_force_change','wrench_free_moment_change','obstacle_free_force','wrench_weight','obstacle_weight'):
        assert getattr(curr,name).to_dict()==getattr(old.curriculum,name).to_dict()
    assert curr.target_goal_direction_deadzone is None and curr.arm_joint_velocity_threshold is None
    new.arm_action_rate.weight=999
    assert old.rewards.arm_action_rate.weight==-.03


def test_convex_collision_rejects_empty_corner_overlap(load_mdp):
    import importlib,numpy as np
    from scipy.spatial import ConvexHull
    fn=importlib.import_module('_baseline_cpu_pkg.mdp.common.stabilize_sweep_geometry').convex_overlap
    # Two tetrahedra have overlapping bounding boxes, but disjoint interiors.
    a=np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]])
    b=np.array([[1.,1.,1.],[0.,1.,1.],[1.,0.,1.],[1.,1.,0.]])
    assert not fn(ConvexHull(a).equations,ConvexHull(b).equations)
    assert fn(ConvexHull(a).equations,ConvexHull(a+.01).equations)


def test_timeout_success_is_strict_and_physical_failure_wins(load_mdp):
    import importlib,torch
    fn=importlib.import_module('_baseline_cpu_pkg.mdp.termination_functions.stabilize_sweep').success_at_timeout
    data=NS(object_link_pos_w=torch.tensor([[.029,0.,0.],[.03,0.,0.],[.029,0.,0.],[.029,0.,0.]])[:,None,:],
            object_link_quat_w=torch.tensor([[[1.,0.,0.,0.]]]).expand(4,1,-1))
    failure=torch.tensor([False,False,False,True])
    env=NS(scene={'object_collection':NS(data=data)},_revision_bounds_t=torch.zeros(4,17,2,3),
           command_manager=NS(get_command=lambda _:torch.zeros(4,7)),episode_length_buf=torch.full((4,),500),
           max_episode_length=500,_sweep_phase=torch.tensor([True,True,False,True]),
           termination_manager=NS(get_term=lambda _:failure))
    assert fn(env).tolist()==[True,False,False,False]
    env.episode_length_buf[:]=499
    assert not fn(env).any()


def test_snapshot_target_missing_fallback_is_bounded_and_removes_obstacles(load_mdp,monkeypatch):
    import importlib,sys,torch,numpy as np
    from types import ModuleType
    module=ModuleType('isaaclab.sim.utils');module.get_current_stage=lambda:None
    monkeypatch.setitem(sys.modules,'isaaclab.sim.utils',module)
    snapshot=importlib.import_module('_baseline_cpu_pkg.mdp.event_functions.stabilize_sweep').snapshot
    class Geometry:
        def __init__(self,always_missing=False):self.calls=0;self.always_missing=always_missing
        def observe(self,path,local,yaw,active,names,device):
            self.calls+=1
            point=None if active[1:].any() or self.always_missing else np.array([.1,.2,.3],dtype=np.float32)
            return point,np.empty((0,3)),{'total_ms':1.}
    geometry=Geometry()
    env=NS(_surface_geometry=geometry,scene=NS(env_prim_paths=['env']),device='cpu',
           cfg=NS(scene=NS(object_collection=NS(rigid_objects=dict.fromkeys(range(17))))))
    local=torch.zeros(1,17,3);yaw=torch.zeros(1,17);active=torch.ones(1,17,dtype=torch.bool)
    shelf=torch.tensor([[1.,2.,3.,1.,0.,0.,0.,0.,0.,0.,0.,0.,0.]])
    target,others,valid,timing=snapshot(env,torch.tensor([0]),local,yaw,active,shelf)
    torch.testing.assert_close(target,torch.tensor([[1.1,2.2,3.3]]))
    assert geometry.calls==2 and active.sum()==1 and not valid.any()
    assert timing[0]['target_missing']==1 and timing[0]['fallback']==1
    env._surface_geometry=Geometry(always_missing=True)
    import pytest
    with pytest.raises(RuntimeError,match='Target-only'):
        snapshot(env,torch.tensor([0]),local,yaw,active,shelf)
    assert env._surface_geometry.calls==2
