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
    # A rightward goal uses the dorsum (EEF -Z), while the palm stays left.
    q = torch.tensor([[.70710678, 0., .70710678, 0.]])
    vertices = torch.tensor([[-.01,-.01,0.],[.01,.01,.02]])
    p, rot = fn(torch.tensor([[0.,0.,1.10]]), torch.tensor([[0.,1.,0.]]),
                torch.tensor([.03]), torch.tensor([[0.,0.,1.13]]), q,
                vertices, torch.full((1,3),.5),torch.tensor([[.7,0.,.79505]]),torch.tensor([-.1427,0.,0.]),target_half_height=torch.tensor([.05]))
    torch.testing.assert_close(p,torch.tensor([[0.,-.08,1.065]]),atol=1e-6,rtol=0)
    from isaaclab.utils.math import quat_apply
    torch.testing.assert_close(quat_apply(rot,torch.tensor([[0.,0.,1.]])),
                               torch.tensor([[0.,-1.,0.]]),atol=1e-6,rtol=0)


def test_initial_hand_keeps_palm_left_and_places_mount_toward_base(load_mdp):
    import importlib, math, torch
    from isaaclab.utils.math import quat_apply
    fn=importlib.import_module('_baseline_cpu_pkg.mdp.common.stabilize_sweep_reset').hand_pose
    angles=torch.tensor([30.,60.,90.,120.,150.,210.,240.,270.,300.,330.])*math.pi/180
    direction=torch.stack((angles.cos(),angles.sin(),torch.zeros_like(angles)),-1)
    n=len(angles)
    center=torch.tensor([[0.,0.,1.10]]).expand(n,-1)
    base=torch.tensor([[.7,0.,.79505]]).expand(n,-1)
    default=torch.tensor([[0.,0.,1.211461]]).expand(n,-1)
    q=torch.tensor([[.70710678,0.,-.70710678,0.]]).expand(n,-1)
    mount=torch.tensor([-.1427,0.,0.])
    vertices=torch.tensor([[-.01,-.01,-.02],[.01,.01,0.]])
    p,rot=fn(center,direction,torch.full((n,),.03),default,q,vertices,
             torch.full((n,3),.5),base,mount,target_half_height=torch.full((n,),.05))
    palm=quat_apply(rot,torch.tensor([0.,0.,1.]).expand(n,-1))
    assert (palm[:,1]<0).all()
    expected=torch.where(direction[:,1:2]>0,-direction,direction)
    torch.testing.assert_close(palm,expected,atol=1e-6,rtol=0)
    toward=base-center;toward[:,2]=0.
    toward-=direction*(toward*direction).sum(-1,keepdim=True)
    toward/=toward.norm(dim=-1,keepdim=True)
    torch.testing.assert_close(quat_apply(rot,mount.expand(n,-1)),.1427*toward,atol=1e-6,rtol=0)
    surface=p[:,None]+quat_apply(rot[:,None].expand(-1,2,-1),vertices.expand(n,-1,-1))
    gap=((center[:,None]-surface)*direction[:,None]).sum(-1).amin(-1)-.03
    torch.testing.assert_close(gap,torch.full((n,),.05),atol=1e-6,rtol=0)
    torch.testing.assert_close(surface[...,2].amin(-1),torch.full((n,),1.055),atol=1e-6,rtol=0)


def test_hand_bottom_tracks_shelf_for_target_sizes_roll_and_yaw_jitter(load_mdp):
    import importlib, math, torch
    from isaaclab.utils.math import quat_apply
    fn=importlib.import_module('_baseline_cpu_pkg.mdp.common.stabilize_sweep_reset').hand_pose
    # Short/tall targets share a shelf. A deliberately high default EEF must not
    # lift the hand; both contact faces and the two wrist elevations are covered.
    half=torch.tensor([.045,.055,.045,.055])
    center=torch.stack((torch.zeros(4),torch.zeros(4),1.05+half),-1)
    direction=torch.tensor([[0.,1.,0.],[0.,-1.,0.],[0.,1.,0.],[0.,-1.,0.]])
    default=torch.tensor([[0.,0.,2.]]).expand(4,-1)
    q=torch.tensor([[.70710678,0.,-.70710678,0.]]).expand(4,-1)
    vertices=torch.tensor([[-.10,-.02,-.03],[.02,.03,.02]])
    draws=torch.tensor([[0.,1.,0.],[1.,0.,1.],[0.,1.,0.],[1.,0.,1.]])
    for elevation in (0.,math.radians(2)):
        p,rot=fn(center,direction,torch.full((4,),.04),default,q,vertices,draws,
                 torch.tensor([[.7,0.,.79505]]).expand(4,-1),torch.tensor([-.1427,0.,0.]),
                 elevation,target_half_height=half)
        points=p[:,None]+quat_apply(rot[:,None].expand(-1,2,-1),vertices.expand(4,-1,-1))
        bottom=points[...,2].amin(-1)
        torch.testing.assert_close(bottom,torch.full((4,),1.055),atol=1e-6,rtol=0)
        assert (bottom<center[:,2]+half).all()
        normal=quat_apply(rot,torch.tensor([0.,0.,1.]).expand(4,-1))
        assert (normal[:,1]<0).all()
        torch.testing.assert_close((normal*direction).sum(-1).abs(),
                                   torch.full((4,),math.cos(math.radians(10))),atol=1e-6,rtol=0)


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


def test_target_surface_width_exposure_and_largest_fragment(load_mdp):
    import importlib, numpy as np
    module=importlib.import_module('_baseline_cpu_pkg.mdp.common.stabilize_sweep_surface')
    def observe(width, occluder=None):
        vertices,faces=module.box_mesh((0.,0.,0.),(.1,width,.1))
        owners=np.zeros(len(faces),dtype=np.int32)
        if occluder is not None:
            y,cover_width=occluder
            v,f=module.box_mesh((.12,y,0.),(.02,cover_width,.12))
            owners=np.concatenate((owners,np.ones(len(f),dtype=np.int32)))
            faces=np.concatenate((faces,f+len(vertices)));vertices=np.concatenate((vertices,v))
        return module.observe_mesh(vertices,faces,owners,device='cpu')
    assert observe(.018)[0] is None
    assert observe(.022)[0] is not None
    # 3cm is over the absolute minimum, but under 20% of the 20cm source.
    assert observe(.2,(.015,.17))[0] is None
    assert observe(.2,(.025,.15))[0] is not None
    target,others,timing=observe(.12,(.01,.02))
    assert target is not None and target[1]<0 and len(others)==1
    assert all(timing[key]>=0 for key in ('upload_ms','readback_ms','ray_ms','components_ms'))


import pytest


@pytest.mark.parametrize('violation',[
    'invalid','no_contact','undefined_direction','nonfinite','parallel_low','parallel_high',
    'perpendicular','moment_z','moment_xy','translation','rotation','linear_speed','angular_speed','squeeze',
])
def test_each_gate_condition_breaks_continuity(load_mdp,violation):
    import importlib,torch,math
    cls=importlib.import_module('_baseline_cpu_pkg.mdp.common.stabilize_sweep_state').StabilizeState
    state=cls(1,'cpu');p=torch.zeros(1,3);q=torch.tensor([[1.,0.,0.,0.]]);g=torch.tensor([[1.,0.,0.]])
    state.reset(torch.tensor([0]),p,q,g);state.begin()
    force=torch.tensor([[.5,0.,0.]]);moment=torch.zeros(1,3);load=torch.tensor([.5])
    valid=torch.tensor([True]);linear=torch.zeros(1);angular=torch.zeros(1)
    state.consecutive[:]=29
    if violation=='invalid':valid[:]=False
    elif violation=='no_contact':load[:]=.009
    elif violation=='undefined_direction':g.zero_()
    elif violation=='nonfinite':force[0,0]=float('nan')
    elif violation=='parallel_low':force[0,0]=.299
    elif violation=='parallel_high':force[0,0]=.701
    elif violation=='perpendicular':force[0,1]=.201
    elif violation=='moment_z':moment[0,2]=.0051
    elif violation=='moment_xy':moment[0,0]=.0101
    elif violation=='translation':p[0,2]=.0031
    elif violation=='rotation':q[0]=torch.tensor([math.cos(math.radians(3.1)/2),math.sin(math.radians(3.1)/2),0.,0.])
    elif violation=='linear_speed':linear[:]=.0051
    elif violation=='angular_speed':angular[:]=.0501
    elif violation=='squeeze':load[:]=5.001
    state.sample(p,q,g,force,moment,load,valid,linear,angular)
    assert state.consecutive.item()==0 and not state.achieved.item()
    if violation in ('invalid','no_contact','undefined_direction','nonfinite'):
        assert state.stage_score.item()==0


def test_initial_pose_cost_allowances_and_sweep_dt_cancellation(load_mdp):
    import importlib,torch,math
    cls=importlib.import_module('_baseline_cpu_pkg.mdp.common.stabilize_sweep_state').StabilizeState
    reward=importlib.import_module('_baseline_cpu_pkg.mdp.reward_functions.stabilize_sweep')
    state=cls(1,'cpu');p=torch.zeros(1,3);q=torch.tensor([[1.,0.,0.,0.]]);g=torch.tensor([[1.,0.,0.]])
    state.reset(torch.tensor([0]),p,q,g);state.begin()
    for ratio,expected in ((.999,0.),(2.,1.)):
        angle=ratio*math.radians(3.)
        rotated=torch.tensor([[math.cos(angle/2),0.,0.,math.sin(angle/2)]])
        state.sample(torch.tensor([[0.,0.,ratio*.003]]),rotated,g,torch.tensor([[.5,0.,0.]]),
                     torch.zeros(1,3),torch.tensor([.5]),torch.tensor([True]),torch.zeros(1),torch.zeros(1))
        torch.testing.assert_close(state.move_cost,torch.tensor([expected]),atol=1e-6,rtol=0)
        torch.testing.assert_close(state.rotate_cost,torch.tensor([expected]),atol=1e-6,rtol=0)
    state.positive[:]=.02;state.retreat[:]=.03
    for dt in (.01,.02,.04):
        env=NS(_stabilize_state=state,step_dt=dt)
        torch.testing.assert_close(reward.sweep_progress(env)*dt,torch.tensor([.02]))
        torch.testing.assert_close(reward.sweep_retreat(env)*dt,torch.tensor([.03]))


@pytest.mark.parametrize('force,contact,valid,height,expected',[
    ((1.,0.,0.),True,True,.119,True),
    ((.999,0.,0.),True,True,.119,False),
    ((1.,0.,0.),False,True,.119,False),
    ((1.,0.,0.),True,False,.119,False),
    ((1.,0.,0.),True,True,.12,False),
    ((-1.,0.,0.),True,True,.119,False),
])
def test_sweep_contact_height_and_force_boundaries(load_mdp,force,contact,valid,height,expected):
    import importlib,torch
    cls=importlib.import_module('_baseline_cpu_pkg.mdp.common.stabilize_sweep_state').StabilizeState
    state=cls(1,'cpu');p=torch.zeros(1,3);q=torch.tensor([[1.,0.,0.,0.]]);g=torch.tensor([[1.,0.,0.]])
    state.reset(torch.tensor([0]),p,q,g);state.achieved[:]=True;state.begin()
    state.contact_history[:]=True;state.contact_count[:]=20
    state.finish(torch.tensor([[.02,0.,0.]]),g,torch.tensor([force]),torch.tensor([contact]),
                 torch.tensor([valid]),torch.tensor([height]),torch.zeros(1))
    assert bool(state.positive.item()>0)==expected


def test_squeeze_is_per_sample_before_policy_average(load_mdp):
    import importlib,torch
    module=importlib.import_module('_baseline_cpu_pkg.mdp.common.stabilize_sweep_contact')
    reward=importlib.import_module('_baseline_cpu_pkg.mdp.reward_functions.stabilize_sweep')
    # External PhysX view fixture: spare packed entry avoids a full-buffer rejection.
    class View:
        sensor_count=1;filter_count=1;load=5.
        def get_contact_data(self,dt):
            return (torch.tensor([[self.load],[0.]]),torch.zeros(2,3),torch.tensor([[1.,0.,0.],[0.,0.,0.]]),
                    None,torch.tensor([[1]]),torch.tensor([[0]]))
        def get_friction_data(self,dt):
            return torch.zeros(2,3),torch.zeros(2,3),torch.tensor([[0]]),torch.tensor([[0]])
    view=View()
    env=NS(num_envs=1,device='cpu',physics_dt=.01,scene={
        'object_collection':NS(object_names=['target'],data=NS(object_com_pos_w=torch.zeros(1,1,3))),
        module.object_hand_sensor_name('target'):NS(contact_physx_view=view)})
    samples=module.ContactSamples(env);env._stabilize_contacts=samples;samples.begin()
    samples.capture(env)
    assert reward.squeeze(env).item()==0.
    view.load=10.;samples.capture(env)
    # mean(0,1)=.5; applying the square after averaging load would give .25.
    assert reward.squeeze(env).item()==.5


def _reset_layout_planner():
    import math, torch
    from typing import NamedTuple
    from time import perf_counter
    scope=dict(torch=torch,math=math,NamedTuple=NamedTuple,perf_counter=perf_counter)
    _execute_definitions(MODULE_ROOT/'mdp/event_functions/stabilize_sweep.py',scope)
    return scope


def test_failed_initial_layout_redraws_only_failed_rows_and_retains_successes():
    import torch
    scope=_reset_layout_planner()
    calls=[];draws=[]
    def sample(bounds,generator,**params):
        target=torch.rand((len(bounds),2),generator=generator)
        goal=target+.1
        draws.append((target.clone(),goal.clone()))
        return target,goal
    def choose(env,ids,local,delta,shelf,root):
        calls.append(ids.tolist());n=len(ids)
        valid=torch.ones(n,dtype=torch.bool)
        if len(calls)==1:valid[-1]=False
        marker=torch.full((n,3),float(len(calls)))
        pose=scope['InitialPose'](marker.clone(),(marker.clone(),marker.clone(),marker.clone()),
            marker.clone(),torch.ones(n,4),torch.zeros(n,2),torch.where(valid,1,4))
        return pose,valid,[]
    scope.update(sample_commands=sample,choose_initial_pose=choose)
    env=NS(_revision_bounds_t=torch.zeros(3,17,2,3),_revision_layout_rng=torch.Generator().manual_seed(7))
    local=torch.zeros(2,17,3);local[:,0,2]=1.05
    delta,pose,resamples=scope['plan_initial_layout'](env,torch.tensor([2,0]),local,
        torch.zeros(2,13),torch.zeros(2,13),{},max_layout_resamples=2)
    assert calls==[[2,0],[0]]
    assert resamples.tolist()==[0,1]
    assert pose.attempts.tolist()==[1,5]
    torch.testing.assert_close(local[0,0,:2],draws[0][0][0])
    torch.testing.assert_close(local[1,0,:2],draws[1][0][0])
    torch.testing.assert_close(local[:,0,2],torch.full((2,),1.05))
    torch.testing.assert_close(delta,torch.full((2,2),.1))
    for field in (pose.joints,*pose.boxes,pose.position_w):
        torch.testing.assert_close(field,torch.tensor([[1.,1.,1.],[2.,2.,2.]]))
    expected_local=local.clone();expected_joints=pose.joints.clone()
    calls.clear();draws.clear();env._revision_layout_rng.manual_seed(7)
    local[:,0,:2]=0.
    _,again,counts=scope['plan_initial_layout'](env,torch.tensor([2,0]),local,
        torch.zeros(2,13),torch.zeros(2,13),{},max_layout_resamples=2)
    torch.testing.assert_close(local,expected_local)
    torch.testing.assert_close(again.joints,expected_joints)
    assert counts.tolist()==[0,1]


def test_reset_layout_budget_is_bounded_and_does_not_commit_invalid_poses():
    import torch,pytest
    scope=_reset_layout_planner();calls=[]
    def sample(bounds,generator,**params):return torch.zeros(len(bounds),2),torch.ones(len(bounds),2)
    def choose(env,ids,local,delta,shelf,root):
        calls.append(ids.tolist());n=len(ids);v=torch.zeros(n,3)
        pose=scope['InitialPose'](v,(v.clone(),v.clone(),v.clone()),v.clone(),torch.zeros(n,4),
                                  torch.ones(n,2),torch.full((n,),4))
        return pose,torch.zeros(n,dtype=torch.bool),[{'test_failure':True}]
    scope.update(sample_commands=sample,choose_initial_pose=choose)
    env=NS(_revision_bounds_t=torch.zeros(1,17,2,3),_revision_layout_rng=torch.Generator())
    # This planning seam has no scene or simulator, so no failed pose can be written.
    with pytest.raises(RuntimeError,match='exhausted 3 layouts'):
        scope['plan_initial_layout'](env,torch.tensor([0]),torch.zeros(1,17,3),
            torch.zeros(1,13),torch.zeros(1,13),{},max_layout_resamples=2)
    assert calls==[[0],[0],[0]]
    def programming_error(*args):raise RuntimeError('unrelated solver failure')
    scope['choose_initial_pose']=programming_error
    with pytest.raises(RuntimeError,match='unrelated solver failure'):
        scope['plan_initial_layout'](env,torch.tensor([0]),torch.zeros(1,17,3),
            torch.zeros(1,13),torch.zeros(1,13),{},max_layout_resamples=2)


def test_registered_sweep_only_child_and_separate_runner():
    entries={}
    _execute_definitions(MODULE_ROOT/'__init__.py',{'gym':NS(
        registry=entries,register=lambda **kw:entries.update({kw['id']:kw}))})
    child=entries['Inspire-Sweep-Only-v0']
    assert child['entry_point'].endswith(':SweepOnlyEnv')
    assert child['kwargs']['env_cfg_entry_point'].endswith(':SweepOnlyEnvCfg')
    assert child['kwargs']['rsl_rl_cfg_entry_point'].endswith(':SweepOnlyPPORunnerCfg')
    assert entries['Inspire-Stabilize-Sweep-v0']['entry_point'].endswith(':StabilizeSweepEnv')


def test_sweep_only_bypasses_gate_from_reset_and_keeps_sweep_contact_formula(load_mdp):
    import importlib,torch
    module=importlib.import_module('_baseline_cpu_pkg.mdp.common.stabilize_sweep_state')
    parent=module.StabilizeState(2,'cpu');child=module.SweepOnlyState(2,'cpu')
    ids=torch.arange(2);center=torch.zeros(2,3);quat=torch.tensor([[1.,0.,0.,0.]]).expand(2,-1)
    goal=torch.tensor([[.1,0.,0.]]).expand(2,-1)
    for state in (parent,child):state.reset(ids,center,quat,goal)
    assert not parent.achieved.any()
    assert child.achieved.all() and child.reward_phase.all()
    for step in range(2):
        for state in (parent,child):
            state.begin()
            # Deliberately violates Gate: no stable force, high speed and moment.
            state.sample(center,quat,goal,torch.zeros(2,3),torch.ones(2,3),
                         torch.zeros(2),torch.zeros(2,dtype=torch.bool),torch.ones(2),torch.ones(2))
            state.finish(center+torch.tensor([.01*(step+1),0.,0.]),goal,
                         torch.tensor([[2.,0.,0.]]).expand(2,-1),torch.ones(2,dtype=torch.bool),
                         torch.ones(2,dtype=torch.bool),torch.full((2,),1.1),torch.full((2,),1.05))
    assert not parent.achieved.any() and not parent.positive.any()
    assert child.achieved.all() and not child.consecutive.any()
    assert not child.setup_sum.any() and not child.move_cost.any() and not child.rotate_cost.any()
    # Keep the existing recent-contact recognition and normalized count (1/20).
    torch.testing.assert_close(child.positive,torch.full((2,),.01/20))
    child.begin()
    child.finish(center,goal,torch.zeros(2,3),torch.zeros(2,dtype=torch.bool),
                 torch.ones(2,dtype=torch.bool),torch.full((2,),1.1),torch.full((2,),1.05))
    torch.testing.assert_close(child.retreat,torch.full((2,),.02))
    held=child.previous_distance[1].clone();child.reset(torch.tensor([0]),center[:1],quat[:1],goal[:1])
    assert child.achieved.all() and child.reward_phase.all()
    assert child.contact_count[0]==0 and child.positive[0]==0 and child.retreat[0]==0
    torch.testing.assert_close(child.previous_distance[1],held)


def test_finger_test_tasks_keep_each_parent_runtime():
    entries={}
    _execute_definitions(MODULE_ROOT/'__init__.py',{'gym':NS(
        registry=entries,register=lambda **kw:entries.update({kw['id']:kw}))})
    for parent,prefix in [('Inspire-Stabilize-Sweep-v0','StabilizeSweepTest'),
                          ('Inspire-Sweep-Only-v0','SweepOnlyTest')]:
        child=entries[parent+'-Test']
        assert child['entry_point']==entries[parent]['entry_point']
        assert child['kwargs']['env_cfg_entry_point'].endswith(':'+prefix+'EnvCfg')
        assert child['kwargs']['rsl_rl_cfg_entry_point'].endswith(':'+prefix+'PPORunnerCfg')


def test_finger_reset_adapter_keeps_surface_snapshot_and_isolates_fallback(load_mdp,monkeypatch):
    import sys,torch
    from isaaclab.utils.math import quat_apply,quat_apply_inverse,quat_mul
    scope=_reset_layout_planner()
    scope.update(__package__='_baseline_cpu_pkg.mdp.event_functions',
                 quat_apply=quat_apply,quat_apply_inverse=quat_apply_inverse,quat_mul=quat_mul)
    calls=[];writes=[];gravity=[]
    def fast(env,ids,**params):
        calls.append((ids.tolist(),params))
        # Physical state is already committed by Finger; no physics step.
        env._active_object_mask[ids,0:2]=True
    monkeypatch.setitem(sys.modules,'_baseline_cpu_pkg.mdp.event_functions.wrench_refactor_relative_ik_test',
                        NS(fast_reset_scene=fast))
    scope['apply_fixed_pool_gravity_mask']=lambda env,ids:gravity.append(ids.tolist())
    states=torch.zeros(3,17,13);states[:,:,3]=1.;states[:,:,2]=1.05
    shelf=torch.zeros(3,13);shelf[:,3]=1.
    objects=NS(data=NS(object_link_state_w=states),
               write_object_link_state_to_sim=lambda value,env_ids:writes.append((env_ids.tolist(),value.clone())))
    env=NS(num_envs=3,device='cpu',scene={'object_collection':objects,'shelf':NS(data=NS(root_state_w=shelf))},
        _active_object_mask=torch.zeros(3,17,dtype=torch.bool),_revision_spawn_positions=torch.zeros(3,17,3),
        _surface_target_w=torch.full((3,3),7.),_surface_obstacles_w=torch.full((3,16,3),7.),
        _surface_valid=torch.ones(3,16,dtype=torch.bool),_final_actual_population=torch.zeros(3,dtype=torch.long))
    def snapshot(env,ids,local,yaw,active,shelf):
        active[:,1:]=False;local[:,1:]=0.;local[:,1:,2]=52.
        return torch.full((1,3),9.),torch.full((1,16,3),8.),torch.zeros(1,16,dtype=torch.bool),[
            dict(total_ms=1.,fallback=1,target_missing=1,slot_overflow=0)]
    scope['snapshot']=snapshot
    scope['reset_scene_finger_test'](env,torch.tensor([2]),ik_params={'height_offset':.01})
    assert len(calls)==1 and calls[0][0]==[2] and calls[0][1]['ik_params']=={'height_offset':.01}
    assert gravity==[[2]]
    assert len(writes)==1 and writes[0][0]==[2]
    assert writes[0][1][0,1,2]==52.
    assert env._surface_target_w[2].tolist()==[9.,9.,9.]
    assert env._surface_obstacles_w[2,0].tolist()==[8.,8.,8.]
    assert (env._surface_target_w[:2]==7.).all() and (env._surface_obstacles_w[:2]==7.).all()
    assert not env._active_object_mask[2,1:].any() and env._final_actual_population[2]==0
    assert env._reset_diagnostics['Reset/requested_obstacles']==1
    assert env._reset_diagnostics['Reset/actual_obstacles']==0
    assert env._reset_diagnostics['Reset/observation_fallbacks']==1
    assert env._reset_diagnostics['Reset/ik_attempts']==1
