"""Stabilize/Sweep public registration and sensor-to-task contracts."""
from types import SimpleNamespace as NS
import pytest
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


@pytest.mark.parametrize('model_name,direct_dim', [('StabilizeSweepModel',67),('SweepOnlyModel',66)])
def test_actor_critic_gradients_and_export(load_mdp,model_name,direct_dim):
    import importlib, torch
    from tensordict import TensorDict
    cls=getattr(importlib.import_module('_baseline_cpu_pkg.agent.stabilize_sweep_models'),model_name)
    obs=TensorDict({'policy':torch.randn(8,direct_dim),'obstacles':torch.randn(8,64)},batch_size=[8])
    obs['obstacles'][:,::4]=1
    groups={'actor':['policy','obstacles'],'critic':['policy','obstacles']}
    actor=cls(obs,groups,'actor',12,distribution_cfg={'class_name':'GaussianDistribution','init_std':1.})
    critic=cls(obs,groups,'critic',1)
    for model,width in ((actor,12),(critic,1)):
        result=model(obs)
        assert result.shape==(8,width) and model.get_latent(obs).shape==(8,direct_dim+16)
        result.square().mean().backward()
        assert model.obstacle_encoder.mlp[0].weight.grad.abs().sum()>0
    export=torch.jit.script(actor.as_jit())
    torch.testing.assert_close(export(torch.cat((obs['policy'],obs['obstacles']),-1)),actor(obs))
    assert actor.as_onnx().get_dummy_inputs()[0].shape==(1,direct_dim+64)


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
    import importlib,sys,torch
    from isaaclab.utils.math import quat_apply,quat_apply_inverse,quat_mul
    scope=_reset_layout_planner()
    scope.update(__package__='_baseline_cpu_pkg.mdp.event_functions',
                 quat_apply=quat_apply,quat_apply_inverse=quat_apply_inverse,quat_mul=quat_mul)
    calls=[];writes=[];gravity=[];cache_calls=[]
    parking=torch.zeros(3,17,3);parking[:,:,0]=3.;parking[:,:,2]=1.1
    layout_module=importlib.import_module('_baseline_cpu_pkg.mdp.common.stabilize_sweep_layout')
    monkeypatch.setattr(layout_module,'test_inactive_positions',lambda env:parking)
    def fast(env,ids,**params):
        assert env._fast_layout.sample_batch_size==512
        assert params['reuse_unchanged_states'] is True
        assert params['populate_legacy_observation'] is False
        torch.testing.assert_close(params['inactive_positions'],parking[ids])
        calls.append((ids.tolist(),params))
        # Physical state is already committed by Finger; no physics step.
        env._active_object_mask[ids,0:2]=True
    monkeypatch.setitem(sys.modules,'_baseline_cpu_pkg.mdp.event_functions.wrench_refactor_relative_ik_test',
                        NS(fast_reset_scene=fast))
    contact=importlib.import_module('_baseline_cpu_pkg.mdp.common.stabilize_sweep_contact')
    def cache(env):
        assert len(calls)==1
        cache_calls.append(env)
    monkeypatch.setattr(contact,'cache_static_local_com',cache)
    scope['apply_fixed_pool_gravity_mask']=lambda env,ids:gravity.append(ids.tolist())
    states=torch.zeros(3,17,13);states[:,:,3]=1.;states[:,:,2]=1.05
    shelf=torch.zeros(3,13);shelf[:,3]=1.
    objects=NS(data=NS(object_link_state_w=states),
               write_object_link_state_to_sim=lambda value,env_ids:writes.append((env_ids.tolist(),value.clone())))
    env=NS(num_envs=3,device='cpu',scene={'object_collection':objects,'shelf':NS(data=NS(root_state_w=shelf))},
        _fast_layout=NS(bounds=torch.empty(3,17,2,3)),
        _active_object_mask=torch.zeros(3,17,dtype=torch.bool),_revision_spawn_positions=torch.zeros(3,17,3),
        _surface_target_w=torch.full((3,3),7.),_surface_obstacles_w=torch.full((3,16,3),7.),
        _surface_valid=torch.ones(3,16,dtype=torch.bool),_final_actual_population=torch.zeros(3,dtype=torch.long))
    def snapshot(env,ids,local,yaw,active,shelf,*,batched=False):
        assert batched
        assert cache_calls==[env]
        active[:,1:]=False;local[:,1:]=0.;local[:,1:,2]=52.
        return torch.full((1,3),9.),torch.full((1,16,3),8.),torch.zeros(1,16,dtype=torch.bool),[
            dict(total_ms=1.,fallback=1,target_missing=1,slot_overflow=0)]
    scope['snapshot']=snapshot
    scope['reset_scene_finger_test'](env,torch.tensor([2]),ik_params={'height_offset':.01})
    assert len(calls)==1 and calls[0][0]==[2] and calls[0][1]['ik_params']=={'height_offset':.01}
    assert cache_calls==[env]
    assert gravity==[[2]]
    assert len(writes)==1 and writes[0][0]==[2]
    torch.testing.assert_close(writes[0][1][0,1:,:3],parking[2,1:])
    assert env._surface_target_w[2].tolist()==[9.,9.,9.]
    assert env._surface_obstacles_w[2,0].tolist()==[8.,8.,8.]
    assert (env._surface_target_w[:2]==7.).all() and (env._surface_obstacles_w[:2]==7.).all()
    assert not env._active_object_mask[2,1:].any() and env._final_actual_population[2]==0
    assert env._reset_diagnostics['Reset/requested_obstacles']==1
    assert env._reset_diagnostics['Reset/actual_obstacles']==0
    assert env._reset_diagnostics['Reset/observation_fallbacks']==1
    assert env._reset_diagnostics['Reset/ik_attempts']==1


def test_test_parking_geometry_cache_and_shelf_frame(load_mdp):
    import importlib,math,torch
    from isaaclab.utils.math import quat_apply
    parking=importlib.import_module('_baseline_cpu_pkg.mdp.common.stabilize_sweep_layout').test_inactive_positions
    bounds=torch.tensor([[-.11,-.09,-.10],[.11,.09,.10]]).expand(3,17,2,3).clone()
    shelf=torch.zeros(3,13);shelf[:,0]=-.7;shelf[:,3]=1.
    shelf[1,3]=shelf[1,6]=math.sqrt(.5)
    root=torch.zeros(3,13);root[:,3]=1.
    spawn_z=1.05+torch.arange(51).reshape(3,17)*.001
    env=NS(_revision_bounds_t=bounds,_refactor_spawn_z=spawn_z,cfg=NS(scene=NS(filter_collisions=True)),
        scene={'shelf':NS(data=NS(default_root_state=shelf)),
               'robot':NS(data=NS(default_root_state=root))})
    before=torch.random.get_rng_state().clone()
    local=parking(env)
    assert local.shape==(3,17,3) and parking(env) is local
    assert torch.equal(torch.random.get_rng_state(),before)
    world=shelf[:,None,:3]+quat_apply(shelf[:,None,3:7].expand(-1,17,-1),local)
    slot=torch.arange(16)
    expected_xy=torch.stack((3.+.45*(slot//4),.45*(slot%4-1.5)),-1)
    torch.testing.assert_close(world[:,1:,:2],expected_xy[None].expand(3,-1,-1),atol=1e-6,rtol=0.)
    torch.testing.assert_close(local[...,2],spawn_z,atol=0,rtol=0.)
    torch.testing.assert_close(world[:,1:,2],spawn_z[:,1:],atol=1e-6,rtol=0.)
    distance=torch.cdist(world[0,1:],world[0,1:])
    distance.fill_diagonal_(float('inf'))
    radius=bounds.abs().amax(-2).norm(dim=-1).max()
    assert distance.min()>2*radius+.002
    assert world[:,1:,0].min()-radius>1.479+.1
    del env._test_inactive_positions
    env.cfg.scene.filter_collisions=False
    with pytest.raises(ValueError,match='collision filtering'): parking(env)
    env.cfg.scene.filter_collisions=True
    env._revision_bounds_t=bounds*2
    with pytest.raises(ValueError,match='radius'): parking(env)


def test_test_layout_batch_preserves_sampler_outputs_rng_and_partial_rows(load_mdp):
    import importlib,torch
    layout_cls=importlib.import_module('_baseline_cpu_pkg.mdp.common.wrench_refactor_layout').CircleResetLayout
    configure=importlib.import_module('_baseline_cpu_pkg.mdp.common.stabilize_sweep_layout').configure_test_layout_batch
    geometry_rng=torch.Generator().manual_seed(301)
    half=.012+torch.rand((280,17,3),generator=geometry_rng)*.013
    offset=(torch.rand((280,17,3),generator=geometry_rng)-.5)*.004
    bounds=torch.stack((offset-half,offset+half),-2)
    spawn=1.05-bounds[:,:,0,2]
    reference=layout_cls(bounds,spawn,cell_size=.02)
    optimized=layout_cls(bounds,spawn,cell_size=.02)
    assert reference.sample_batch_size==optimized.sample_batch_size==64
    ids=torch.randperm(280,generator=geometry_rng)[:270]
    for seed in (3,29):
        expected_rng=torch.Generator().manual_seed(seed)
        expected=reference.sample(ids,expected_rng)
        assert (expected[-1]==0).any() and (expected[-1]==6).any()
        for size in (256,512,64):
            configure(NS(_fast_layout=optimized),batch_size=size)
            actual_rng=torch.Generator().manual_seed(seed)
            actual=optimized.sample(ids,actual_rng)
            for value,baseline in zip(actual,expected):
                torch.testing.assert_close(value,baseline,atol=0,rtol=0)
            torch.testing.assert_close(actual_rng.get_state(),expected_rng.get_state(),atol=0,rtol=0)
            assert len(optimized._scratch[0])==size
    # A sparse partial reset after a larger scratch allocation follows the
    # exact same stream; other rows never participate in the output batch.
    partial=torch.tensor([13,1,200,3])
    configure(NS(_fast_layout=optimized))
    left=torch.Generator().manual_seed(51);right=torch.Generator().manual_seed(51)
    for value,baseline in zip(optimized.sample(partial,left),reference.sample(partial,right)):
        torch.testing.assert_close(value,baseline,atol=0,rtol=0)
    torch.testing.assert_close(left.get_state(),right.get_state(),atol=0,rtol=0)
    assert reference.sample_batch_size==64


def _batched_snapshot_scope(monkeypatch):
    """Run the production adapter with only the geometry query substituted."""
    import sys
    from isaaclab.utils.math import quat_apply
    scope=_reset_layout_planner()
    scope.update(__package__='_baseline_cpu_pkg.mdp.event_functions',quat_apply=quat_apply)
    # No USD stage or simulator is needed when the environment owns its cache.
    def unexpected_constructor(env):
        raise AssertionError('The cached observer must be reused')
    monkeypatch.setitem(sys.modules,'_baseline_cpu_pkg.mdp.common.stabilize_sweep_surface',
                        NS(BatchedSurfaceGeometry=unexpected_constructor))
    return scope


def test_batched_snapshot_retries_missing_and_overflow_rows_together(load_mdp,monkeypatch):
    import numpy as np
    import torch
    scope=_batched_snapshot_scope(monkeypatch)
    ids=torch.tensor([8,2,11,5])
    local=torch.arange(4*17*3,dtype=torch.float32).reshape(4,17,3)/100
    yaw=torch.arange(4*17,dtype=torch.float32).reshape(4,17)/10
    active=torch.ones(4,17,dtype=torch.bool)
    before_local,before_yaw,before_active=local.clone(),yaw.clone(),active.clone()
    calls=[]
    targets=np.array([[.1,.2,.3],[.4,.5,.6],[.7,.8,.9],[1.,1.1,1.2]],dtype=np.float32)
    healthy_others=[np.array([[.3,-.2,.1]],dtype=np.float32),
                    np.arange(48,dtype=np.float32).reshape(16,3)/100]
    timing_keys=('total_ms','build_ms','upload_ms','readback_ms','ray_ms','components_ms','rays')
    def observe_batch(indices,positions,angles,mask):
        assert all(isinstance(value,np.ndarray) for value in (indices,positions,angles,mask))
        calls.append(tuple(value.copy() for value in (indices,positions,angles,mask)))
        if len(calls)==1:
            assert indices.tolist()==[8,2,11,5]
            rows=[(targets[0],healthy_others[0]),(None,np.empty((0,3))),
                  (targets[2],healthy_others[1]),(targets[3],np.ones((17,3)))]
            return [(point,others,dict.fromkeys(timing_keys,1.)) for point,others in rows]
        assert len(calls)==2
        assert indices.tolist()==[2,5]
        assert mask[:,0].all() and not mask[:,1:].any()
        np.testing.assert_array_equal(positions[:,0],before_local[[1,3],0].numpy())
        np.testing.assert_array_equal(angles[:,0],before_yaw[[1,3],0].numpy())
        np.testing.assert_array_equal(positions[:,1:,:2],np.zeros((2,16,2)))
        np.testing.assert_array_equal(positions[:,1:,2],np.broadcast_to(np.arange(52.,84.,2.),(2,16)))
        np.testing.assert_array_equal(angles[:,1:],np.zeros((2,16)))
        return [(targets[row],np.empty((0,3),dtype=np.float32),dict.fromkeys(timing_keys,2.))
                for row in (1,3)]
    env=NS(device='cpu',_surface_batch_geometry=NS(observe_batch=observe_batch))
    shelf=torch.zeros(4,13)
    shelf[:,:3]=torch.tensor([[10.,20.,30.],[40.,50.,60.],[70.,80.,90.],[100.,110.,120.]])
    shelf[:,3]=shelf[:,6]=2**-.5  # Rotate local X to world Y by 90 degrees.
    target_w,obstacle_w,valid,timings=scope['_snapshot_batched'](env,ids,local,yaw,active,shelf)
    assert len(calls)==2
    torch.testing.assert_close(local[[0,2]],before_local[[0,2]])
    torch.testing.assert_close(yaw[[0,2]],before_yaw[[0,2]])
    torch.testing.assert_close(active[[0,2]],before_active[[0,2]])
    for actual,expected in zip((local[[1,3]],yaw[[1,3]],active[[1,3]]),calls[1][1:]):
        torch.testing.assert_close(actual,torch.as_tensor(expected))
    target_local=torch.from_numpy(targets)
    expected_target=torch.stack((-target_local[:,1],target_local[:,0],target_local[:,2]),-1)+shelf[:,:3]
    torch.testing.assert_close(target_w,expected_target)
    assert valid.sum(-1).tolist()==[1,0,16,0]
    for row,points in ((0,healthy_others[0]),(2,healthy_others[1])):
        points=torch.from_numpy(points)
        expected=torch.stack((-points[:,1],points[:,0],points[:,2]),-1)+shelf[row,:3]
        torch.testing.assert_close(obstacle_w[row,:len(points)],expected)
    torch.testing.assert_close(obstacle_w[[1,3]],shelf[[1,3],None,:3].expand(-1,16,-1))
    assert [item['fallback'] for item in timings]==[0,1,0,1]
    assert [item['target_missing'] for item in timings]==[0,1,0,0]
    assert [item['slot_overflow'] for item in timings]==[0,0,0,1]
    for row,timing in enumerate(timings):
        assert all(timing[key]==(3. if row in (1,3) else 1.) for key in timing_keys)


def test_batched_snapshot_success_uses_one_query_and_keeps_physical_state(load_mdp,monkeypatch):
    import numpy as np
    import torch
    scope=_batched_snapshot_scope(monkeypatch)
    ids=torch.tensor([9,3])
    local=torch.randn(2,17,3);yaw=torch.randn(2,17)
    active=torch.ones(2,17,dtype=torch.bool)
    before=[value.clone() for value in (local,yaw,active)]
    calls=[]
    def observe_batch(indices,positions,angles,mask):
        calls.append(indices.tolist())
        return [(np.array([1.,2.,3.]),np.zeros((16,3)),{'total_ms':2.}),
                (np.array([4.,5.,6.]),np.empty((0,3)),{'total_ms':3.})]
    env=NS(device='cpu',_surface_batch_geometry=NS(observe_batch=observe_batch))
    shelf=torch.zeros(2,13);shelf[:,3]=1.
    targets,_,valid,timings=scope['_snapshot_batched'](env,ids,local,yaw,active,shelf)
    assert calls==[[9,3]]
    torch.testing.assert_close(targets,torch.tensor([[1.,2.,3.],[4.,5.,6.]]))
    assert valid.sum(-1).tolist()==[16,0]
    assert not any(item['fallback'] for item in timings)
    for actual,expected in zip((local,yaw,active),before):
        torch.testing.assert_close(actual,expected)


@pytest.mark.parametrize('initial_overflow',[False,True])
def test_batched_snapshot_target_only_failure_is_bounded(load_mdp,monkeypatch,initial_overflow):
    import numpy as np
    import torch
    scope=_batched_snapshot_scope(monkeypatch)
    calls=[]
    def observe_batch(indices,positions,angles,mask):
        calls.append(indices.tolist())
        if len(calls)==1 and initial_overflow:
            return [(np.zeros(3),np.zeros((17,3)),{'total_ms':1.})]
        return [(None,np.empty((0,3)),{'total_ms':1.})]
    env=NS(device='cpu',_surface_batch_geometry=NS(observe_batch=observe_batch))
    shelf=torch.zeros(1,13);shelf[:,3]=1.
    with pytest.raises(RuntimeError,match='Target-only visible snapshot missing after bounded fallback: env=42'):
        scope['_snapshot_batched'](env,torch.tensor([42]),torch.zeros(1,17,3),
            torch.zeros(1,17),torch.ones(1,17,dtype=torch.bool),shelf)
    assert calls==[[42],[42]]


def test_fused_red_distance_matches_torch_cells_boundaries_and_degenerate_polygons(load_mdp):
    import importlib,torch
    module=importlib.import_module('_baseline_cpu_pkg.mdp.common.stabilize_sweep_layout')
    layout_cls=importlib.import_module('_baseline_cpu_pkg.mdp.common.wrench_refactor_layout').CircleResetLayout
    half=torch.full((40,17,3),.02)
    bounds=torch.stack((-half,half),-2)
    layout=layout_cls(bounds,1.05-bounds[:,:,0,2])
    rng=torch.Generator().manual_seed(63)
    start,goal=layout.sample_commands(torch.arange(32),rng)
    # Include zero-area points/segments, zero extension, a very small area and
    # negative zero. Boundary points exercise projection clamps at both ends.
    start=torch.cat((start,torch.tensor([[0.,0.],[0.,0.],[0.,0.],[.2,0.],[.3,.1],[0.,0.],[-0.,-0.]])))
    goal=torch.cat((goal,torch.tensor([[0.,0.],[.1,0.],[0.,.1],[.3,.1],[.4,.2],[.1,1e-12],[-0.,-0.]])))
    points=torch.cat((layout.cells,start,goal))
    # Interop must respect Torch strides instead of assuming contiguous inputs.
    strided=torch.empty((len(start),4));strided[:,::2]=start
    start=strided[:,::2]
    expected=layout_cls.red_distance(points,start,goal)
    actual=module._launch_red_distance(points,start,goal)
    assert torch.equal(actual.view(torch.uint8),expected.view(torch.uint8))
    threshold=layout.expanded[:len(start),0,None]+layout.expanded[:len(start),1,None]+layout.cell_margin
    assert torch.equal(actual>=threshold,expected>=threshold)
    assert module._launch_red_distance(points,start[:0],goal[:0]).shape==(0,len(points))
    assert module._launch_red_distance(points[:0],start,goal).shape==(len(start),0)


def test_fused_red_distance_cpu_fallback_leaves_parent_and_instance_unchanged(load_mdp,monkeypatch):
    import importlib,torch
    module=importlib.import_module('_baseline_cpu_pkg.mdp.common.stabilize_sweep_layout')
    layout_cls=importlib.import_module('_baseline_cpu_pkg.mdp.common.wrench_refactor_layout').CircleResetLayout
    original=layout_cls.red_distance
    half=torch.full((2,17,3),.02)
    bounds=torch.stack((-half,half),-2)
    layout=layout_cls(bounds,1.05-bounds[:,:,0,2])
    module.configure_test_layout_batch(NS(_fast_layout=layout))
    assert 'red_distance' not in layout.__dict__
    assert layout_cls.red_distance is original
    def forbidden(*args):
        raise AssertionError('CPU inputs must retain the original Torch path')
    monkeypatch.setattr(module,'_launch_red_distance',forbidden)
    for dtype in (torch.float32,torch.float64):
        points=torch.tensor([[.1,0.],[-.1,0.],[.1,.1]],dtype=dtype)
        start=torch.zeros(2,2,dtype=dtype)
        goal=torch.tensor([[.1,0.],[0.,.1]],dtype=dtype)
        actual=module.fused_test_red_distance(points,start,goal)
        expected=original(points,start,goal)
        assert torch.equal(actual.view(torch.uint8),expected.view(torch.uint8))


def test_fused_red_distance_preserves_complete_sample_and_rng(load_mdp):
    import importlib,torch
    module=importlib.import_module('_baseline_cpu_pkg.mdp.common.stabilize_sweep_layout')
    layout_cls=importlib.import_module('_baseline_cpu_pkg.mdp.common.wrench_refactor_layout').CircleResetLayout
    geometry_rng=torch.Generator().manual_seed(88)
    half=.012+torch.rand((48,17,3),generator=geometry_rng)*.013
    offset=(torch.rand((48,17,3),generator=geometry_rng)-.5)*.004
    bounds=torch.stack((offset-half,offset+half),-2)
    spawn=1.05-bounds[:,:,0,2]
    reference=layout_cls(bounds,spawn,cell_size=.01)
    optimized=layout_cls(bounds,spawn,cell_size=.01)
    # Execute the production kernel on CPU while production CPU configuration
    # itself continues to use Torch. No simulator or CUDA is needed here.
    optimized.red_distance=module._launch_red_distance
    optimized.sample_batch_size=16
    for ids in (torch.randperm(48,generator=geometry_rng),torch.tensor([35,0,16,3])):
        for seed in (3,29):
            left=torch.Generator().manual_seed(seed);right=torch.Generator().manual_seed(seed)
            expected=reference.sample(ids,left)
            actual=optimized.sample(ids,right)
            for value,baseline in zip(actual,expected):
                assert torch.equal(value.contiguous().view(torch.uint8),baseline.contiguous().view(torch.uint8))
            assert torch.equal(left.get_state(),right.get_state())
    assert 'red_distance' not in reference.__dict__


def _legacy_observation_reset_env(configs):
    import importlib, torch
    from test_wrench_refactor import pool_bounds
    from types import SimpleNamespace as NS
    events = configs('event_functions', 'wrench_refactor_relative_ik_test')
    m = importlib.import_module('_baseline_cpu_pkg.mdp.common.wrench_refactor_layout')
    n = 8
    bounds = pool_bounds(n)
    origin = torch.zeros(n, 3); origin[:, 1] = torch.arange(n) * 2.5
    root = torch.zeros(n, 13); root[:, 3] = 1.
    shelf_root = root.clone(); shelf_root[:, 0] = -.7
    class Robot:
        def __init__(self):
            self.data = NS(default_root_state=root, default_joint_pos=torch.ones(n, 18)*.2,
                default_joint_vel=torch.zeros(n, 18), body_link_pos_w=torch.zeros(n, 1, 3))
            self.joints = torch.ones(n, 18)*9
            self.roots = torch.ones(n, 13)*9
            self.writes = 0
        def write_root_state_to_sim(self, value, env_ids): self.roots[env_ids] = value
        def write_joint_state_to_sim(self, q, v, env_ids):
            self.writes += 1
            self.joints[env_ids] = q
            assert not v.any()
        def set_joint_position_target(self, *args, **kwargs): pass
        def set_joint_velocity_target(self, *args, **kwargs): pass
    class Shelf:
        data = NS(default_root_state=shelf_root)
        def write_root_state_to_sim(self, values, env_ids): self.values = values.clone()
    class Objects:
        def __init__(self):
            self.state = torch.ones(n, 17, 13)*9
            self.root_physx_view = NS(set_disable_gravities=lambda *args:None, wake_up=lambda *args:None)
        def write_object_link_state_to_sim(self, value, env_ids): self.state[env_ids] = value
    class Scene(dict): pass
    scene = Scene(robot=Robot(), shelf=Shelf(), object_collection=Objects(), ee_frame=NS(reset=lambda ids:None))
    scene.env_origins = origin
    angles = torch.arange(6).float()[None]
    solver = NS(solve=lambda p,q,initial,limits:(angles.expand(len(p),-1),torch.ones(len(p),dtype=torch.bool),torch.zeros(len(p)),torch.zeros(len(p))))
    cfg = NS(placement_clearance=.01, layout_cell_size=.005, layout_geometry_epsilon=1e-6, debug_initial_ik_bounds=False)
    env = NS(cfg=cfg, device='cpu', num_envs=n, scene=scene, _revision_bounds_t=bounds,
        _fast_layout=m.CircleResetLayout(bounds, 1.05-bounds[...,0,2]),
        _fast_visibility_radii=bounds[...,:2].abs().amax(-2).norm(dim=-1),
        _fast_geometry_key=(.01,.005,1e-6,bounds._version),
        _fast_ik=solver, _fast_base_offset=(torch.zeros(n,3),root[:,3:7]),
        _fast_initial=torch.zeros(n,6), _fast_limits=torch.tensor([-6.,6.]).expand(n,6,2),
        _fast_default_eef=(torch.tensor([[-.4,0.,1.2]]).expand(n,-1),root[:,3:7]),
        _goal_reset_arm_ids=list(range(6)), _goal_reset_rng=torch.Generator().manual_seed(1749),
        _revision_layout_rng=torch.Generator().manual_seed(42),
        _wrench_gravity_flags=torch.ones(n*17,1,dtype=torch.uint8),
        _active_object_mask=torch.zeros(n,17,dtype=torch.bool),target_id=torch.ones(n,1,dtype=torch.long),
        target_width=torch.zeros(n,1),sweep_dir=torch.zeros(n,3),_final_actual_population=torch.zeros(n,dtype=torch.long),
        _revision_layout_complete=torch.zeros(n,dtype=torch.bool),_revision_spawn_positions=torch.zeros(n,17,3),
        sensor_data_fresh=torch.ones(n,dtype=torch.bool),_initial_obstacle_observation=torch.zeros(n,64),
        _goal_reset_desired_pos_w=torch.zeros(n,3),_goal_reset_desired_quat_w=torch.zeros(n,4),_goal_reset_errors=torch.zeros(n,2))
    env._initial_obstacle_observation.fill_(7.)
    return env, events

def _legacy_observation_reset_snapshot(env):
    import torch
    values={k:v.clone() for k,v in vars(env).items() if isinstance(v,torch.Tensor)}
    values.update(robot_root=env.scene['robot'].roots.clone(),
                  robot_joint=env.scene['robot'].joints.clone(),
                  object_state=env.scene['object_collection'].state.clone(),
                  layout_rng=env._revision_layout_rng.get_state(),
                  ik_rng=env._goal_reset_rng.get_state(),global_rng=torch.get_rng_state())
    if hasattr(env.scene['shelf'],'values'):
        values['shelf_values']=env.scene['shelf'].values.clone()
    return values


@pytest.mark.parametrize('ids',[[1,4,7],[7,0],[0],[]])
@pytest.mark.parametrize('swap',[0.,.1,1.])
def test_legacy_observation_skip_preserves_full_reset_state_and_rng(configs,ids,swap):
    import torch
    kwargs=dict(ik_params=dict(height_offset=.01,front_clearance=(.08,.10),lateral_offset=(.10,.18),height_jitter=.02),observation_swap_probability=swap)
    reference,events=_legacy_observation_reset_env(configs)
    selected=torch.tensor(ids,dtype=torch.long)
    torch.manual_seed(517)
    events.fast_reset_scene(reference,selected,**kwargs)
    expected=_legacy_observation_reset_snapshot(reference)
    for populate in (True,False):
        actual,events=_legacy_observation_reset_env(configs)
        torch.manual_seed(517)
        events.fast_reset_scene(actual,selected,**kwargs,populate_legacy_observation=populate)
        got=_legacy_observation_reset_snapshot(actual)
        assert expected.keys()==got.keys()
        for key,value in expected.items():
            if key=='_initial_obstacle_observation' and not populate:
                assert not got[key][selected].any()
                keep=torch.ones(actual.num_envs,dtype=torch.bool);keep[selected]=False
                assert (got[key][keep]==7.).all()
            else:
                torch.testing.assert_close(got[key],value,atol=0,rtol=0,msg=key)


@pytest.mark.parametrize('bad',[{'visibility_threshold':0.},{'visibility_threshold':float('nan')},
                              {'observation_swap_probability':-1.},{'observation_swap_probability':float('nan')}])
def test_legacy_observation_skip_preserves_invalid_parameter_errors_and_rng(configs,bad):
    import torch
    kwargs=dict(ik_params=dict(height_offset=.01,front_clearance=(.08,.10),lateral_offset=(.10,.18),height_jitter=.02),**bad)
    values=[]
    for prototype in (False,True):
        env,events=_legacy_observation_reset_env(configs)
        torch.manual_seed(517)
        function=events.fast_reset_scene
        extra={'populate_legacy_observation':False} if prototype else {}
        with pytest.raises(ValueError) as error:
            function(env,torch.tensor([7,0]),**kwargs,**extra)
        values.append((str(error.value),_legacy_observation_reset_snapshot(env)))
    assert values[0][0]==values[1][0]
    for key,value in values[0][1].items():
        torch.testing.assert_close(values[1][1][key],value,atol=0,rtol=0,msg=key)


def test_test_observations_do_not_read_legacy_centroid_buffer(load_mdp):
    import ast, importlib, torch
    functions=importlib.import_module('_baseline_cpu_pkg.mdp.observation_functions.stabilize_sweep')
    target=functions.target_surface_in_eef
    obstacles=functions.obstacle_surfaces_in_eef
    config=ast.parse((MODULE_ROOT/'mdp/observations/stabilize_sweep.py').read_text())
    configured={node.targets[0].id:node.value for node in ast.walk(config)
                if isinstance(node,ast.Assign) and isinstance(node.targets[0],ast.Name)}
    for name,function in (('target_obs_state',target),('initial_positions',obstacles)):
        actual=next(arg.value for arg in configured[name].keywords if arg.arg=='func')
        assert isinstance(actual,ast.Name) and actual.id==function.__name__
    frame=NS(target_pos_w=torch.zeros(2,1,3),
             target_quat_w=torch.tensor([1.,0.,0.,0.]).expand(2,1,4))
    env=NS(scene={'ee_frame':NS(data=frame)},
           _surface_target_w=torch.arange(6,dtype=torch.float32).reshape(2,3),
           _surface_obstacles_w=torch.arange(96,dtype=torch.float32).reshape(2,16,3),
           _surface_valid=torch.ones(2,16,dtype=torch.bool),
           _initial_obstacle_observation=torch.full((2,64),float('nan')))
    expected=(target(env),obstacles(env))
    env._initial_obstacle_observation.zero_()
    for actual,reference in zip((target(env),obstacles(env)),expected):
        torch.testing.assert_close(actual,reference,atol=0,rtol=0)


def _reference_circle_cell_selection(points, boundary, red, target_radius, radii, requested, draws, cell_margin, cell_size):
    """Independent original eager-Torch selection, including partial failures."""
    import torch
    n = len(radii)
    complete = torch.ones(n,dtype=torch.bool,device=points.device)
    failed = torch.full((n,),-1,dtype=torch.long,device=points.device)
    indices,enableds,distances = [],[],[]
    for slot in range(6):
        radius = radii[:,slot]
        needed = requested > slot
        valid = boundary[None] >= radius[:,None]+cell_margin
        valid &= red >= target_radius[:,None]+radius[:,None]+cell_margin
        for previous,distance in enumerate(distances):
            valid &= distance >= (radius+radii[:,previous]+cell_margin)[:,None].square()
        prefix = valid.cumsum(-1,dtype=torch.int32)
        totals = prefix[:,-1]
        ok = (totals>0) | ~needed
        failed = torch.where(complete & ~ok,slot,failed)
        complete &= ok
        rank = (draws[:,slot,0]*totals).long()
        index = torch.searchsorted(prefix,rank.to(torch.int32)[:,None],right=True).squeeze(-1).clamp_max(len(points)-1)
        centers = points[index]+(draws[:,slot,1:3]-.5)*cell_size
        enabled = needed & complete
        distance = torch.sum((points[None]-centers[:,None]).square(),dim=-1)
        distances.append(distance.masked_fill(~enabled[:,None],float('inf')))
        indices.append(index);enableds.append(enabled)
    return NS(indices=torch.stack(indices,-1),enabled=torch.stack(enableds,-1),
                           complete=complete,failed_slot=failed)


def test_cooperative_circle_selection_matches_raw_indices_and_failure_boundaries(load_mdp):
    import importlib,torch
    module=importlib.import_module('_baseline_cpu_pkg.mdp.common.stabilize_sweep_layout')
    layout_cls=importlib.import_module('_baseline_cpu_pkg.mdp.common.wrench_refactor_layout').CircleResetLayout
    half=torch.full((14,17,3),.02)
    bounds=torch.stack((-half,half),-2)
    layout=layout_cls(bounds,1.05-bounds[:,:,0,2])
    for seed in (7,31):
        rng=torch.Generator().manual_seed(seed)
        start,goal=layout.sample_commands(torch.arange(14),rng)
        red=layout.red_distance(layout.cells,start,goal)
        radii=.022+torch.rand((14,6),generator=rng)*.025
        radii[0,0]=.7;radii[1,2]=.7
        requested=torch.arange(14)%7;requested[0]=6;requested[1]=6
        draws=torch.rand((14,6,4),generator=rng)
        draws[2,:,0]=0.
        draws[3,:,0]=torch.nextafter(torch.tensor(1.),torch.tensor(0.))
        inputs=(layout.cells,layout.boundary,red,layout.expanded[:,0],radii,requested,draws,layout.cell_margin,layout.cell_size)
        expected=_reference_circle_cell_selection(*inputs)
        actual=module._select_test_cells(*inputs)
        for name in ('indices','enabled','complete','failed_slot'):
            assert torch.equal(getattr(actual,name),getattr(expected,name))
        assert actual.failed_slot[0]==0 and actual.failed_slot[1]==2


def test_cooperative_circle_sampler_preserves_outputs_rng_and_failure_payload(load_mdp):
    import importlib,torch
    module=importlib.import_module('_baseline_cpu_pkg.mdp.common.stabilize_sweep_layout')
    layout_cls=importlib.import_module('_baseline_cpu_pkg.mdp.common.wrench_refactor_layout').CircleResetLayout
    rng=torch.Generator().manual_seed(88)
    half=.012+torch.rand((48,17,3),generator=rng)*.013
    offset=(torch.rand((48,17,3),generator=rng)-.5)*.004
    bounds=torch.stack((offset-half,offset+half),-2)
    spawn=1.05-bounds[:,:,0,2]
    reference=layout_cls(bounds,spawn,cell_size=.01)
    optimized=layout_cls(bounds,spawn,cell_size=.01)
    optimized.cell_selector=module._select_test_cells
    optimized.sample_batch_size=16
    for ids in (torch.randperm(48,generator=rng),torch.tensor([35,0,16,3])):
        for seed in (3,29):
            left=torch.Generator().manual_seed(seed);right=torch.Generator().manual_seed(seed)
            expected=reference.sample(ids,left)
            actual=optimized.sample(ids,right)
            for value,baseline in zip(actual,expected):
                assert torch.equal(value.contiguous().view(torch.uint8),baseline.contiguous().view(torch.uint8))
            assert torch.equal(left.get_state(),right.get_state())
    assert optimized._scratch is None
    assert reference._scratch is not None and 'cell_selector' not in reference.__dict__
    impossible=bounds[:4].clone();impossible[:,1:,0,:2]=-.3;impossible[:,1:,1,:2]=.3
    failures=[]
    for use_selector in (False,True):
        layout=layout_cls(impossible,spawn[:4])
        if use_selector:layout.cell_selector=module._select_test_cells
        try:
            layout.sample(torch.arange(4),torch.Generator().manual_seed(42))
        except RuntimeError as error:
            failures.append(str(error))
    assert len(failures)==2 and failures[0]==failures[1]


def test_cooperative_circle_configuration_is_cuda_float32_test_only(load_mdp):
    import importlib,torch
    module=importlib.import_module('_baseline_cpu_pkg.mdp.common.stabilize_sweep_layout')
    for cuda,dtype,count,enabled in ((False,torch.float32,16000,False),
                                     (True,torch.float64,16000,False),
                                     (True,torch.float32,16001,False),
                                     (True,torch.float32,0,False),
                                     (True,torch.float32,16000,True)):
        layout=NS(bounds=NS(is_cuda=cuda,dtype=dtype),cells=range(count))
        module.configure_test_layout_batch(NS(_fast_layout=layout))
        assert hasattr(layout,'cell_selector') is enabled
        if enabled:assert layout.cell_selector is module._select_test_cells
    # Kernel shared-memory bounds are checked even for direct internal use.
    import pytest
    with pytest.raises(ValueError,match='16000'):
        module._select_test_cells(torch.empty(16001,2),None,None,None,None,None,None,.01,.005)


def test_ik_obstacle_batch_preserves_robot_clearance_layout_and_rng(load_mdp):
    import importlib,torch
    cls=importlib.import_module('_baseline_cpu_pkg.mdp.common.wrench_refactor_layout').CircleResetLayout
    module=importlib.import_module('_baseline_cpu_pkg.mdp.common.stabilize_sweep_layout')
    n=35
    rng=torch.Generator().manual_seed(113)
    half=.015+.025*torch.rand(n,17,3,generator=rng)
    bounds=torch.stack((-half,half),-2)
    layout=cls(bounds,1.05+half[...,2],cell_size=.02)
    ids=torch.randperm(n,generator=rng)
    selected=torch.rand(n,16,generator=rng).argsort(-1)[:,:6]+1
    requested=torch.arange(n)%7
    delta=torch.zeros(n,2);delta[:,1]=torch.where(torch.arange(n)%2==0,.08,-.09)
    centers=torch.zeros(n,2,3);centers[:,0]=torch.tensor([.08,.08,1.08]);centers[:,1]=torch.tensor([-.06,-.06,1.08])
    # Alternating high/low robot parts exercise height-overlap filtering.
    centers[::2,1,2]=2.
    robot=(centers,torch.full_like(centers,.025),torch.eye(3).expand(n,2,3,3))
    results=[]
    for size in (8,1,32,64):
        local=torch.zeros(n,17,3);local[...,2]=50.;local[:,0,2]=layout.spawn_z[ids,0]
        yaw=torch.zeros(n,17);active=torch.zeros(n,17,dtype=torch.bool);active[:,0]=True
        generator=torch.Generator().manual_seed(912)
        module.place_obstacles(layout,ids,local,yaw,active,delta,robot,selected,requested,generator,batch_size=size)
        results.append((local,yaw,active,generator.get_state()))
    for result in results[1:]:
        for actual,expected in zip(result,results[0]):
            torch.testing.assert_close(actual,expected,atol=0,rtol=0)
    assert results[0][2][:,1:].any()


def test_cooperative_robot_layout_matches_reference_partial_rows_failures_and_rng(load_mdp):
    import importlib,torch
    cls=importlib.import_module('_baseline_cpu_pkg.mdp.common.wrench_refactor_layout').CircleResetLayout
    module=importlib.import_module('_baseline_cpu_pkg.mdp.common.stabilize_sweep_layout')
    n=13
    rng=torch.Generator().manual_seed(611)
    half=.012+.035*torch.rand(n,17,3,generator=rng)
    bounds=torch.stack((-half,half),-2)
    offsets=(torch.rand(n,17,3,generator=rng)-.5)*.015;offsets[...,2]=0.
    bounds+=offsets[:,:,None]
    # A blocked large candidate must not disable later smaller candidates.
    bounds[7,1,0,:2]=-.3;bounds[7,1,1,:2]=.3
    layout=cls(bounds,1.05+half[...,2],cell_size=.02)
    ids=torch.tensor([7,3,11,0,8,2,12])
    selected=torch.rand(len(ids),16,generator=rng).argsort(-1)[:,:6]+1
    selected[0]=torch.arange(1,7)
    requested=torch.tensor([6,0,1,2,3,4,5])
    delta=torch.zeros(len(ids),2);delta[:,1]=torch.where(torch.arange(len(ids))%2==0,.08,-.09)
    centers=torch.rand(len(ids),5,3,generator=rng)*.2-.1
    centers[:,:,2]=torch.tensor([1.08,1.15,1.9,1.04,1.12])
    robot=(centers,torch.full_like(centers,.025),torch.eye(3).expand(len(ids),5,3,3))
    initial=torch.zeros(len(ids),17,3);initial[...,2]=50.;initial[:,0,2]=layout.spawn_z[ids,0]
    results=[]
    for accelerated in (False,True):
        local=initial.clone();yaw=torch.zeros(len(ids),17)
        active=torch.zeros(len(ids),17,dtype=torch.bool);active[:,0]=True
        generator=torch.Generator().manual_seed(912)
        if accelerated:
            radii=layout.expanded[ids[:,None],selected]
            eligible=torch.arange(6)[None]<requested[:,None]
            order=torch.where(eligible,radii,-1.).argsort(dim=-1,descending=True,stable=True)
            sorted_ids=selected.gather(1,order);radii=radii.gather(1,order)
            draws=torch.rand((len(ids),6,4),generator=generator)
            cloud=module.corners(robot)
            module._place_robot_cells(layout,ids,local,yaw,active,delta,sorted_ids,radii,requested,draws,
                                      cloud.amin(-2)-.01,cloud.amax(-2)+.01,3)
        else:
            module.place_obstacles(layout,ids,local,yaw,active,delta,robot,selected,requested,generator,batch_size=3)
        results.append((local,yaw,active,generator.get_state()))
    for actual,expected in zip(results[1],results[0]):
        torch.testing.assert_close(actual,expected,atol=0,rtol=0)
    assert not results[1][2][0,1] and results[1][2][0,2:].any()
    assert results[1][2][1].sum()==1


def test_fused_robot_clearance_matches_xy_and_height_boundaries(load_mdp):
    import importlib,torch
    module=importlib.import_module('_baseline_cpu_pkg.mdp.common.stabilize_sweep_layout')
    rng=torch.Generator().manual_seed(122)
    points=torch.rand(700,2,generator=rng)-.5
    centers=torch.rand(4,9,3,generator=rng)-.5
    half=torch.rand(4,9,3,generator=rng)*.08+.01
    overlap=torch.rand(4,6,9,generator=rng)>.35
    limits=torch.rand(4,6,generator=rng)*.1+.03
    # Include exact contact and the adjacent float32 points on both sides.
    centers[0,0,:2]=0.;half[0,0,:2]=.125;limits[0,0]=.125
    overlap[0,0]=False;overlap[0,0,0]=True
    points[0]=torch.tensor([.25,0.]);points[1]=points[0];points[2]=points[0]
    points[1,0]=torch.nextafter(points[0,0],torch.tensor(0.))
    points[2,0]=torch.nextafter(points[0,0],torch.tensor(1.))
    gap=(points[None,:,None]-centers[:,None,:,:2]).abs()-half[:,None,:,:2]
    distances=gap.clamp_min(0.).norm(dim=-1)
    expected=torch.where(overlap[:,:,None,:],distances[:,None],float('inf')).amin(-1)>=limits[:,:,None]
    actual=module._robot_clear_cells(points,centers,half,overlap,limits)
    assert torch.equal(actual,expected)
    assert actual[0,0,:3].tolist()==[True,False,True]


def test_convex_face_separation_requires_a_positive_gap(load_mdp):
    import importlib,itertools,numpy as np
    from scipy.spatial import ConvexHull
    module=importlib.import_module('_baseline_cpu_pkg.mdp.common.stabilize_sweep_geometry')
    vertices=np.array(list(itertools.product((-.1,.1),repeat=3)))
    planes=ConvexHull(vertices).equations
    # Overlap, the LP's 1um interior boundary, exact touch, an uncertain tiny
    # positive gap, and clear separation must all retain their original result.
    shifts=np.array([.1,.2-2e-6,.2,.2+2e-9,.2+1e-6])
    moved=vertices[None]+np.column_stack((shifts,np.zeros((len(shifts),2))))[:,None]
    proven=module.convex_face_separated(np.broadcast_to(planes,(len(shifts),*planes.shape)),moved)
    assert proven.tolist()==[False,False,False,False,True]
    for index in np.flatnonzero(proven):
        assert not module.convex_overlap(planes,ConvexHull(moved[index]).equations)


def test_convex_broad_phase_preserves_exact_lp_results(load_mdp):
    import importlib,numpy as np,torch
    from scipy.spatial import ConvexHull
    from isaaclab.utils.math import matrix_from_quat
    module=importlib.import_module('_baseline_cpu_pkg.mdp.common.stabilize_sweep_geometry')
    random=np.random.default_rng(1403)
    # Non-box polyhedra cover OBB false positives, actual collisions and nearly
    # touching poses. The two shells have different shapes and local offsets.
    clouds=[random.uniform(-.15,.15,(18,3)),random.uniform(-.12,.12,(22,3))]
    hulls=[ConvexHull(v) for v in clouds]
    geometry=object.__new__(module.RobotResetGeometry)
    geometry.hulls=[h.equations for h in hulls]
    geometry.vertex_clouds=[v[h.vertices] for v,h in zip(clouds,hulls)]
    geometry.face_subsets=geometry.hulls
    geometry.centers=torch.tensor([[.01,-.02,.005],[-.02,.01,-.005]])
    geometry.self_pairs=torch.tensor([[False,True],[False,False]])
    count=128
    centers=torch.tensor(random.uniform(-.20,.20,(count,2,3)),dtype=torch.float32)
    quats=torch.tensor(random.normal(size=(count,2,4)),dtype=torch.float32)
    quats/=quats.norm(dim=-1,keepdim=True)
    rotations=matrix_from_quat(quats)
    rotations[:,0,0,1]+=3e-8  # float32 matrices need not be exactly orthogonal.
    half=torch.full_like(centers,.25)
    boxes=(centers,half,rotations)
    far=(torch.full((count,1,3),10.),torch.full((count,1,3),.1),torch.eye(3).expand(count,1,3,3))
    expected=[];expected_proof=[]
    for row in range(count):
        transformed=[];world_vertices=[]
        for index in range(2):
            hull=geometry.hulls[index]
            normal=hull[:,:3]@rotations[row,index].numpy().T
            constant=hull[:,3]+hull[:,:3]@geometry.centers[index].numpy()-normal@centers[row,index].numpy()
            transformed.append(np.column_stack((normal,constant)))
            local=geometry.vertex_clouds[index]-geometry.centers[index].numpy()
            world_vertices.append(local@np.linalg.inv(rotations[row,index].numpy().astype(np.float64))+centers[row,index].numpy())
        expected.append(not module.convex_overlap(*transformed))
        expected_proof.append(bool(module.convex_face_separated(transformed[0][None],world_vertices[1][None])[0]
                                   or module.convex_face_separated(transformed[1][None],world_vertices[0][None])[0]))
    actual=geometry.valid(boxes,far,far)
    assert actual.tolist()==expected
    assert any(expected) and not all(expected)
    candidates=torch.stack((torch.arange(count),torch.zeros(count,dtype=torch.long),torch.ones(count,dtype=torch.long)),dim=1)
    assert geometry.convex_separated_kernel(boxes,candidates).tolist()==expected_proof
    # Existing hand exceptions remain authoritative, including overlapping
    # convex interiors: the accelerator must not introduce new self pairs.
    geometry.self_pairs.zero_()
    assert geometry.valid(boxes,far,far).all()


def test_fused_collision_fk_matches_tree_and_partial_rows(load_mdp):
    import importlib,torch
    module=importlib.import_module('_baseline_cpu_pkg.mdp.common.stabilize_sweep_geometry')
    random=torch.Generator().manual_seed(43)
    def quaternion():
        value=torch.randn(4,generator=random)
        return value/value.norm()
    geometry=object.__new__(module.RobotResetGeometry)
    geometry.base='base';geometry.steps=[]
    nodes=['base']
    for index in range(37):
        # Serial arm plus branching fingers, with both revolute and fixed joints.
        parent=nodes[max(0,index-1 if index<8 else 7+(index-8)//5*5)]
        child=f'node{index}';nodes.append(child)
        geometry.steps.append((child,(parent,torch.randn(3,generator=random)*.05,quaternion(),
            torch.randn(3,generator=random)*.05,quaternion(),index%3,index if index<18 else None)))
    geometry.owners=[*nodes,nodes[-1]]  # Two collision shapes may share a body.
    geometry.centers=torch.randn(len(geometry.owners),3,generator=random)*.02
    geometry.half=torch.full((len(geometry.owners),3),.03)
    joints=torch.randn(64,18,generator=random)*1.7
    position=torch.randn(64,3,generator=random)
    rotation=torch.randn(64,4,generator=random);rotation/=rotation.norm(dim=-1,keepdim=True)
    for ids in (torch.arange(64),torch.tensor([63,0,19,7])):
        actual=geometry.boxes_kernel(joints[ids],position[ids],rotation[ids])
        expected=geometry.boxes_reference(joints[ids],position[ids],rotation[ids])
        for observed,reference in zip(actual,expected):
            torch.testing.assert_close(observed,reference,atol=3e-6,rtol=0)
        assert torch.equal(actual[1],expected[1])


@pytest.mark.parametrize('case', ['mixed', 'partial_failure', 'all_failure'])
def test_initial_pose_checks_only_solved_ik_and_preserves_row_results(load_mdp, case):
    import torch
    from isaaclab.utils import math as math_utils
    scope=_reset_layout_planner()
    for name in ('quat_apply','quat_apply_inverse','quat_mul','subtract_frame_transforms','matrix_from_quat'):
        scope[name]=getattr(math_utils,name)
    scope['_prepare_fast_ik']=lambda env:None
    scope['_geometry']=lambda env:None
    scope['hand_pose']=lambda center,direction,radius,position,quaternion,*args,**kwargs:(center,quaternion)
    schedules={
        'mixed':[[True,False,True],[True,False],[False],[True]],
        'partial_failure':[[True,False,True],[False,False],[False,False],[False,False]],
        'all_failure':[[False,False,False]]*4,
    }
    draws=[];fk_inputs=[];collision_inputs=[]
    def solve(position,quaternion,initial,limits):
        ok=torch.tensor(schedules[case][len(draws)],dtype=torch.bool)
        assert len(ok)==len(position)
        draws.append(len(position))
        return position[:,:1]+1.,ok,torch.zeros(len(ok)),torch.zeros(len(ok))
    def boxes(joints,position,quaternion):
        fk_inputs.append(joints.clone())
        centers=joints[:,:1,None].expand(-1,1,3).clone()
        return centers,torch.ones_like(centers),torch.eye(3).expand(len(joints),1,3,3).clone()
    def valid(geometry,target,shelf):
        collision_inputs.append(geometry[0].clone())
        accepted=torch.ones(len(geometry[0]),dtype=torch.bool)
        if len(collision_inputs)==1:accepted[-1]=False
        return accepted
    ids=torch.tensor([2,0,3]);identity=torch.zeros(4,4);identity[:,0]=1.
    default=torch.zeros(4,2);default[:,0]=100.+torch.arange(4)
    bounds=torch.empty(4,17,2,3);bounds[:,:,0]=-.02;bounds[:,:,1]=.02
    env=NS(device='cpu',scene={'robot':NS(data=NS(default_joint_pos=default))},
        cfg=NS(scene=NS(ee_frame=NS(target_frames=[NS(offset=NS(rot=(1.,0.,0.,0.),pos=(0.,0.,0.)))]))),
        _fast_base_offset=(torch.zeros(4,3),identity),_fast_default_eef=(torch.zeros(4,3),identity),
        _revision_bounds_t=bounds,_stabilize_shelf_boxes=(torch.zeros(1,3),torch.ones(1,3)),
        _stabilize_hand_ids=[1],_goal_reset_arm_ids=[0],_stabilize_hand_vertices=torch.zeros(1,3),
        _goal_reset_rng=torch.Generator().manual_seed(391),_fast_initial=torch.zeros(4,1),
        _fast_limits=torch.zeros(4,1,2),_fast_ik=NS(solve=solve),
        _stabilize_robot_geometry=NS(boxes=boxes,valid=valid))
    local=torch.zeros(3,17,3);local[:,0,0]=ids.float()
    state=torch.zeros(3,13);state[:,3]=1.
    pose,accepted,diagnostics=scope['choose_initial_pose'](env,ids,local,torch.ones(3,2),state,state)
    if case=='mixed':
        expected=torch.tensor([3.,1.,4.]);assert accepted.tolist()==[True,True,True]
        assert pose.attempts.tolist()==[1,2,4]
        assert [len(value) for value in fk_inputs]==[2,1,1]
    elif case=='partial_failure':
        expected=torch.tensor([3.,100.,103.]);assert accepted.tolist()==[True,False,False]
        assert pose.attempts.tolist()==[1,4,4]
        assert [len(value) for value in fk_inputs]==[2,2]
    else:
        expected=default[ids,0];assert not accepted.any()
        assert pose.attempts.tolist()==[4,4,4]
        assert len(fk_inputs)==1 and not collision_inputs
    torch.testing.assert_close(pose.joints[:,0],expected,atol=0,rtol=0)
    torch.testing.assert_close(pose.boxes[0][:,0,0],expected,atol=0,rtol=0)
    assert len(collision_inputs)==(3 if case=='mixed' else 1 if case=='partial_failure' else 0)
    for diagnostic in diagnostics:
        for ik_ok,collision_free in zip(diagnostic['ik_valid'],diagnostic['collision_free']):
            assert (collision_free is None)==(not ik_ok)
    reference_rng=torch.Generator().manual_seed(391)
    for n in draws:torch.rand((n,3),generator=reference_rng)
    assert torch.equal(env._goal_reset_rng.get_state(),reference_rng.get_state())


@pytest.mark.parametrize('case', ['random', 'touching', 'near_parallel'])
def test_fused_collision_broadphase_preserves_reference_boundaries(load_mdp, case):
    import importlib
    import torch
    module=importlib.import_module('_baseline_cpu_pkg.mdp.common.stabilize_sweep_geometry')
    from isaaclab.utils.math import matrix_from_quat
    random=torch.Generator().manual_seed(678)
    n=256
    def quaternion():
        q=torch.randn(n,3,4,generator=random)
        return q/q.norm(dim=-1,keepdim=True)
    center=torch.randn(n,3,3,generator=random)*.2
    half=torch.rand(n,3,3,generator=random)*.1+.01
    rotation=matrix_from_quat(quaternion())
    target_half=torch.rand(n,1,3,generator=random)*.1+.01
    target_center=torch.randn(n,1,3,generator=random)*.2
    target_rotation=matrix_from_quat(quaternion()[:,:1])
    if case!='random':
        if case=='near_parallel':
            rotation=torch.eye(3).expand(n,3,3,3).clone()
            q=torch.zeros(n,1,4)
            angles=torch.linspace(0.,4e-6,n)[:,None]
            q[:,:,0]=torch.cos(angles/2);q[:,:,3]=torch.sin(angles/2)
            target_rotation=matrix_from_quat(q)
        else:
            target_rotation=rotation[:,:1].clone()
        offset=torch.linspace(-2e-6,2e-6,n)[:,None]
        distance=half[:,:1,0]+target_half[:,:,0]-1e-6+offset
        target_center=center[:,:1]+rotation[:,:1,:,0]*distance.unsqueeze(-1)
    boxes=(center,half,rotation)
    target=(target_center,target_half,target_rotation)
    shelf=(center[:,:1]+4.,half[:,:1],rotation[:,:1])
    geometry=object.__new__(module.RobotResetGeometry)
    geometry.self_pairs=torch.triu(torch.ones(3,3,dtype=torch.bool),diagonal=1)
    for ids in (torch.arange(n),torch.tensor([255,0,18,7])):
        values=[tuple(v[ids] for v in group) for group in (boxes,target,shelf)]
        expected=geometry.broadphase_reference(*values)
        actual=geometry.broadphase_kernel(*values)
        assert all(torch.equal(a,b) for a,b in zip(actual,expected))
    # Existing exclusions can change before reset; invalidate the cached pair table.
    geometry.self_pairs[0,1]=False
    expected=geometry.broadphase_reference(boxes,target,shelf)
    actual=geometry.broadphase_kernel(boxes,target,shelf)
    assert all(torch.equal(a,b) for a,b in zip(actual,expected))
    with torch.inference_mode():
        geometry.self_pairs=geometry.self_pairs.clone()
    actual=geometry.broadphase_kernel(boxes,target,shelf)
    assert all(torch.equal(a,b) for a,b in zip(actual,expected))


def _stabilize_runtime_without_sim(state_module):
    """Run the real family constructors/step around a minimal manager boundary."""
    import torch
    constructed=[]

    def observed_state(base):
        class ObservedState(base):
            def __init__(self, count, device):
                super().__init__(count,device)
                constructed.append(self)

            def finish(self, *args, **kwargs):
                self.received_height_limit=kwargs['height_limit']
                return super().finish(*args,**kwargs)
        return ObservedState

    class Scene(dict):
        def update(self, dt):
            self.updates.append(dt)

    class Base:
        def __init__(self, cfg, render_mode=None, **kwargs):
            self.cfg,self.device,self.num_envs=cfg,cfg.sim.device,cfg.scene.num_envs
            self.physics_dt=.01
            # Real managers can retain the task state. It must already be the
            # selected family state and must survive the rest of construction.
            self.state_seen_by_managers=self._stabilize_state
            self.phase_seen_by_managers=self._sweep_phase.clone()
            self._stabilize_state.initial_center[0,0]=17.
            vector=torch.zeros(self.num_envs,17,3)
            quaternion=torch.zeros(self.num_envs,17,4);quaternion[...,0]=1.
            self.scene=Scene(object_collection=NS(data=NS(
                object_com_ang_vel_w=vector.clone(),object_com_lin_vel_w=vector.clone(),
                object_com_pos_w=vector.clone(),object_link_quat_w=quaternion)),
                ee_frame=NS(data=NS(target_pos_w=torch.zeros(self.num_envs,1,3))))
            self.scene.updates=[]
            self._revision_initial_shelf_pos=torch.zeros(self.num_envs,3)
            goal=torch.zeros(self.num_envs,3);goal[:,0]=1.
            self.command_manager=NS(get_command=lambda name:goal)

        def step(self, action):
            for _ in range(self.cfg.decimation):
                self.scene.update(self.physics_dt)
            return action

    class Contacts:
        def __init__(self, env):
            self.state_seen_at_creation=env._stabilize_state
            self.force=torch.zeros(env.num_envs,3);self.force[:,0]=2.
            self.moment=torch.zeros_like(self.force)
            self.load=torch.ones(env.num_envs)
            self.valid=torch.ones(env.num_envs,dtype=torch.bool)
            self.contact=self.valid.clone();self.all_valid=self.valid.clone()
            self.begin()

        def begin(self):
            self.samples=0;self.force_sum=torch.zeros_like(self.force)

        def capture(self, env):
            self.samples+=1;self.force_sum+=self.force

    scope=_execute_definitions(MODULE_ROOT/'runtime/stabilize_sweep.py',dict(
        torch=torch,ManagerBasedRLEnv=Base,ContactSamples=Contacts,
        StabilizeState=observed_state(state_module.StabilizeState),
        SweepOnlyState=observed_state(state_module.SweepOnlyState),
        centroid=lambda env:torch.zeros(env.num_envs,3)))
    return scope,constructed


@pytest.mark.parametrize('runtime_name,sweep_only',[
    ('StabilizeSweepEnv',False),('SweepOnlyEnv',True)])
@pytest.mark.parametrize('height_limit',[.12,.075])
def test_family_runtime_constructs_selected_state_once_and_passes_height(
        load_mdp,runtime_name,sweep_only,height_limit):
    import importlib,torch
    states=importlib.import_module('_baseline_cpu_pkg.mdp.common.stabilize_sweep_state')
    runtime,constructed=_stabilize_runtime_without_sim(states)
    cfg=NS(scene=NS(num_envs=2),sim=NS(device='cpu'),seed=42,
        timeout_metric_window=3,reset_ik_backend='warp',max_obstacles=6,
        obstacle_observation_slots=16,decimation=2,sweep_contact_height_limit=height_limit)
    env=getattr(runtime,runtime_name)(cfg)
    expected_type=states.SweepOnlyState if sweep_only else states.StabilizeState
    assert len(constructed)==1 and isinstance(constructed[0],expected_type)
    assert env._stabilize_state is env.state_seen_by_managers is constructed[0]
    assert env._stabilize_contacts.state_seen_at_creation is constructed[0]
    assert env._sweep_phase is env._stabilize_state.achieved
    assert env.phase_seen_by_managers.tolist()==[sweep_only,sweep_only]
    assert env._stabilize_state.initial_center[0,0]==17.
    original_update=env.scene.update
    action=torch.zeros(2,12)
    assert env.step(action) is action
    assert env._stabilize_state.received_height_limit==height_limit
    assert env.scene.updates==[.01,.01] and env.scene.update==original_update


@pytest.mark.parametrize('state_name',['StabilizeState','SweepOnlyState'])
@pytest.mark.parametrize('height_limit',[None,.075])
def test_sweep_height_setting_preserves_strict_boundary_and_retreat(
        load_mdp,state_name,height_limit):
    import importlib,torch
    states=importlib.import_module('_baseline_cpu_pkg.mdp.common.stabilize_sweep_state')
    state=getattr(states,state_name)(3,'cpu')
    actual_limit=.12 if height_limit is None else height_limit
    center=torch.zeros(3,3);quat=torch.tensor([[1.,0.,0.,0.]]).expand(3,-1)
    goal=torch.tensor([[1.,0.,0.]]).expand(3,-1)
    state.reset(torch.arange(3),center,quat,goal)
    state.achieved[:]=True;state.begin()
    state.contact_history[:]=True;state.contact_count[:]=20
    height=torch.tensor([actual_limit-1e-4,actual_limit,actual_limit+1e-4])
    kwargs={} if height_limit is None else {'height_limit':height_limit}
    contact=torch.ones(3,dtype=torch.bool)
    force=torch.tensor([[2.,0.,0.]]).expand(3,-1)
    state.finish(center+torch.tensor([.02,0.,0.]),goal,force,contact,contact,
                 height,torch.zeros(3),**kwargs)
    torch.testing.assert_close(state.positive,torch.tensor([.02,0.,0.]),atol=1e-7,rtol=0)
    assert state.contact_count.tolist()==[20,18,18]
    # Retreat remains charged even when the EEF is at/above the contact ceiling.
    state.finish(center,goal,force,contact,contact,height,torch.zeros(3),**kwargs)
    torch.testing.assert_close(state.retreat,torch.full((3,),.02),atol=1e-7,rtol=0)
    assert not state.positive.any()
