"""Fast reset geometry, actual Warp arithmetic and isolated configuration."""
import importlib
import math
import os
from pathlib import Path

import pytest
import torch
from test_wrench_refactor_ik import chain
from test_wrench_refactor import pool_bounds
from test_final_configuration import configs
from test_omni_configuration import omni_config


@pytest.fixture(autouse=True)
def single_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def module(name):
    return importlib.import_module('_baseline_cpu_pkg.mdp.common.' + name)


def test_warp_matches_original_equations(load_mdp):
    import warp as wp
    wp.config.kernel_cache_dir = '/tmp/inspire_fast_warp_cache'
    m = module('wrench_refactor_ik')
    w = module('wrench_refactor_warp_ik')
    ik = chain(m)
    solver = w.WarpResetArmIK(ik)
    for n in (1, 4, 8, 16, 64, 128, 2048):
        seed = torch.tensor([[.3, -.5, 1., .2, -.6, -.2]]).repeat(n, 1)
        desired = seed + .2 * (torch.rand((n, 6), generator=torch.Generator().manual_seed(n)) - .5)
        p, q = ik.forward(desired)
        limits = torch.tensor([-6., 6.]).expand(n, 6, 2)
        actual = solver.solve(p, q, seed, limits)
        expected = ik.solve(p, q, seed, limits)
        assert torch.equal(actual[1], expected[1])
        assert actual[1].all()
        torch.testing.assert_close(actual[0], expected[0], atol=3e-5, rtol=1e-4)
        pp, qq = ik.forward(actual[0])
        assert (p - pp).norm(dim=-1).max() < .0005
        assert m.axis_angle_from_quat(m.quat_mul(q, m.quat_conjugate(qq))).norm(dim=-1).max() < .003
    actual = solver.solve(p + 10, q, seed, limits, iterations=4)
    assert not actual[1].any()


def test_circle_layout_geometry_and_reproducibility(load_mdp):
    m = module('wrench_refactor_layout')
    bounds = pool_bounds(128)
    layout = m.CircleResetLayout(bounds, 1.05 - bounds[..., 0, 2])
    ids = torch.arange(len(bounds))
    values = layout.sample(ids, torch.Generator().manual_seed(42))
    again = layout.sample(ids, torch.Generator().manual_seed(42))
    for a, b in zip(values, again):
        assert torch.equal(a, b)
    local, yaw, active, delta, requested = values
    assert torch.equal(active[:, 1:].sum(-1), requested)
    offset = layout.offset
    centers = local[..., :2] + torch.stack((yaw.cos()*offset[..., 0] - yaw.sin()*offset[..., 1],
                                          yaw.sin()*offset[..., 0] + yaw.cos()*offset[..., 1]), -1)
    # Independent pairwise distances and exact polygon buffering via Shapely.
    from scipy.optimize import lsq_linear
    import numpy as np
    for row in range(len(ids)):
        start = centers[row, 0].numpy()
        goal = start + delta[row].numpy()
        length = max(0., .2 - min(start[0], goal[0]))
        basis = np.stack((goal-start, [length, 0]), axis=1)
        objects = active[row, 1:].nonzero().flatten() + 1
        for i in objects.tolist():
            x, y = centers[row, i].tolist()
            r = layout.expanded[row, i].item()
            assert abs(x) + r <= .200001 and abs(y) + r <= .500001
            fit = lsq_linear(basis, np.array([x,y])-start, bounds=(0.,1.), tol=1e-12)
            distance = np.linalg.norm(basis @ fit.x - (np.array([x,y])-start))
            assert distance >= r + layout.expanded[row, 0].item() - 1e-6
            for j in objects.tolist():
                if i < j:
                    assert (centers[row, i] - centers[row, j]).norm() >= r + layout.expanded[row, j] - 1e-6
    assert set(requested.tolist()) == set(range(7))


def test_red_degenerate_and_inside(load_mdp):
    m = module('wrench_refactor_layout')
    points = torch.tensor([[.1, 0.], [-.1, 0.], [.1, .1]])
    start = torch.zeros(2, 2)
    goal = torch.tensor([[.1, 0.], [0., .1]])
    distance = m.CircleResetLayout.red_distance(points, start, goal)
    torch.testing.assert_close(distance, torch.tensor([[0., .1, .1], [0., .1, 0.]]))


def test_no_space_is_explicit_failure(load_mdp):
    m = module('wrench_refactor_layout')
    bounds = pool_bounds(4)
    bounds[:, 1:, 0, :2] = -.3
    bounds[:, 1:, 1, :2] = .3
    layout = m.CircleResetLayout(bounds, 1.05 - bounds[..., 0, 2])
    with pytest.raises(RuntimeError, match='no certified free cell'):
        layout.sample(torch.arange(4), torch.Generator().manual_seed(42))


def test_fast_event_isolation(configs):
    events = configs('events', 'wrench_refactor_relative_ik_test')
    parent = events.WrenchRefactorRelativeIKV0_2TestEventsCfg().to_dict()
    fast = events.WrenchRefactorRelativeIKV0_2_2_2FastTestEventsCfg()
    assert fast.reset_all is None and fast.robot_initial_pose is None
    assert fast.object_spawn.func.__name__ == 'fast_reset_scene'
    assert fast.object_spawn.params['ik_params'] == parent['robot_initial_pose']['params']
    assert events.WrenchRefactorRelativeIKV0_2TestEventsCfg().to_dict() == parent


def test_full_fast_config_matches_parent_except_reset(configs, omni_config, monkeypatch):
    import copy
    import sys
    from types import SimpleNamespace as NS
    from conftest import MODULE_ROOT
    from test_omni_configuration import _AssetNode, _execute_definitions
    monkeypatch.setitem(sys.modules, 'isaaclab.utils.noise', importlib.import_module('_omni_official_utils.noise'))
    robot = sys.modules['_baseline_cpu_pkg.assets.robot']
    tactile = importlib.import_module('_baseline_cpu_pkg.assets.tactile_regions')
    contacts = module('contact_sweep_clutter_fusion_contacts')
    ns = dict(vars(omni_config.baseline))
    ns.update(vars(robot))
    stem = 'wrench_refactor_relative_ik_test'
    ns.update(Path=Path, __file__=str(MODULE_ROOT/f'assets/scenes/{stem}.py'), sim=ns['sim_utils'],
              RevisionRobotSpawnerCfg=lambda **kw: _AssetNode(**kw, sensor_height=.0254),
              WrenchRefactorShelfSpawnerCfg=_AssetNode, RevisionObjectSpawnerCfg=_AssetNode,
              REVISION_OBJECT_KINDS=('cylinder','cube','mug','bottle','003_cracker_box','004_sugar_box','005_tomato_soup_can','002_master_chef_can'),
              TACTILE_PARENT_BODY_NAMES=tactile.TACTILE_PARENT_BODY_NAMES,
              parent_sensor_name=tactile.parent_sensor_name, object_hand_sensor_name=contacts.object_hand_sensor_name,
              make_robot_cfg=omni_config.baseline.make_robot_cfg)
    ns['_tool_frame'] = lambda frames, body='inspire_base_link', **kw: _AssetNode(target_frames=[
        _AssetNode(prim_path=f'Robot/{body}', offset=_AssetNode(pos=xyz, rot=(.5,-.5,-.5,-.5))) for name,xyz in frames])
    import ast
    path = MODULE_ROOT/f'assets/scenes/{stem}.py'
    tree = ast.parse(path.read_text())
    tree.body = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == '_robot']
    exec(compile(tree, str(path), 'exec'), ns)
    scene = _execute_definitions(path, ns, {'WrenchRefactorRelativeIKV0_2TestSceneCfg'})
    ns.update(vars(scene))
    for role in ('commands','observations','rewards','events','terminations','curricula'):
        ns.update({k:v for k,v in vars(configs(role,stem)).items() if k.startswith('WrenchRefactor')})
    ns.update(WrenchRefactorRelativeIKV0_2TestActionsCfg=lambda: _AssetNode(arm_action=_AssetNode()),
              WrenchRefactorRelativeIKV0_2TestOSCActionCfg=NS(OffsetCfg=_AssetNode),
              GroupedResetObjectSpawnerCfg=_AssetNode, ViewerCfg=_AssetNode)
    envs = _execute_definitions(MODULE_ROOT/f'envs/{stem}_envs.py', ns)
    Parent = envs.WrenchRefactorRelativeIKV0_2_2_2TestEnvCfg
    Child = envs.WrenchRefactorRelativeIKV0_2_2_2FastTestEnvCfg
    before = Parent().to_dict()
    child = Child()
    current = child.to_dict()
    expected = copy.deepcopy(before)
    changed = ('events','max_obstacles','placement_clearance','layout_cell_size','layout_geometry_epsilon',
               'reset_ik_backend','debug_initial_ik_bounds','max_gray_obstacles','max_white_obstacles',
               'gray_padding','red_margin','rear_offset')
    for key in changed:
        expected[key] = current[key]
    assert current == expected
    assert Child.__bases__ == (Parent,)
    assert Parent().to_dict() == before
    assert child.max_obstacles == 6 and child.max_gray_obstacles is None
    assert not child.debug_initial_ik_bounds
    newest = envs.WrenchRefactorRelativeIKV0_2_2_3FastTestEnvCfg
    before_fast = Child().to_dict()
    new = newest().to_dict()
    expected_new = copy.deepcopy(before_fast)
    expected_new['rewards']['hand_contact'] = new['rewards']['hand_contact']
    expected_new['rewards']['goal_hold'] = None
    expected_new['curriculum']['goal_speed_scale'] = None
    assert new == expected_new
    assert newest.__bases__ == (Child,)
    assert Child().to_dict() == before_fast
    Latest = envs.WrenchRefactorRelativeIKV0_2_2_4EnvCfg
    latest = Latest().to_dict()
    expected_latest = copy.deepcopy(new)
    expected_latest['rewards']['object_acceleration'] = None
    expected_latest['curriculum']['object_free_acceleration'] = None
    expected_latest['curriculum']['acceleration_weight'] = None
    expected_latest['terminations']['success'] = latest['terminations']['success']
    expected_latest['debug_spawn_regions'] = False
    expected_latest['debug_spawn_max_envs'] = 4
    expected_latest['timeout_metric_window'] = 10000
    assert latest == expected_latest
    assert Latest.__bases__ == (newest,)
    assert newest().to_dict() == new
    assert latest['terminations']['success']['time_out'] is True
    assert list(latest['terminations'])[-1] == 'success'
    Final = envs.WrenchRefactorRelativeIKV0_2_2_5EnvCfg
    final = Final().to_dict()
    expected_final = copy.deepcopy(latest)
    expected_final['rewards']['goal_distance'] = final['rewards']['goal_distance']
    expected_final['rewards']['direction_deviation'] = final['rewards']['direction_deviation']
    expected_final['rewards']['hand_contact'] = None
    assert final == expected_final
    assert Final.__bases__ == (Latest,)
    assert Latest().to_dict() == latest
    Follow = envs.WrenchRefactorRelativeIKV0_2_2_6EnvCfg
    follow = Follow().to_dict()
    expected_follow = copy.deepcopy(latest)
    expected_follow['rewards']['reaching'] = follow['rewards']['reaching']
    assert follow == expected_follow
    assert Follow.__bases__ == (Latest,)
    assert Latest().to_dict() == latest
    assert follow['rewards']['goal_distance']['weight'] == 100.
    assert follow['rewards']['hand_contact'] is not None
    assert 'direction_deviation' not in follow['rewards']


def test_integrated_reset_writes_final_selected_states_once(configs):
    from types import SimpleNamespace as NS
    events = configs('event_functions', 'wrench_refactor_relative_ik_test')
    m = module('wrench_refactor_layout')
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
    ids = torch.tensor([1,4,7]); keep = torch.tensor([0,2,3,5,6])
    events.fast_reset_scene(env, ids, dict(height_offset=.01,front_clearance=(.08,.10),lateral_offset=(.10,.18),height_jitter=.02))
    assert scene['robot'].writes == 1
    torch.testing.assert_close(scene['robot'].joints[ids,:6],angles.expand(3,-1))
    assert (scene['robot'].joints[ids,6:] == .2).all()
    assert (scene['robot'].joints[keep] == 9).all()
    assert (scene['object_collection'].state[keep] == 9).all()
    assert not scene['object_collection'].state[ids,:,7:].any()
    assert (env._active_object_mask[ids,1:].sum(-1) == env._final_actual_population[ids]).all()
    assert not env._revision_layout_complete[keep].any()
    assert not env._initial_obstacle_observation[keep].any()
    torch.testing.assert_close(scene['shelf'].values[:,:3],shelf_root[ids,:3]+origin[ids])
    before_state = scene['object_collection'].state.clone()
    before_joints = scene['robot'].joints.clone()
    solver.solve = lambda p,q,initial,limits: (angles.expand(len(p),-1),torch.zeros(len(p),dtype=torch.bool),torch.ones(len(p)),torch.ones(len(p)))
    with pytest.raises(RuntimeError, match='Fast reset IK failed'):
        events.fast_reset_scene(env, ids, dict(height_offset=.01,front_clearance=(.08,.10),lateral_offset=(.10,.18),height_jitter=.02))
    assert scene['robot'].writes == 1
    assert torch.equal(scene['object_collection'].state, before_state)
    assert torch.equal(scene['robot'].joints, before_joints)


def test_warp_nonidentity_fixed_joint_frames(load_mdp):
    import warp as wp
    wp.config.kernel_cache_dir = '/tmp/inspire_fast_warp_cache'
    m = module('wrench_refactor_ik')
    w = module('wrench_refactor_warp_ik')
    ik = chain(m)
    ik.joints.append((torch.tensor([.02,-.01,.03]),torch.tensor([.5,.5,.5,.5]),
                      torch.tensor([.01,.02,-.01]),torch.tensor([.5,-.5,.5,-.5]),None,None))
    seed = torch.tensor([[.3,-.5,1.,.2,-.6,-.2]]).repeat(64,1)
    desired = seed + .2 * (torch.rand((64,6), generator=torch.Generator().manual_seed(73))-.5)
    p,q = ik.forward(desired)
    limits = torch.tensor([-6.,6.]).expand(64,6,2)
    actual = w.WarpResetArmIK(ik).solve(p,q,seed,limits)
    expected = ik.solve(p,q,seed,limits)
    assert torch.equal(actual[1],expected[1]) and actual[1].all()
    torch.testing.assert_close(actual[0],expected[0],atol=3e-5,rtol=1e-4)


@pytest.mark.skipif(not os.environ.get('FAST_LAYOUT_STRESS'), reason='explicit large geometry validation')
def test_large_circle_stress(load_mdp):
    import json
    import time
    n = int(os.environ['FAST_LAYOUT_STRESS'])
    m = module('wrench_refactor_layout')
    histogram = torch.zeros(7,dtype=torch.long)
    generator = torch.Generator().manual_seed(20261109)
    started = time.perf_counter()
    for begin in range(0,n,256):
        count = min(256,n-begin)
        bounds = pool_bounds(count)
        # Geometry stress uses explicitly synthetic offset boxes, not USD assets.
        scale = .9 + .2 * torch.rand((count,17,1,1), generator=generator)
        bounds *= scale
        layout = m.CircleResetLayout(bounds,1.05-bounds[...,0,2])
        local,yaw,active,delta,requested = layout.sample(torch.arange(count),generator,
            excluded_x_half_angle=math.radians((begin//256 % 7)*5))
        histogram += requested.bincount(minlength=7)
        assert torch.equal(active[:,1:].sum(-1),requested)
        off=layout.offset
        centers=local[...,:2]+torch.stack((yaw.cos()*off[...,0]-yaw.sin()*off[...,1],
                                         yaw.sin()*off[...,0]+yaw.cos()*off[...,1]),-1)
        mask=active.clone();mask[:,0]=False
        assert ((centers[...,0].abs()+layout.expanded <= .200001)|~mask).all()
        assert ((centers[...,1].abs()+layout.expanded <= .500001)|~mask).all()
        distances=torch.cdist(centers,centers)
        pairs=mask[:,:,None]&mask[:,None,:]&~torch.eye(17,dtype=torch.bool)[None]
        assert ((distances >= layout.expanded[:,:,None]+layout.expanded[:,None,:]-1e-6)|~pairs).all()
        assert torch.isfinite(local).all()
    assert (histogram.float()/n-1/7).abs().max()<.01
    out=Path('outputs/relative_ik_fast_validation');out.mkdir(parents=True,exist_ok=True)
    (out/'synthetic_layout_stress.json').write_text(json.dumps(dict(
        samples=n, geometry='synthetic offset bounds, uniform scale 0.9-1.1; NOT actual USD',
        counts=histogram.tolist(),failures=0,seconds=time.perf_counter()-started),indent=2))


def test_warp_actual_usd_chain(load_mdp, tensor_math):
    """Use authored USD joints plus the production Axia fixed-joint construction."""
    from conftest import MODULE_ROOT
    import ast
    import sys
    import warp as wp
    try:
        from pxr import Usd, UsdGeom, UsdPhysics, Gf, Sdf
    except ImportError:
        pytest.skip('USD native libraries require the Isaac Sim library path')
    wp.config.kernel_cache_dir = '/tmp/inspire_fast_warp_cache'
    path = MODULE_ROOT/'assets/robot.py'
    ns = dict(Usd=Usd, UsdGeom=UsdGeom, UsdPhysics=UsdPhysics, Gf=Gf, Sdf=Sdf)
    tree = ast.parse(path.read_text())
    names = {'_world_transform','_offset_matrix','_matrix_to_pose','_set_world_transform','_create_fixed_joint'}
    tree.body = [node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name in names]
    exec(compile(tree,str(path),'exec'),ns)
    stage=Usd.Stage.Open(str(MODULE_ROOT/'assets/data/wrench_right_hand/ur5e_inspire_right.usda'))
    root=str(stage.GetDefaultPrim().GetPath())
    mount=ns['_world_transform'](stage.GetPrimAtPath(root+'/inspire_base_link'))
    sensor=UsdGeom.Xform.Define(stage,root+'/axia80_link').GetPrim()
    ns['_set_world_transform'](sensor, ns['_offset_matrix']((0.,0.,.0254/2))*mount)
    ns['_create_fixed_joint'](stage,root+'/joints/ur5e_to_axia80_joint',
                              stage.GetPrimAtPath(root+'/wrist_3_link'),sensor,mount)
    arm=sys.modules['_baseline_cpu_pkg.assets.robot'].ARM_JOINT_NAMES
    m=module('wrench_refactor_ik');w=module('wrench_refactor_warp_ik')
    ik=m.ResetArmIK.from_stage(stage,root,arm,(0.,0.,.1427),(.5,-.5,-.5,-.5),'cpu')
    n=2048
    initial=torch.tensor([[0.,-2.2,2.2,0.,1.57,-math.pi/2]]).repeat(n,1)
    position,rotation=ik.forward(initial)
    base_p=torch.tensor([[0.,0.,.79505]]).expand(n,-1)
    base_q=torch.tensor([[0.,0.,0.,1.]]).expand(n,-1)
    shelf_p=torch.tensor([[-.7,0.,0.]]).expand(n,-1)
    default_local=base_p+tensor_math.quat_apply(base_q,position)-shelf_p
    default_local[:,2]+=.01
    rng=torch.Generator().manual_seed(42)
    target=(torch.rand(n,3,generator=rng)-.5)*.04
    sweep=torch.rand(n,3,generator=rng)-.5
    desired=m.sample_initial_eef(target,sweep,default_local,rng,(.08,.10),(.10,.18),.02)
    desired_base=tensor_math.quat_apply_inverse(base_q,shelf_p+desired-base_p)
    limits=torch.tensor([-2*math.pi,2*math.pi]).expand(n,6,2)
    actual=w.WarpResetArmIK(ik).solve(desired_base,rotation,initial,limits)
    expected=ik.solve(desired_base,rotation,initial,limits)
    assert torch.equal(actual[1],expected[1]) and actual[1].all()
    torch.testing.assert_close(actual[0],expected[0],atol=5e-5,rtol=1e-4)
    p,q=ik.forward(actual[0])
    pe=(p-desired_base).norm(dim=-1)
    re=m.axis_angle_from_quat(m.quat_mul(rotation,m.quat_conjugate(q))).norm(dim=-1)
    assert pe.max()<.0005 and re.max()<.003
    import json
    out=Path('outputs/relative_ik_fast_validation');out.mkdir(parents=True,exist_ok=True)
    (out/'usd_ik_cpu.json').write_text(json.dumps(dict(samples=n,device='Warp CPU',
        max_joint_difference=float((actual[0]-expected[0]).abs().max()),
        max_position_error=float(pe.max()),max_rotation_error=float(re.max())),indent=2))


def test_spawn_region_footprints_match_circle_geometry(load_mdp):
    m = module('wrench_refactor_layout')
    bounds = pool_bounds(2)
    layout = m.CircleResetLayout(bounds, 1.05-bounds[..., 0, 2])
    ids = torch.arange(2)
    start = torch.tensor([[0., 0.], [0., .1]])
    goal = start + torch.tensor([0., .1])
    centers = torch.zeros(2, 17, 2)
    centers[:, 1] = torch.tensor([-.1, -.3])
    active = torch.zeros(2, 17, dtype=torch.bool)
    active[0, 1] = True
    colors = m.spawn_region_classes(layout, ids, start, goal, centers, active)
    red = layout.red_distance(layout.cells, start, goal) <= layout.expanded[:, 0, None] + layout.cell_margin
    assert torch.equal(colors == 1, red)
    assert (colors[0] == 2).any() and not (colors[1] == 2).any()
    assert (colors == 0).any()
    occupied = (layout.cells - centers[0, 1]).norm(dim=-1) <= layout.expanded[0, 1] + layout.cell_margin
    assert torch.equal(colors[0] == 2, occupied & ~red[0])
    # Deterministic overlay leaves the sampler and its random stream untouched.
    assert layout._scratch is None


def test_spawn_overlay_partial_reset_snapshots_and_no_colliders(configs, monkeypatch):
    import ast
    import sys
    from types import SimpleNamespace as NS, ModuleType
    from conftest import MODULE_ROOT
    from isaaclab.utils.math import quat_apply
    sim = ModuleType('isaaclab.sim')
    monkeypatch.setitem(sys.modules, 'isaaclab.sim', sim)
    monkeypatch.setattr(sim, 'CuboidCfg', lambda **kw: NS(**kw), raising=False)
    monkeypatch.setattr(sim, 'PreviewSurfaceCfg', lambda **kw: NS(**kw), raising=False)
    markers = ModuleType('isaaclab.markers')
    class Marker:
        def __init__(self, cfg): self.cfg = cfg
        def visualize(self, **kw): self.payload = kw
    markers.VisualizationMarkers = Marker
    markers.VisualizationMarkersCfg = lambda **kw: NS(**kw)
    monkeypatch.setitem(sys.modules, 'isaaclab.markers', markers)
    tree = ast.parse((MODULE_ROOT/'runtime/wrench_refactor_relative_ik_test.py').read_text())
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'WrenchRefactorRelativeIKV0_2_2_4Env')
    ns = {'torch': torch, 'WrenchRefactorRelativeIKV0_2_2_2FastTestEnv': object,
          '__package__': '_baseline_cpu_pkg.runtime'}
    exec(compile(ast.Module(body=[node], type_ignores=[]), '<overlay-runtime>', 'exec'), ns)
    env = ns[node.name].__new__(ns[node.name])
    env.num_envs, env.device = 3, 'cpu'
    env.cfg = NS(debug_spawn_max_envs=2)
    bounds = pool_bounds(3)
    env._fast_layout = module('wrench_refactor_layout').CircleResetLayout(bounds, 1.05-bounds[..., 0, 2])
    env._revision_initial_shelf_pos = torch.tensor([[0., 0., 0.], [2., 0., 0.], [4., 0., 0.]])
    env._revision_initial_shelf_quat = torch.tensor([[1., 0., 0., 0.]]).repeat(3, 1)
    env._revision_initial_shelf_quat[1] = torch.tensor([2**-.5, 0., 0., 2**-.5])
    local = torch.zeros(3, 17, 3); local[..., 2] = 1.1
    env._revision_spawn_positions = env._revision_initial_shelf_pos[:, None] + quat_apply(
        env._revision_initial_shelf_quat[:, None].expand(-1, 17, -1), local)
    q = env._revision_initial_shelf_quat[:, None].expand(-1, 17, -1).clone()
    env.scene = {'object_collection': NS(data=NS(object_quat_w=q))}
    goals = env._revision_spawn_positions[:, 0].clone(); goals[:, 1] += .1
    env.command_manager = NS(get_command=lambda _: goals)
    env._active_object_mask = torch.zeros(3, 17, dtype=torch.bool)
    env._update_spawn_regions(torch.tensor([0]))
    first = env._spawn_region_positions[0].clone()
    assert env._spawn_region_marker.payload['translations'].shape == (16000, 3)
    env._update_spawn_regions(torch.tensor([1, 2]))
    assert env._spawn_region_marker.payload['translations'].shape == (32000, 3)
    torch.testing.assert_close(env._spawn_region_positions[0], first)
    assert env._spawn_region_active.tolist() == [True, True]
    for cfg in env._spawn_region_marker.cfg.markers.values():
        assert cfg.rigid_props is None and cfg.collision_props is None
    local_cells = torch.cat((env._fast_layout.cells, torch.full((16000, 1), 1.052)), -1)
    expected = env._revision_initial_shelf_pos[1] + quat_apply(
        env._revision_initial_shelf_quat[1].expand(16000, -1), local_cells)
    torch.testing.assert_close(env._spawn_region_positions[1], expected)
