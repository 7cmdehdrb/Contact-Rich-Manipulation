"""Config isolation, real GPU reset validation and comparable timings. No training."""
import argparse
import json
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from isaaclab.app import AppLauncher
from cli_args import apply_renderer_compatibility
parser=argparse.ArgumentParser()
parser.add_argument('--version',default='v0_2_2',choices=['v0_2_1','v0_2_2'])
parser.add_argument('--num_envs',type=int,default=4)
parser.add_argument('--resets',type=int,default=30)
parser.add_argument('--local-shelf',action='store_true',help='Use a local geometric shelf fixture if Nucleus is unavailable')
parser.add_argument('--output',type=Path,default=Path('outputs/final_compact'))
AppLauncher.add_app_launcher_args(parser)
args=parser.parse_args()
apply_renderer_compatibility(args)
app=AppLauncher(args).app
try:
    if args.local_shelf:
        import os
        from pxr import Usd,UsdGeom,UsdPhysics,Gf
        args.output.mkdir(parents=True,exist_ok=True)
        shelf_path=(args.output/'shelf_fixture.usda').resolve()
        fixture=Usd.Stage.CreateNew(str(shelf_path))
        root=UsdGeom.Xform.Define(fixture,'/Shelf').GetPrim()
        fixture.SetDefaultPrim(root)
        UsdGeom.SetStageUpAxis(fixture,UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(fixture,1.)
        UsdPhysics.RigidBodyAPI.Apply(root).CreateKinematicEnabledAttr(True)
        for name,center,scale in [('base',(0,0,1.025),(.4,1.,.05)),
                                  ('back',(-.225,0,1.25),(.05,1.,.4)),
                                  ('left',(0,-.525,1.25),(.4,.05,.4)),
                                  ('right',(0,.525,1.25),(.4,.05,.4))]:
            cube=UsdGeom.Cube.Define(fixture,'/Shelf/'+name)
            cube.CreateSizeAttr(1.)
            cube.AddTranslateOp().Set(Gf.Vec3d(*center))
            cube.AddScaleOp().Set(Gf.Vec3d(*scale))
            UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
        fixture.GetRootLayer().Save()
        os.environ['INSPIRE_SWEEP_SHELF_USD_PATH']=str(shelf_path)
    import torch
    import gymnasium as gym
    from pxr import UsdPhysics
    import inspire_sweep_baseline
    from isaaclab_tasks.utils import load_cfg_from_registry
    from isaaclab.utils.math import quat_apply_inverse,quat_mul,quat_conjugate
    from test_bbox_scene_layout import assert_geometry,layout
    prefix='Isaac-Inspire-Final-GoalAccuracy-'
    before={v:load_cfg_from_registry(prefix+v,'env_cfg_entry_point').to_dict() for v in ('v0','v0_1','v0_2','v0_3','v1','v0_2_1')}
    child=load_cfg_from_registry(prefix+'v0_2_2','env_cfg_entry_point')
    parent=load_cfg_from_registry(prefix+'v0_2_1','env_cfg_entry_point')
    assert type(child).__bases__==(type(parent),)
    expected=parent.to_dict()
    for key in ('scene','events','max_gray_obstacles','max_white_obstacles'):expected[key]=child.to_dict()[key]
    assert expected==child.to_dict()
    assert child.events.object_spawn.params==parent.events.object_spawn.params
    expected_ppo=load_cfg_from_registry(prefix+'v0_2_1','rsl_rl_cfg_entry_point').to_dict()
    expected_ppo['experiment_name']='inspire_final_goal_accuracy_v0_2_2'
    assert expected_ppo==load_cfg_from_registry(prefix+'v0_2_2','rsl_rl_cfg_entry_point').to_dict()
    assert len(child.scene.object_collection.rigid_objects)==12
    for v,snapshot in before.items():assert snapshot==load_cfg_from_registry(prefix+v,'env_cfg_entry_point').to_dict()
    cfg=child if args.version=='v0_2_2' else parent
    cfg.scene.num_envs=args.num_envs
    cfg.sim.device=args.device
    cfg.seed=42
    cfg.log_dir=str(args.output)
    args.output.mkdir(parents=True,exist_ok=True)
    start=time.perf_counter()
    env=gym.make(prefix+args.version,cfg=cfg)
    raw=env.unwrapped
    obs,_=env.reset(seed=42)
    torch.cuda.synchronize()
    startup=time.perf_counter()-start
    assert {k:v.shape[-1] for k,v in obs.items()}=={'policy':57,'obstacles':80,'privileged':3}
    pool=raw.scene['object_collection']
    physical_count=pool.physical.num_objects
    stage=raw.sim.stage
    def materials():return sum(1 for p in stage.Traverse() if p.HasAPI(UsdPhysics.MaterialAPI))
    material_before=materials()
    if args.version=='v0_2_2':
        assert physical_count==12 and pool.num_objects==10
        assert material_before<100, material_before
        assert not stage.GetPrimAtPath('/World/FinalCompactTemplates')
    def validate(ids):
        shelf=raw.scene['shelf'].data
        q=shelf.root_quat_w[ids,None].expand(-1,pool.num_objects,-1)
        local=quat_apply_inverse(q,pool.data.object_pos_w[ids]-shelf.root_pos_w[ids,None])
        relative=quat_mul(quat_conjugate(q),pool.data.object_quat_w[ids])
        yaw=2*torch.atan2(relative[...,3],relative[...,0])
        bounds=raw._final_object_bounds_t[ids]
        half=(bounds[...,1,:2]-bounds[...,0,:2])/2
        offset=(bounds[...,1,:2]+bounds[...,0,:2])/2
        center=local[...,:2]+torch.einsum('bmi,bmij->bmj',offset,layout.axes(yaw))
        board=torch.stack((center[...,1],.2-center[...,0]),-1)
        delta=quat_apply_inverse(shelf.root_quat_w[ids],raw.sweep_dir[ids])[:,:2]
        target=board[:,0]
        goal=target+torch.stack((delta[:,1],-delta[:,0]),-1)
        indices=list(range(1,10)) if pool.num_objects==10 else [1,2,3,4,11,12,13,14,15]
        from types import SimpleNamespace as NS
        result=NS(centers=board[:,indices],yaw=yaw[:,indices]-torch.pi/2,
            active=raw._active_object_mask[ids][:,indices],actual=raw._final_actual_population[ids],
            requested=torch.tensor([4,5],device=raw.device).expand(len(ids),-1))
        fraction=raw.event_manager.get_term_cfg('object_spawn').params.get('gray_red_padding_fraction',0.)
        assert_geometry(result,(target,goal,half[:,0],yaw[:,0]-torch.pi/2,half[:,indices]),
            layout.LayoutConfig(gray_padding=.06*(1+fraction)))
    ids=torch.arange(args.num_envs,device=raw.device)
    validate(ids[:min(4,len(ids))])
    times=[]
    seen=set()
    torch.cuda.reset_peak_memory_stats()
    for i in range(args.resets):
        raw.set_learning_iteration((0,2500,4500,5000,10000)[i%5])
        subset=ids[::2] if i%2 else ids
        untouched=pool.data.object_link_state_w[ids[1::2]].clone()
        torch.cuda.synchronize()
        start=time.perf_counter()
        raw._reset_idx(subset)
        torch.cuda.synchronize()
        times.append(time.perf_counter()-start)
        if i%2:torch.testing.assert_close(pool.data.object_link_state_w[ids[1::2]],untouched)
        validate(subset[:min(4,len(subset))])
        if args.version=='v0_2_2':
            physics=raw._clutter_physics
            material=pool.physical.root_physx_view.get_material_properties().cpu()
            mapped=material[pool.physics_rows]
            expected_static=physics.static_friction[:,1].repeat(10)
            torch.testing.assert_close(mapped[:,0,0],expected_static,atol=1e-5,rtol=1e-5)
            actual_mass=pool.physical.root_physx_view.get_masses().cpu()[pool.physics_rows,0]
            torch.testing.assert_close(actual_mass,physics.mass.repeat(10))
        seen.update(raw.target_kind_id.cpu().tolist())
        for _ in range(2):
            obs,reward,_,_,_=env.step(torch.zeros(args.num_envs,8,device=raw.device))
            assert torch.isfinite(reward).all()
        if args.version=='v0_2_2':
            assert (pool.mapping.sort(-1).values.diff(dim=-1)>0).all()
            assert (pool.mapping[:,1:]>=2).all()
            matrices=raw._role_sensor_bank.forces()
            physical=torch.stack([s.data.force_matrix_w for s in raw._role_sensor_bank.sensors],1)
            torch.testing.assert_close(matrices,physical[pool.env_rows[:,None],pool.mapping])
    step_times=[]
    for _ in range(20):
        torch.cuda.synchronize()
        start=time.perf_counter()
        env.step(torch.zeros(args.num_envs,8,device=raw.device))
        torch.cuda.synchronize()
        step_times.append(time.perf_counter()-start)
    assert materials()==material_before
    report=dict(version=args.version,num_envs=args.num_envs,physical_objects_per_env=physical_count,
        shelf_fixture=args.local_shelf,
        logical_objects=pool.num_objects,usd_physics_materials=material_before,reset_cycles=args.resets,
        startup_s=startup,full_reset_mean_ms=1000*sum(times[::2])/len(times[::2]),
        partial_reset_mean_ms=1000*sum(times[1::2])/len(times[1::2]),
        step_mean_ms=1000*sum(step_times)/len(step_times),seen_target_variants=sorted(seen),
        torch_peak_allocated_mb=torch.cuda.max_memory_allocated()/1024**2,
        status='passed')
    (args.output/'result.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('FINAL_COMPACT_PASS',report,flush=True)
    env.close()
except BaseException:
    import traceback
    traceback.print_exc()
    raise
finally:
    app.close()
