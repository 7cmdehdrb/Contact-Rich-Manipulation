"""Isolated support regression: real shelf props on a flat static floor, no robot."""
import sys,json,os,traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from isaaclab.app import AppLauncher
app=AppLauncher(headless=True).app
try:
 import torch,numpy as np
 import isaaclab.sim as sim
 from isaaclab.assets import RigidObjectCollection,RigidObjectCollectionCfg,RigidObjectCfg
 from inspire_sweep_baseline.assets.scenes.contact_sweep_clutter_fusion import ContactSweepClutterFusionSceneCfg
 from inspire_sweep_baseline.assets.contact_sweep_clutter_fusion_holding_osc_object_random_spawners import prop_variants
 from pxr import Usd,UsdGeom,UsdPhysics
 ctx=sim.SimulationContext(sim.SimulationCfg(dt=.01,device='cuda:0'))
 floor=sim.GroundPlaneCfg();floor.func('/World/Floor',floor)
 variants=prop_variants(ContactSweepClutterFusionSceneCfg().object_collection.rigid_objects['mass_cylinder'].spawn)
 cfg=RigidObjectCollectionCfg(rigid_objects={f'prop_{i}':RigidObjectCfg(prim_path=f'/World/Prop_{i}',spawn=c,init_state=RigidObjectCfg.InitialStateCfg(pos=(i*.3,0.,0.))) for i,c in enumerate(variants)})
 objects=RigidObjectCollection(cfg)
 stage=sim.get_current_stage()
 for i in range(len(variants)):
  root=stage.GetPrimAtPath(f'/World/Prop_{i}')
  colliders=[p for p in Usd.PrimRange(root) if p.HasAPI(UsdPhysics.CollisionAPI) and UsdPhysics.CollisionAPI(p).GetCollisionEnabledAttr().Get()]
  assert len(colliders)==1,colliders
 ctx.reset();objects.reset()
 states=[]
 for step in range(400):
  ctx.step(render=False);objects.update(.01)
  states.append(objects.data.object_state_w[0].cpu().numpy().copy())
 data=np.array(states);q=data[:,:,3:7].astype(np.float64)
 tilt=np.arccos(np.clip(1-2*(q[:,:,1]**2+q[:,:,2]**2),-1,1))
 rows=[]
 for i,c in enumerate(variants):
  rows.append(dict(asset=c.usd_path.split('/')[-1],fraction=c.size_fraction,max_tilt_deg=float(np.rad2deg(tilt[:,i]).max()),max_speed=float(np.linalg.norm(data[:,i,7:10],axis=-1).max()),max_angular_speed=float(np.linalg.norm(data[:,i,10:13],axis=-1).max())))
 report=dict(results=rows,coms=objects.root_physx_view.get_coms().cpu().tolist(),inertias=objects.root_physx_view.get_inertias().cpu().tolist())
 out=sys.argv[1] if len(sys.argv)>1 else 'outputs/object_random_validation/support_regression.json'
 Path(out).parent.mkdir(parents=True,exist_ok=True)
 with open(out,'w') as f:json.dump(report,f,indent=2)
 print(json.dumps(rows,indent=2),flush=True)
 assert max(r['max_tilt_deg'] for r in rows)<.1,'Uncommanded tilt exceeds 0.1 degrees on a flat support'
 assert max(r['max_speed'] for r in rows)<.005,'Uncommanded speed exceeds 5 mm/s on a flat support'
 print('SUPPORT STABILITY PASSED',flush=True)
except BaseException:
 traceback.print_exc();sys.stdout.flush();sys.stderr.flush();os._exit(1)
finally:app.close(skip_cleanup=True)
