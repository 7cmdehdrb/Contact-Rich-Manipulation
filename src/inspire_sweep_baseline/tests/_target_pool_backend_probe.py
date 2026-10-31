"""Check GPU actor identity across simulation-disable toggles before pool integration."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from isaaclab.app import AppLauncher
from cli_args import apply_renderer_compatibility
parser = argparse.ArgumentParser()
parser.add_argument('--disable-simulation', action='store_true')
parser.add_argument('--toggle-collisions', action='store_true')
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
apply_renderer_compatibility(args)
app = AppLauncher(args).app
try:
    import torch
    import isaaclab.sim as sim
    from isaaclab.assets import RigidObjectCfg, RigidObjectCollection, RigidObjectCollectionCfg
    ctx = sim.SimulationContext(sim.SimulationCfg(dt=.01, device=args.device))
    for i in range(2):
        sim.create_prim(f'/World/envs/env_{i}', 'Xform')
    cfg = RigidObjectCollectionCfg(rigid_objects={
        f'candidate_{j}': RigidObjectCfg(
            prim_path=f'/World/envs/env_.*/candidate_{j}',
            spawn=sim.CuboidCfg(size=(.1,.1,.1),
                rigid_props=sim.RigidBodyPropertiesCfg(disable_gravity=True),
                mass_props=sim.MassPropertiesCfg(mass=1),
                collision_props=sim.CollisionPropertiesCfg()),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(j*2.,0,2))) for j in range(3)})
    objects = RigidObjectCollection(cfg)
    ctx.reset()
    print('PROBE_SIM_RESET', flush=True)
    view = objects.root_physx_view
    from pxr import UsdPhysics
    stage = sim.get_current_stage()
    for k in range(12):
        print('PROBE_TOGGLE', k, flush=True)
        flags = torch.ones(6,1,dtype=torch.uint8)
        flags[k%3*2,0] = 0
        flags[(k+1)%3*2+1,0] = 0
        if args.disable_simulation:
            view.set_disable_simulations(flags, torch.arange(6))
        elif args.toggle_collisions:
            for j in range(3):
                for i in range(2):
                    root = stage.GetPrimAtPath(f'/World/envs/env_{i}/candidate_{j}')
                    for prim in __import__('pxr.Usd', fromlist=['PrimRange']).PrimRange(root):
                        if prim.HasAPI(UsdPhysics.CollisionAPI):
                            UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Set(not bool(flags[j*2+i]))
        else:
            view.set_disable_gravities(torch.ones_like(flags), torch.arange(6))
        state = objects.data.default_object_state.clone()
        state[1,:,1] += 10.0
        state[:,:,0] += k * .01
        objects.write_object_link_state_to_sim(state)
        ctx.step()
        objects.update(.01)
        torch.testing.assert_close(objects.data.object_pos_w, state[:,:,:3])
    print('TARGET_POOL_BACKEND_PASS', flush=True)
except BaseException:
    import traceback
    traceback.print_exc()
    raise
finally:
    app.close()
