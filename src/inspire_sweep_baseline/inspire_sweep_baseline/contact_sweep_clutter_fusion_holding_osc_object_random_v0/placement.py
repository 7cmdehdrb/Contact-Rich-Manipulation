"""Keep parent zone proposals, reject overlap using actual oriented bounds."""
import numpy as np
import torch
from ..contact_sweep_clutter.layout import GenerationError
from ..contact_sweep_clutter_fusion_v1_0.layout import shelf_xy_to_generator_cm, generator_cm_to_shelf_xy


def valid_layout(centers, yaw, half, gap=.002):
    axes=np.stack((np.stack((np.cos(yaw),np.sin(yaw)),-1),
                   np.stack((-np.sin(yaw),np.cos(yaw)),-1)),axis=1)
    world_half=np.einsum('nij,ni->nj',np.abs(axes),half)
    if np.any(np.abs(centers)+world_half+gap>np.array([.20,.50])):
        return False
    for i in range(len(centers)):
        for j in range(i):
            directions=np.concatenate((axes[i],axes[j]))
            distance=np.abs(directions@(centers[i]-centers[j]))
            extent_i=np.abs(directions@axes[i].T)@half[i]
            extent_j=np.abs(directions@axes[j].T)@half[j]
            if not np.any(distance>=extent_i+extent_j+gap):
                return False
    return True


def sample_layout(env, ids, generator, target, goal, gray, white):
    from pxr import Usd, UsdGeom
    import isaaclab.sim as sim
    objects=env.scene['object_collection']
    if not hasattr(env,'_prop_half_extents'):
        stage=sim.get_current_stage();bounds=[]
        cache=UsdGeom.BBoxCache(Usd.TimeCode.Default(),['default','render'])
        for env_id in range(env.num_envs):
            row=[]
            for name in objects.object_names:
                root=stage.GetPrimAtPath(f'/World/envs/env_{env_id}/{name}')
                if name == 'mass_cylinder':
                    row.append([.04, .04])  # Conservative square around the unchanged target cylinder.
                    continue
                box=cache.ComputeRelativeBound(stage.GetPrimAtPath(str(root.GetPath())+'/Geometry'),root).ComputeAlignedRange()
                row.append([max(abs(box.GetMin()[k]),abs(box.GetMax()[k])) for k in range(2)])
            bounds.append(row)
        env._prop_half_extents=np.asarray(bounds)
    active_indices=[0,*range(1,1+gray),*range(1+env.cfg.max_gray_obstacles,1+env.cfg.max_gray_obstacles+white)]
    targets=target.cpu().numpy();goals=goal.cpu().numpy()
    selected=[];all_yaw=[]
    for row,env_id in enumerate(ids.cpu().tolist()):
        for attempt in range(200):
            try:
                proposal=generator.generate(shelf_xy_to_generator_cm(targets[row:row+1]),shelf_xy_to_generator_cm(goals[row:row+1]))
            except GenerationError:
                continue  # Retry only obstacles; never resample the target or goal.
            obstacles=generator_cm_to_shelf_xy(proposal)[0]
            yaw=generator.rng.uniform(-np.pi,np.pi,objects.num_objects)
            yaw[0] = 0.0  # Cylinder is rotationally symmetric; keep its OBB conservative.
            centers=np.concatenate((targets[row:row+1],obstacles))
            if valid_layout(centers,yaw[active_indices],env._prop_half_extents[env_id,active_indices]):
                selected.append(obstacles);all_yaw.append(yaw);break
        else:
            raise RuntimeError(f'No non-overlapping enlarged layout for env {env_id} after 200 attempts; target/goal unchanged')
    return (torch.as_tensor(np.asarray(selected),device=env.device,dtype=target.dtype),
            torch.as_tensor(np.asarray(all_yaw),device=env.device,dtype=target.dtype))
