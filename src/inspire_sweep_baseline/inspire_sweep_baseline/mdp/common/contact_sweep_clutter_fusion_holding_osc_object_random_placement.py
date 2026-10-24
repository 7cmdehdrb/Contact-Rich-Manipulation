"""Keep parent zone proposals, reject overlap using actual oriented bounds."""

from ...assets.scenes.contact_sweep_clutter_layout import GenerationError
from ...assets.scenes.contact_sweep_clutter_layout import generator_cm_to_shelf_xy
from ...assets.scenes.contact_sweep_clutter_layout import shelf_xy_to_generator_cm
from ...assets.scenes.contact_sweep_clutter_layout import ClutterLayoutGenerator, scene_generator
from dataclasses import replace
import math
import numpy as np
import torch as torch


class GrayPaddingLayoutGenerator(ClutterLayoutGenerator):
    """Admit Gray centers up to a fraction of the original Red radius inward.

    White keeps the original Red/Gray classification. Outer Gray bounds, rear
    quota, target/object separation and approach certificates stay unchanged.
    """

    def __init__(self, *args, gray_red_padding_fraction=0.0, **kwargs):
        super().__init__(*args, **kwargs)
        if not math.isfinite(gray_red_padding_fraction) or not 0 <= gray_red_padding_fraction <= 0.15:
            raise ValueError("Gray padding fraction must be in [0, 0.15]")
        radius = 2 * self.config.radius + self.config.red_margin
        self._gray_zone_config = replace(
            self.config, red_margin=radius * (1 - gray_red_padding_fraction) - 2 * self.config.radius,
        )
        self._gray_zone_config.check()

    def zone_masks(self, points, target, direction, distance, role):
        config = self.config if role == "white" else self._gray_zone_config
        return scene_generator.zone_masks(points, target, direction, distance, config)


def padded_population_layout_generator(env, gray, white, padding):
    """Cache only the current padding stage; retain a seeded RNG per population."""
    seed = getattr(env, "_population_layout_seed", env.cfg.seed)
    stage = (padding, seed)
    if getattr(env, "_gray_padding_stage", None) != stage:
        env._gray_padding_stage = stage
        env._gray_padding_layout_generators = {}
    key = (gray, white)
    cache = env._gray_padding_layout_generators
    if key not in cache:
        from ...assets.scenes.contact_sweep_clutter_layout import DEFAULT_LAYOUT_CONFIG
        config = replace(DEFAULT_LAYOUT_CONFIG, obstacles=gray + white,
                         gray_obstacles=gray, rear_obstacles=min(1, gray))
        cache[key] = GrayPaddingLayoutGenerator(
            seed=seed, config=config,
            max_obstacles=env.cfg.obstacle_observation_slots, gray_red_padding_fraction=padding,
        )
    return cache[key]

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
    from pxr import Usd
    from pxr import UsdGeom
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
