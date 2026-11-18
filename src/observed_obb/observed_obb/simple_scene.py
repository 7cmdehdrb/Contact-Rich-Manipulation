"""Read-only reuse of Simple's assets and placement; no RL environment is built.

Import after AppLauncher. Source bounds are already scaled by the original
spawner; the asset-root offset must be rotated before forming an observed box.
"""
import math
import random
from time import perf_counter

import numpy as np
import torch
from pxr import Gf, Usd, UsdGeom, UsdPhysics

from inspire_sweep_baseline.assets.scenes.sweep_simple import SweepSimpleSceneCfg
from inspire_sweep_baseline.assets import revision_spawners
from inspire_sweep_baseline.mdp.common.wrench_refactor_layout import CircleResetLayout
from inspire_sweep_baseline.mdp.events.sweep_simple import SweepSimpleEventsCfg

from .observation import Box
from .placement import sample_inspection_layout


class SimpleScene:
    """A single frozen source scene with the original fixed 17-body asset pool."""

    def __init__(self, stage, seed: int = 0):
        start = perf_counter()
        self.stage = stage
        config = SweepSimpleSceneCfg(num_envs=1)
        self.shelf_position = tuple(config.shelf.init_state.pos)
        if tuple(config.shelf.init_state.rot) != (1., 0., 0., 0.):
            raise ValueError("Simple inspection expects the current upright, unrotated shelf")
        self.paths = tuple(f"/World/envs/env_0/{name}" for name in config.object_collection.rigid_objects)
        UsdGeom.Xform.Define(stage, "/World/envs/env_0")
        shelf = config.shelf.spawn
        shelf.func("/World/envs/env_0/Shelf", shelf, translation=self.shelf_position)
        # Keep the source's independent scale draws, without altering callers'
        # global RNG state. New placement does not redraw immutable geometry.
        previous_random_state = random.getstate()
        random.seed(seed)
        try:
            for path, source in zip(self.paths, config.object_collection.rigid_objects.values()):
                source.spawn.func(path, source.spawn, translation=source.init_state.pos)
        finally:
            random.setstate(previous_random_state)
        metadata = revision_spawners._prototype_cache["metadata"]
        self.bounds = np.asarray([metadata[path][2] for path in self.paths], dtype=np.float64)
        self.bounds.setflags(write=False)
        # Disable dynamics on the source scene. Markers never acquire a physics
        # schema at all; this also keeps objects stationary if timeline is played.
        for prim in Usd.PrimRange(stage.GetPrimAtPath("/World/envs/env_0")):
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                UsdPhysics.RigidBodyAPI(prim).CreateRigidBodyEnabledAttr(False)
        bounds_t = torch.tensor(self.bounds[None], dtype=torch.float32)
        self.layout = CircleResetLayout(bounds_t, 1.05 - bounds_t[..., 0, 2])
        self.generator = torch.Generator(device="cpu").manual_seed(seed)
        params = SweepSimpleEventsCfg().object_spawn.params
        self.command_params = {key: params[key] for key in (
            "spawn_radius", "edge_clearance", "excluded_x_half_angle", "min_goal_distance", "max_goal_distance")}
        self.active_indices: tuple[int, ...] = ()
        self.startup_seconds = perf_counter() - start

    def new_layout(self) -> tuple[Box, ...]:
        local, yaw, active, _, _ = sample_inspection_layout(
            self.layout, self.generator, self.command_params)
        self.active_indices = tuple(active[0].nonzero().flatten().tolist())
        boxes = []
        for index, path in enumerate(self.paths):
            root = self.stage.GetPrimAtPath(path)
            position = local[0, index].numpy().astype(float) + self.shelf_position
            angle = float(yaw[0, index])
            root.GetAttribute("xformOp:translate").Set(Gf.Vec3d(*position))
            root.GetAttribute("xformOp:orient").Set(Gf.Quatf(math.cos(angle/2), 0, 0, math.sin(angle/2)))
            imageable = UsdGeom.Imageable(root)
            if index not in self.active_indices:
                imageable.MakeInvisible()
                continue
            imageable.MakeVisible()
            low, high = self.bounds[index]
            offset = (low + high) / 2
            c, s = math.cos(angle), math.sin(angle)
            rotated = (c*offset[0] - s*offset[1], s*offset[0] + c*offset[1], offset[2])
            boxes.append(Box(tuple(position + rotated), tuple(high-low), angle))
        return tuple(boxes)
