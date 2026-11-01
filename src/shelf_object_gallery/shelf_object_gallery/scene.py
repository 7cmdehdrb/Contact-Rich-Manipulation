"""Isaac Lab scene assembled from the active Final environment's spawners.

Import after AppLauncher. No robot, task registration, RL managers or policy.
"""

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from inspire_sweep_baseline.assets.contact_sweep_clutter_fusion_holding_osc_object_random_spawners import ASSETS, ShelfPropCfg
from inspire_sweep_baseline.assets.final_spawners import FinalShelfObjectCfg, shelf_object_meshes
from inspire_sweep_baseline.assets.mass_cylinder import MassCylinderCfg
from inspire_sweep_baseline.assets.paths import SHELF_USD_PATH
from inspire_sweep_baseline.assets.realistic_spawners import RealisticCylinderSpawnerCfg, RealisticShelfSpawnerCfg
from inspire_sweep_baseline.mdp.common.contact_sweep_clutter_fusion import (
    OBJECT_MASS, SHELF_HALF_EXTENTS, SHELF_SURFACE_HEIGHT,
)

from .layout import KINDS, grid_layout


@configclass
class GallerySceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/Ground", spawn=sim_utils.GroundPlaneCfg())
    light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=2500.0, color=(0.9, 0.9, 0.9)),
    )


def object_spawner(kind):
    """Use the Final spawners, including fitted/textured YCB assets at fraction 1."""
    common = dict(
        mass_props=sim_utils.MassPropertiesCfg(mass=OBJECT_MASS),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
        collision_props=sim_utils.CollisionPropertiesCfg(
            collision_enabled=True, contact_offset=0.001, rest_offset=0.0),
        activate_contact_sensors=False,
    )
    if kind in ASSETS:
        return ShelfPropCfg(
            **common, usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/YCB/Axis_Aligned/{kind}.usd",
            size_fraction=1.0, collision_kind="box" if kind.endswith("_box") else "can")
    colors = {"cylinder": (0.1, 0.4, 0.8), "cube": (0.7, 0.55, 0.3),
              "mug": (0.3, 0.65, 0.75), "bottle": (0.45, 0.7, 0.4)}
    if kind == "cylinder":
        return RealisticCylinderSpawnerCfg(
            **common, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=colors[kind]))
    return FinalShelfObjectCfg(**common, kind=kind, cube_size=(0.06, 0.08, 0.10), color=colors[kind])


def object_bounds():
    """Measure the source mesh vertices, rather than copying mug/bottle sizes."""
    cylinder = MassCylinderCfg()
    bounds = {"cylinder": ((-cylinder.radius, -cylinder.radius, 0.0),
                           (cylinder.radius, cylinder.radius, cylinder.height))}
    for kind in KINDS[1:4]:
        pieces = shelf_object_meshes(kind, cube_size=(0.06, 0.08, 0.10))
        vertices = [point for points, _, _ in pieces for point in points]
        bounds[kind] = (tuple(min(p[axis] for p in vertices) for axis in range(3)),
                        tuple(max(p[axis] for p in vertices) for axis in range(3)))
    # Measure the fitted source spawner output, so dimensions are not hardcoded
    # to one downloaded YCB version. Remove these temporary prims before PhysX starts.
    from pxr import Usd, UsdGeom
    stage = sim_utils.get_current_stage()
    for kind in ASSETS:
        path = "/World/GalleryBounds_" + kind
        if stage.GetPrimAtPath(path):
            raise RuntimeError(f"Temporary measurement path already exists: {path}")
        try:
            spawn = object_spawner(kind)
            prim = spawn.func(path, spawn)
            box = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"]).ComputeUntransformedBound(prim).ComputeAlignedRange()
            bounds[kind] = (tuple(box.GetMin()), tuple(box.GetMax()))
        finally:
            stage.RemovePrim(path)
    return bounds


class ShelfObjectGallery:
    """One static display scene, stepped through the Isaac Lab simulator."""

    def __init__(self, *, rows=2, cols=4, gap=0.05, shelf_usd=None):
        self.bounds = object_bounds()
        self.placements = grid_layout(
            rows, cols, self.bounds, gap=gap,
            shelf_half_extents=SHELF_HALF_EXTENTS, surface_height=SHELF_SURFACE_HEIGHT,
        )
        self.shelf_usd = shelf_usd or SHELF_USD_PATH
        cfg = GallerySceneCfg(num_envs=1, env_spacing=2.5, replicate_physics=False)
        cfg.shelf = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/Shelf",
            spawn=RealisticShelfSpawnerCfg(
                usd_path=self.shelf_usd,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
                mass_props=sim_utils.MassPropertiesCfg(mass=100),
            ),
            init_state=AssetBaseCfg.InitialStateCfg(pos=(-0.7, 0.0, 0.0)),
        )
        for i, item in enumerate(self.placements):
            spawn = object_spawner(item.kind)
            name = f"object_{i:03d}_{item.kind}"
            x, y, z = item.position
            setattr(cfg, name, RigidObjectCfg(
                # Source procedural spawner takes a concrete path (not a regex).
                prim_path="/World/envs/env_0/" + name, spawn=spawn,
                init_state=RigidObjectCfg.InitialStateCfg(pos=(x - 0.7, y, z)),
            ))
        self.scene = InteractiveScene(cfg)

    def step(self, sim):
        self.scene.write_data_to_sim()
        sim.step()
        self.scene.update(sim.get_physics_dt())

    def validate(self):
        """Check a live stage for exact bounds, static bodies and no robot."""
        import numpy as np
        from pxr import Usd, UsdGeom, UsdPhysics
        stage = sim_utils.get_current_stage()
        assert not stage.GetPrimAtPath("/World/envs/env_0/Robot")
        cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"])
        for i, item in enumerate(self.placements):
            name = f"object_{i:03d}_{item.kind}"
            prim = stage.GetPrimAtPath("/World/envs/env_0/" + name)
            box = cache.ComputeWorldBound(prim).ComputeAlignedRange()
            low, high = np.array(box.GetMin()), np.array(box.GetMax())
            expected_size = np.subtract(self.bounds[item.kind][1], self.bounds[item.kind][0])
            np.testing.assert_allclose(high - low, expected_size, atol=1e-5)
            assert abs(low[2] - SHELF_SURFACE_HEIGHT) < 1e-5
            assert low[0] >= -0.7 - SHELF_HALF_EXTENTS[0] + 0.02 - 1e-5
            assert high[0] <= -0.7 + SHELF_HALF_EXTENTS[0] - 0.02 + 1e-5
            assert low[1] >= -SHELF_HALF_EXTENTS[1] + 0.02 - 1e-5
            assert high[1] <= SHELF_HALF_EXTENTS[1] - 0.02 + 1e-5
            assert UsdPhysics.RigidBodyAPI(prim).GetKinematicEnabledAttr().Get()
            if item.kind in ASSETS:
                assert prim.GetCustomDataByKey("object_random_asset").endswith(item.kind + ".usd")
                assert stage.GetPrimAtPath(str(prim.GetPath()) + "/Collision")
                meshes = [p for p in Usd.PrimRange(prim) if p.IsA(UsdGeom.Mesh)
                          and "/Geometry/Upright/Asset/" in str(p.GetPath())]
                assert meshes, f"YCB visual meshes missing: {item.kind}"
            assert not stage.GetPrimAtPath("/World/GalleryBounds_" + item.kind)
            position = self.scene[name].data.root_pos_w[0].cpu().numpy()
            np.testing.assert_allclose(position, (item.position[0] - 0.7, *item.position[1:]), atol=1e-5)
