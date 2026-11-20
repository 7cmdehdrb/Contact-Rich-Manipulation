"""Robot-free studio scene using the active Sweep Only shelf and object spawners."""

import isaaclab.sim as sim
from isaaclab.assets import AssetBaseCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

from ..paths import SHELF_USD_PATH
from ..wrench_refactor_spawners import WrenchRefactorShelfSpawnerCfg


@configclass
class ShelfPhotographySceneCfg(InteractiveSceneCfg):
    num_envs: int = 1
    env_spacing: float = 2.5
    replicate_physics: bool = False
    ground = AssetBaseCfg(prim_path="/World/Ground", spawn=sim.GroundPlaneCfg())
    light = AssetBaseCfg(prim_path="/World/Light", spawn=sim.DomeLightCfg(
        intensity=1000., color=(.9, .9, .9)))
    backdrop = AssetBaseCfg(
        prim_path="/World/Backdrop", spawn=sim.CuboidCfg(size=(.02, 4., 4.),
            visual_material=sim.PreviewSurfaceCfg(diffuse_color=(.18, .20, .23), roughness=1.)),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(-1.3, 0., 1.2)),
    )
    shelf = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Shelf",
        spawn=WrenchRefactorShelfSpawnerCfg(usd_path=SHELF_USD_PATH,
            rigid_props=sim.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True)),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(-.7, 0., 0.)),
    )
