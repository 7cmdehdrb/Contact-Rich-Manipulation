"""Independent Revision scene: eight available shapes, ten physical objects."""

from isaaclab.assets import AssetBaseCfg, RigidObjectCfg, RigidObjectCollectionCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass
import isaaclab.sim as sim

from ...mdp.common.contact_sweep_clutter_fusion_contacts import object_hand_sensor_name
from ..paths import SHELF_USD_PATH
from ..revision_spawners import (
    REVISION_OBJECT_NAMES, RevisionObjectSpawnerCfg, RevisionRobotSpawnerCfg, RevisionShelfSpawnerCfg,
)
from ..robot import HAND_CONTACT_BODY_NAMES, PALM_SENSOR_BODY_NAMES, make_robot_cfg
from ..tactile_regions import TACTILE_PARENT_BODY_NAMES, parent_sensor_name
from .contact_sweep_clutter_fusion import _tool_frame


def _robot():
    robot = make_robot_cfg()
    robot.spawn = RevisionRobotSpawnerCfg()
    return robot


@configclass
class RevisionSceneCfg(InteractiveSceneCfg):
    num_envs: int = 2048
    env_spacing: float = 2.5
    replicate_physics: bool = False
    ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim.GroundPlaneCfg())
    light = AssetBaseCfg(prim_path="/World/light", spawn=sim.DomeLightCfg(color=(.75, .75, .75), intensity=2500.))
    shelf = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Shelf",
        spawn=RevisionShelfSpawnerCfg(usd_path=SHELF_USD_PATH, mass_props=sim.MassPropertiesCfg(mass=100)),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-.7, 0., 0.)),
    )
    robot = _robot()
    object_collection = RigidObjectCollectionCfg(rigid_objects={})
    ee_frame = _tool_frame((("end_effector", (0., 0., .13 + robot.spawn.sensor_height / 2)),),
                           body="axia80_link", debug_vis=True)
    finger_frame = _tool_frame((("l_finger", (.07, 0., .13)), ("r_finger", (-.07, 0., .13))))
    wrist_frame = _tool_frame((("wrist", (0., 0., -.17)),))
    palm_tactile = None
    hand_tactile = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/(" + "|".join(PALM_SENSOR_BODY_NAMES) + ")",
        update_period=0., history_length=0, force_threshold=.01, debug_vis=False,
    )
    whole_hand_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/(" + "|".join(HAND_CONTACT_BODY_NAMES) + ")",
        update_period=0., history_length=0, force_threshold=.01, debug_vis=False,
    )

    def __post_init__(self):
        super().__post_init__()
        self.configure_population_capacity(4, 5, 16)

    def configure_population_capacity(self, max_gray_obstacles=4, max_white_obstacles=5, observation_slots=16):
        """The sixteen observation slots do not allocate sixteen physical bodies."""
        if (max_gray_obstacles, max_white_obstacles, observation_slots) != (4, 5, 16):
            raise ValueError("Revision capacity is Gray 4, White 5, observation slots 16")
        if self.object_collection.rigid_objects:
            if tuple(self.object_collection.rigid_objects) != REVISION_OBJECT_NAMES:
                raise ValueError("Unexpected Revision physical object roster")
            return
        objects = {}
        for index, name in enumerate(REVISION_OBJECT_NAMES):
            # Gravity is off before reset. Parking is above the ground plane and
            # separated by 2m vertically; no unfinished pose can fall or overlap.
            objects[name] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/" + name, spawn=RevisionObjectSpawnerCfg(),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0., 0., 50. + 2. * index)),
            )
        self.object_collection.rigid_objects = objects
        paths = [obj.prim_path for obj in objects.values()]
        for parent in TACTILE_PARENT_BODY_NAMES:
            setattr(self, parent_sensor_name(parent), ContactSensorCfg(
                prim_path="{ENV_REGEX_NS}/Robot/" + parent,
                filter_prim_paths_expr=paths.copy(), update_period=0., history_length=0,
                debug_vis=False, max_contact_data_count_per_prim=16 * len(paths),
            ))
        for name, obj in objects.items():
            setattr(self, object_hand_sensor_name(name), ContactSensorCfg(
                prim_path=obj.prim_path,
                filter_prim_paths_expr=["{ENV_REGEX_NS}/Robot/" + body for body in HAND_CONTACT_BODY_NAMES],
                update_period=0., history_length=0, debug_vis=False,
                max_contact_data_count_per_prim=16 * len(HAND_CONTACT_BODY_NAMES),
            ))


@configclass
class RevisionV1TestSceneCfg(RevisionSceneCfg):
    def __post_init__(self):
        super().__post_init__()
        self.object_collection.rigid_objects["mass_cylinder"].spawn.kind_override = "cylinder"
        # _add_temporary_eef_height_guides(self)  # TEMP disabled after height tuning.


# def _add_temporary_eef_height_guides(scene):
#     """TEMP visualization only: remove this helper and its call after height tuning."""
#     from ...mdp.common.contact_sweep_clutter_fusion import SHELF_HALF_EXTENTS, SHELF_SURFACE_HEIGHT
#
#     shelf_x, shelf_y, shelf_z = scene.shelf.init_state.pos
#     # Surface-relative center heights; colors ordered from bottom to top.
#     for height_mm, color in (
#         (90, (0.15, 0.45, 1.0)),
#         (95, (0.15, 1.0, 0.35)),
#         (100, (1.0, 0.65, 0.1)),
#     ):
#         setattr(scene, f"temp_eef_height_{height_mm}mm", AssetBaseCfg(
#             # Regex spawners require an existing parent; env prims already exist.
#             prim_path=f"{{ENV_REGEX_NS}}/TempEEFHeight_{height_mm}mm",
#             spawn=sim.CuboidCfg(
#                 size=(2 * SHELF_HALF_EXTENTS[0], 2 * SHELF_HALF_EXTENTS[1], 0.0005),
#                 collision_props=None,
#                 rigid_props=None,
#                 mass_props=None,
#                 activate_contact_sensors=False,
#                 visual_material=sim.PreviewSurfaceCfg(
#                     diffuse_color=color, opacity=0.90, roughness=1.0,
#                 ),
#             ),
#             init_state=AssetBaseCfg.InitialStateCfg(
#                 pos=(shelf_x, shelf_y, shelf_z + SHELF_SURFACE_HEIGHT + height_mm / 1000.0),
#             ),
#         ))


@configclass
class RevisionV2TestSceneCfg(RevisionSceneCfg):
    def __post_init__(self):
        super().__post_init__()
        self.object_collection.rigid_objects["mass_cylinder"].spawn.kind_override = "cylinder"
