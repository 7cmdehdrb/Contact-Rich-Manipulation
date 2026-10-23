"""Explicit robot, four cylinders and tactile sensors without task inheritance.

Add whole-hand and object-specific contacts while preserving tactile input."""

from ...mdp.common.contact_sweep_clutter_fusion import OBJECT_MASS
from ...mdp.common.contact_sweep_clutter_fusion import TOOL_ROTATION_H
from ...mdp.common.contact_sweep_clutter_fusion_contacts import object_hand_sensor_name
from ...mdp.common.contact_sweep_clutter_fusion_population import PARKING_HEIGHT
from ...mdp.common.contact_sweep_clutter_fusion_population import validate_capacity
from ..paths import SHELF_USD_PATH
from ..realistic_spawners import RealisticCylinderSpawnerCfg
from ..realistic_spawners import RealisticShelfSpawnerCfg
from ..robot import HAND_CONTACT_BODY_NAMES
from ..robot import PALM_SENSOR_BODY_NAMES
from ..robot import make_robot_cfg
from ..tactile_regions import TACTILE_PARENT_BODY_NAMES
from ..tactile_regions import parent_sensor_name
from .contact_sweep_clutter_layout import DEFAULT_LAYOUT_CONFIG
from isaaclab.assets import AssetBaseCfg
from isaaclab.assets import RigidObjectCfg
from isaaclab.assets import RigidObjectCollectionCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.utils import configclass
import isaaclab.sim as sim_utils

def _tool_frame(targets, body="inspire_base_link", debug_vis=False):
    markers = FRAME_MARKER_CFG.copy()
    markers.markers["frame"].scale = (0.05, 0.05, 0.05)
    markers.prim_path = f"/Visuals/InspireSweepBaseline/{targets[0][0]}"
    return FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_link",
        debug_vis=debug_vis,
        visualizer_cfg=markers,
        target_frames=[FrameTransformerCfg.FrameCfg(
            prim_path=f"{{ENV_REGEX_NS}}/Robot/{body}", name=name,
            offset=OffsetCfg(pos=position, rot=TOOL_ROTATION_H),
        ) for name, position in targets],
    )


@configclass
class ContactSweepClutterFusionSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=2500.0),
    )
    shelf = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Shelf",
        spawn=RealisticShelfSpawnerCfg(usd_path=SHELF_USD_PATH, mass_props=sim_utils.MassPropertiesCfg(mass=100)),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-0.7, 0.0, 0.0)),
    )
    robot = make_robot_cfg().replace(prim_path="{ENV_REGEX_NS}/Robot")
    object_collection = RigidObjectCollectionCfg(rigid_objects={
        "mass_cylinder": RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/mass_cylinder",
            init_state=RigidObjectCfg.InitialStateCfg(pos=(-0.7, 0.0, 1.05)),
            spawn=RealisticCylinderSpawnerCfg(
                mass_props=sim_utils.MassPropertiesCfg(mass=OBJECT_MASS),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16, solver_velocity_iteration_count=1,
                    max_angular_velocity=1000.0, max_linear_velocity=1000.0,
                    max_depenetration_velocity=5.0, disable_gravity=False,
                ),
                collision_props=sim_utils.CollisionPropertiesCfg(
                    collision_enabled=True, contact_offset=0.001, rest_offset=0.0,
                ),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.4, 0.8)),
            ),
        ),
    })
    ee_frame = _tool_frame(
        (("end_effector", (0.0, 0.0, 0.13 + robot.spawn.sensor_height / 2)),),
        body="axia80_link", debug_vis=True,
    )
    finger_frame = _tool_frame((
        ("l_finger", (0.07, 0.0, 0.13)), ("r_finger", (-0.07, 0.0, 0.13)),
    ))
    wrist_frame = _tool_frame((("wrist", (0.0, 0.0, -0.17)),))
    palm_tactile = None
    hand_tactile = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/(" + "|".join(PALM_SENSOR_BODY_NAMES) + ")",
        update_period=0.0, history_length=0, force_threshold=0.01, debug_vis=False,
    )

    def __post_init__(self):
        super().__post_init__()
        target = self.object_collection.rigid_objects["mass_cylinder"]
        for index in range(DEFAULT_LAYOUT_CONFIG.obstacles):
            obstacle = target.copy()
            obstacle.prim_path = f"{{ENV_REGEX_NS}}/obstacle_{index}"
            obstacle.spawn.visual_material = sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.85, 0.35, 0.10),
            )
            obstacle.init_state.pos = (
                -0.82 + 0.12 * (index % 3), -0.30 if index < 3 else 0.30, 1.05,
            )
            self.object_collection.rigid_objects[f"obstacle_{index}"] = obstacle
        paths = [obj.prim_path for obj in self.object_collection.rigid_objects.values()]
        for parent in TACTILE_PARENT_BODY_NAMES:
            setattr(self, parent_sensor_name(parent), ContactSensorCfg(
                prim_path="{ENV_REGEX_NS}/Robot/" + parent,
                filter_prim_paths_expr=paths.copy(),
                update_period=0.0, history_length=0, debug_vis=False,
                max_contact_data_count_per_prim=16 * len(paths),
            ))


@configclass
class ContactSweepClutterFusionV1_0SceneCfg(ContactSweepClutterFusionSceneCfg):
    whole_hand_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/(" + "|".join(HAND_CONTACT_BODY_NAMES) + ")",
        update_period=0.0, history_length=0, force_threshold=0.01, debug_vis=False,
    )

    def __post_init__(self):
        super().__post_init__()
        self.configure_population_capacity(max_gray_obstacles=3, max_white_obstacles=3)

    def configure_population_capacity(self, max_gray_obstacles, max_white_obstacles, observation_slots=10):
        """Build fixed gray/white slots before simulation starts.

        Counts live on EnvCfg because InteractiveScene treats scalar scene
        fields as assets. The runtime calls this again after Hydra overrides.
        """
        validate_capacity(max_gray_obstacles, max_white_obstacles, observation_slots)
        for name in tuple(vars(self)):
            if name.startswith("hand_contact_"):
                delattr(self, name)
        target = self.object_collection.rigid_objects["mass_cylinder"].copy()
        objects = {"mass_cylinder": target}
        for index in range(max_gray_obstacles + max_white_obstacles):
            obstacle = target.copy()
            obstacle.prim_path = f"{{ENV_REGEX_NS}}/obstacle_{index}"
            # Slots remain stable for the entire run: gray capacity first,
            # then white capacity, irrespective of the current active counts.
            color = (0.45, 0.45, 0.45) if index < max_gray_obstacles else (0.90, 0.90, 0.90)
            obstacle.spawn.visual_material = sim_utils.PreviewSurfaceCfg(diffuse_color=color)
            obstacle.init_state.pos = (-0.7, 0.2 * index, PARKING_HEIGHT)
            objects[f"obstacle_{index}"] = obstacle
        self.object_collection.rigid_objects = objects
        paths = [obj.prim_path for obj in objects.values()]
        for parent in TACTILE_PARENT_BODY_NAMES:
            tactile = getattr(self, parent_sensor_name(parent))
            tactile.filter_prim_paths_expr = paths.copy()
            tactile.max_contact_data_count_per_prim = 16 * len(paths)
        # Reverse the contact query: each cylinder is ONE reporting rigid body
        # filtered against all hand bodies. One view per object suffices, rather than a
        # separate filtered sensor for every hand link/pad. Explicit paths select
        # the 13 hand links plus all 17 tactile pads, excluding the arm and Axia80
        # housing. Existing tactile sensors retain their own filters.
        for name, obj in self.object_collection.rigid_objects.items():
            obj.spawn.activate_contact_sensors = True
            setattr(self, object_hand_sensor_name(name), ContactSensorCfg(
                prim_path=obj.prim_path,
                filter_prim_paths_expr=["{ENV_REGEX_NS}/Robot/" + body for body in HAND_CONTACT_BODY_NAMES],
                update_period=0.0, history_length=0, debug_vis=False,
                max_contact_data_count_per_prim=16 * len(HAND_CONTACT_BODY_NAMES),
            ))
