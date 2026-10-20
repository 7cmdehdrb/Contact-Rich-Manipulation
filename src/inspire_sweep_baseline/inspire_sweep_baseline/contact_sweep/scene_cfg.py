"""Robot, object and all tactile views, without environment Cfg inheritance."""

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg, RigidObjectCollectionCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.utils import configclass

from ..assets.paths import SHELF_USD_PATH
from ..assets.realistic_spawners import RealisticCylinderSpawnerCfg, RealisticShelfSpawnerCfg
from ..assets.robot import PALM_SENSOR_BODY_NAMES, make_robot_cfg
from ..assets.tactile_regions import TACTILE_PARENT_BODY_NAMES, parent_sensor_name
from .common import OBJECT_MASS, TOOL_ROTATION_H


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
class ContactSweepSceneCfg(InteractiveSceneCfg):
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
    # Sensor housing center is shifted +height/2 from the original hand mount.
    # Hand base is +height: this TCP is exactly hand-local (0, 0, 0.13).
    ee_frame = _tool_frame(
        (("end_effector", (0.0, 0.0, 0.13 + robot.spawn.sensor_height / 2)),),
        body="axia80_link", debug_vis=True,
    )
    # Virtual source finger/wrist points used by shelf penalties/terminations.
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
        for parent in TACTILE_PARENT_BODY_NAMES:
            setattr(self, parent_sensor_name(parent), ContactSensorCfg(
                prim_path="{ENV_REGEX_NS}/Robot/" + parent,
                filter_prim_paths_expr=["{ENV_REGEX_NS}/mass_cylinder"],
                update_period=0.0, history_length=0, debug_vis=False,
                max_contact_data_count_per_prim=16,
            ))
