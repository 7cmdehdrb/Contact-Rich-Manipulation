"""Self-contained baseline configuration, derived from the v2-mass behavior.

Baseline mechanics with one independent binary observation per Hand pad."""

from ...mdp.common.sweep import TOOL_ROTATION_H
from ..mass_cylinder import MassCylinderCfg
from ..paths import SHELF_USD_PATH
from ..robot import PALM_SENSOR_BODY_NAMES
from ..robot import make_robot_cfg
from ..tactile_regions import TACTILE_PARENT_BODY_NAMES
from ..tactile_regions import parent_sensor_name
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
class SweepSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=2500.0),
    )
    shelf = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Shelf",
        spawn=sim_utils.UsdFileCfg(usd_path=SHELF_USD_PATH, mass_props=sim_utils.MassPropertiesCfg(mass=100)),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-0.7, 0.0, 0.0)),
    )
    robot = make_robot_cfg().replace(prim_path="{ENV_REGEX_NS}/Robot")
    object_collection = RigidObjectCollectionCfg(rigid_objects={
        "mass_cylinder": RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/mass_cylinder",
            init_state=RigidObjectCfg.InitialStateCfg(pos=(-0.75, 0.0, 1.05)),
            spawn=MassCylinderCfg(
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
    palm_tactile = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/inspire_palm_force_sensor",
        update_period=0.0, history_length=1, force_threshold=0.05, debug_vis=False,
    )


# Exact existing rigid-body names; do not include parent Hand or Arm colliders.
TACTILE_BODY_REGEX = "(" + "|".join(PALM_SENSOR_BODY_NAMES) + ")"


@configclass
class TactileSweepSceneCfg(SweepSceneCfg):
    # Replace the single-pad reader rather than maintaining duplicate sensors.
    palm_tactile = None
    hand_tactile = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/" + TACTILE_BODY_REGEX,
        update_period=0.0, history_length=0, force_threshold=0.05, debug_vis=False,
    )

    def __post_init__(self):
        super().__post_init__()
        # Keep frame transforms for observations, without creating the optional
        # Fabric PointInstancer that warns about mismatched debug prototypes.
        self.ee_frame.debug_vis = False
        # Individual parent bodies satisfy ContactSensor's one-to-many filter
        # contract. Read every contact point, never their averaged position.
        for parent in TACTILE_PARENT_BODY_NAMES:
            setattr(self, parent_sensor_name(parent), ContactSensorCfg(
                prim_path="{ENV_REGEX_NS}/Robot/" + parent,
                # Imported Shelf colliders and the static ground are not valid
                # detailed-contact GPU filters on the supported backend. Keep
                # all 17 raw pad contacts unfiltered; project object contacts
                # only, using the package's known primitive cylinder collider.
                filter_prim_paths_expr=["{ENV_REGEX_NS}/mass_cylinder"],
                update_period=0.0, history_length=0, debug_vis=False,
                max_contact_data_count_per_prim=16,
            ))
