"""Baseline mechanics with one independent binary observation per Hand pad."""

from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass

from .assets.robot import PALM_SENSOR_BODY_NAMES
from .assets.tactile_regions import TACTILE_PARENT_BODY_NAMES, parent_sensor_name
from .env_cfg import InspireSweepEnvCfg, ObservationsCfg, SweepSceneCfg
from .mdp.tactile_observations import ProjectedTactileContactBits


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


@configclass
class TactileObservationsCfg(ObservationsCfg):
    @configclass
    class PolicyCfg(ObservationsCfg.PolicyCfg):
        tactile = ObsTerm(
            func=ProjectedTactileContactBits, params={"sensor_name": "hand_tactile", "threshold_n": 0.05},
        )

    policy: PolicyCfg = PolicyCfg()


@configclass
class InspireSweepTactileEnvCfg(InspireSweepEnvCfg):
    scene: TactileSweepSceneCfg = TactileSweepSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: TactileObservationsCfg = TactileObservationsCfg()
