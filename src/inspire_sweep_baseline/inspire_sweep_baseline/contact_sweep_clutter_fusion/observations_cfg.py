"""Independent 55D policy, 40D initial obstacles, and 3D privileged target."""

from isaaclab.managers import ObservationGroupCfg as ObsGroup, ObservationTermCfg as ObsTerm
from isaaclab.utils import configclass
from isaaclab.utils.noise import ConstantNoiseCfg, NoiseModelWithAdditiveBiasCfg, UniformNoiseCfg

from ..contact_sweep_clutter.mdp import initial_obstacle_positions
from ..contact_sweep_clutter_v1.mdp import SharedTactileContactBits
from ..contact_sweep_clutter_v2.observations_cfg import PureWristWrench
from ..mdp import observations as obs
from ..mdp.asymmetric_observations import InitialObjectPositionInRoot
from .common import arm_entity
from .observations import initial_goal_polar_in_root


@configclass
class ContactSweepClutterFusionObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        # Arm 6+6, action 8, initial target 3, EEF 7, goal 2, tactile 17, F/T 6.
        joint_pos = ObsTerm(func=obs.arm_joint_pos_rel, params={"asset_cfg": arm_entity()})
        joint_vel = ObsTerm(func=obs.arm_joint_vel_rel, params={"asset_cfg": arm_entity()})
        actions = ObsTerm(func=obs.last_action)
        target_obs_state = ObsTerm(
            func=InitialObjectPositionInRoot, params={"command_name": "target_goal_pos"},
            noise=NoiseModelWithAdditiveBiasCfg(
                noise_cfg=ConstantNoiseCfg(bias=0.0),
                bias_noise_cfg=UniformNoiseCfg(n_min=-0.01, n_max=0.01, operation="abs"),
                sample_bias_per_component=True,
            ),
        )
        ee_pose = ObsTerm(func=obs.eef_pose_in_root)
        goal_pos = ObsTerm(func=initial_goal_polar_in_root, params={"command_name": "target_goal_pos"})
        tactile = ObsTerm(
            func=SharedTactileContactBits,
            params={"sensor_name": "hand_tactile", "threshold_n": 0.01},
        )
        wrist_wrench = ObsTerm(func=PureWristWrench)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class ObstaclesCfg(ObsGroup):
        initial_positions = ObsTerm(func=initial_obstacle_positions)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class PrivilegedCfg(ObsGroup):
        target_position = ObsTerm(func=obs.object_position_in_root)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    obstacles: ObstaclesCfg = ObstaclesCfg()
    privileged: PrivilegedCfg = PrivilegedCfg()
