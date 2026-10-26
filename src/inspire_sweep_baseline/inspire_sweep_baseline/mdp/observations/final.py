"""Independent 55D policy, 64D initial obstacles, and 3D privileged target."""


from ..common.contact_sweep_clutter_fusion import arm_entity
from ..observation_functions.asymmetric_observations import InitialObjectPositionInRoot
from ..observation_functions.contact_sweep_clutter import PureWristWrench
from ..observation_functions.contact_sweep_clutter import SharedTactileContactBits
from ..observation_functions.contact_sweep_clutter import initial_obstacle_positions
from ..observation_functions.contact_sweep_clutter_fusion import initial_goal_polar_in_root
from ..observation_functions.contact_sweep_clutter_fusion import bounded_last_action
from ..observation_functions.sweep import arm_joint_pos_rel
from ..observation_functions.sweep import arm_joint_vel_rel
from ..observation_functions.sweep import eef_pose_in_root
from ..observation_functions.sweep import object_position_in_root
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils import configclass
from isaaclab.utils.noise import ConstantNoiseCfg
from isaaclab.utils.noise import NoiseModelWithAdditiveBiasCfg
from isaaclab.utils.noise import UniformNoiseCfg


@configclass
class FinalObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        # Arm 6+6, action 8, initial target 3, EEF 7, goal 2, tactile 17, F/T 6.
        joint_pos = ObsTerm(func=arm_joint_pos_rel, params={"asset_cfg": arm_entity()})
        joint_vel = ObsTerm(func=arm_joint_vel_rel, params={"asset_cfg": arm_entity()})
        actions = ObsTerm(func=bounded_last_action)
        target_obs_state = ObsTerm(
            func=InitialObjectPositionInRoot, params={"command_name": "target_goal_pos"},
            noise=NoiseModelWithAdditiveBiasCfg(
                noise_cfg=ConstantNoiseCfg(bias=0.0),
                bias_noise_cfg=UniformNoiseCfg(n_min=-0.01, n_max=0.01, operation="abs"),
                sample_bias_per_component=True,
            ),
        )
        ee_pose = ObsTerm(func=eef_pose_in_root)
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
        target_position = ObsTerm(func=object_position_in_root)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    obstacles: ObstaclesCfg = ObstaclesCfg()
    privileged: PrivilegedCfg = PrivilegedCfg()


from ...assets.robot import HAND_JOINT_NAMES
from isaaclab.managers import SceneEntityCfg
from ..observation_functions.final import hand_configuration, previous_action_command
from ..observation_functions.final import tactile_observation_noise, wrist_wrench_observation_noise
from isaaclab.utils.noise import NoiseCfg


@configclass
class FinalV1ObservationsCfg(FinalObservationsCfg):
    @configclass
    class PolicyCfg(FinalObservationsCfg.PolicyCfg):
        actions = ObsTerm(func=previous_action_command)
        tactile = ObsTerm(
            func=SharedTactileContactBits,
            params={'sensor_name': 'hand_tactile', 'threshold_n': 0.01},
            noise=NoiseCfg(func=tactile_observation_noise),
        )
        wrist_wrench = ObsTerm(func=PureWristWrench, noise=NoiseCfg(func=wrist_wrench_observation_noise))
        hand_configuration = ObsTerm(
            func=hand_configuration,
            params={'asset_cfg': SceneEntityCfg('robot', joint_names=list(HAND_JOINT_NAMES), preserve_order=True)},
        )

    policy: PolicyCfg = PolicyCfg()
