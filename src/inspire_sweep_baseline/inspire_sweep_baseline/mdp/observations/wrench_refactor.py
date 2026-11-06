"""Equal 74D raw Actor/Critic; target snapshot replaced by live XYZ in Critic."""

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import ConstantNoiseCfg, NoiseCfg, NoiseModelWithAdditiveBiasCfg, UniformNoiseCfg

from ...assets.robot import ARM_JOINT_NAMES, HAND_JOINT_NAMES
from ..observation_functions.asymmetric_observations import InitialObjectPositionInRoot
from ..observation_functions.contact_sweep_clutter import PureWristWrench
from ..observation_functions.wrench_refactor import RightHandTactile
from ..observation_functions.contact_sweep_clutter_fusion import initial_goal_polar_in_root
from ..observation_functions.final import (
    hand_configuration,
    previous_action_command,
    tactile_observation_noise,
    wrist_wrench_observation_noise,
)
from ..observation_functions.revision import (
    arm_joint_positions,
    arm_joint_velocities,
    current_obstacle_positions,
    initial_obstacle_positions,
)
from ..observation_functions.sweep import eef_pose_in_root, object_position_in_root


@configclass
class WrenchRefactorObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        # [0:12] actual arm, [12:14] measured hand, [14:21] EEF pose,
        # [21:24] initial target, [24:26] goal, [26:34] previous action,
        # [34:68] front/back tactile bits, [68:74] compensated F/T wrench.
        joint_pos = ObsTerm(
            func=arm_joint_positions,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=list(ARM_JOINT_NAMES), preserve_order=True)},
        )
        joint_vel = ObsTerm(
            func=arm_joint_velocities,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=list(ARM_JOINT_NAMES), preserve_order=True)},
        )
        hand_configuration = ObsTerm(
            func=hand_configuration,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=list(HAND_JOINT_NAMES), preserve_order=True)},
        )
        ee_pose = ObsTerm(func=eef_pose_in_root)
        target_obs_state = ObsTerm(
            func=InitialObjectPositionInRoot,
            params={"command_name": "target_goal_pos"},
            noise=NoiseModelWithAdditiveBiasCfg(
                noise_cfg=ConstantNoiseCfg(bias=0.0),
                bias_noise_cfg=UniformNoiseCfg(n_min=-0.01, n_max=0.01, operation="abs"),
                sample_bias_per_component=True,
            ),
        )
        goal_pos = ObsTerm(func=initial_goal_polar_in_root, params={"command_name": "target_goal_pos"})
        actions = ObsTerm(func=previous_action_command)
        tactile = ObsTerm(
            func=RightHandTactile,
            params={"sensor_name": "hand_tactile", "threshold_n": 0.05},
            noise=NoiseCfg(func=tactile_observation_noise),
        )
        wrist_wrench = ObsTerm(func=PureWristWrench, noise=NoiseCfg(func=wrist_wrench_observation_noise))

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        # [0:12] actual arm, [12:14] measured hand, [14:21] EEF pose,
        # [21:24] initial target, [24:26] goal, [26:34] previous action,
        # [34:68] front/back tactile bits, [68:74] compensated F/T wrench.
        joint_pos = ObsTerm(
            func=arm_joint_positions,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=list(ARM_JOINT_NAMES), preserve_order=True)},
        )
        joint_vel = ObsTerm(
            func=arm_joint_velocities,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=list(ARM_JOINT_NAMES), preserve_order=True)},
        )
        hand_configuration = ObsTerm(
            func=hand_configuration,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=list(HAND_JOINT_NAMES), preserve_order=True)},
        )
        ee_pose = ObsTerm(func=eef_pose_in_root)
        target_obs_state = ObsTerm(func=object_position_in_root)
        goal_pos = ObsTerm(func=initial_goal_polar_in_root, params={"command_name": "target_goal_pos"})
        actions = ObsTerm(func=previous_action_command)
        tactile = ObsTerm(
            func=RightHandTactile,
            params={"sensor_name": "hand_tactile", "threshold_n": 0.05},
            noise=NoiseCfg(func=tactile_observation_noise),
        )
        wrist_wrench = ObsTerm(func=PureWristWrench, noise=NoiseCfg(func=wrist_wrench_observation_noise))

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
    class CriticObstaclesCfg(ObsGroup):
        current_positions = ObsTerm(func=current_obstacle_positions, params={"adjacent_swap_probability": 0.0})

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    obstacles: ObstaclesCfg = ObstaclesCfg()
    critic: CriticCfg = CriticCfg()
    critic_obstacles: CriticObstaclesCfg = CriticObstaclesCfg()


@configclass
class WrenchRefactorRelativeObservationsCfg(WrenchRefactorObservationsCfg):
    """One shared 58D policy + 64D obstacle sample for Actor and Critic."""

    @configclass
    class PolicyCfg(WrenchRefactorObservationsCfg.PolicyCfg):
        def __post_init__(self):
            super().__post_init__()
            from ..observation_functions.wrench_refactor import (
                InitialObjectPositionInEEF, RelativeRightHandTactile,
                goal_position_in_eef, paired_tactile_observation_noise,
            )
            self.target_obs_state.func = InitialObjectPositionInEEF
            self.target_obs_state.params['coordinate_noise'] = .01
            # Apply the parent's reset-fixed visual bias before frame conversion.
            self.target_obs_state.noise = None
            self.goal_pos.func = goal_position_in_eef
            self.tactile.func = RelativeRightHandTactile
            self.tactile.noise = NoiseCfg(func=paired_tactile_observation_noise)

    @configclass
    class ObstaclesCfg(WrenchRefactorObservationsCfg.ObstaclesCfg):
        def __post_init__(self):
            super().__post_init__()
            from ..observation_functions.wrench_refactor import initial_obstacle_positions_in_eef
            self.initial_positions.func = initial_obstacle_positions_in_eef

    policy: PolicyCfg = PolicyCfg()
    obstacles: ObstaclesCfg = ObstaclesCfg()
    critic = None
    critic_obstacles = None
