"""Shared initial-only 58D policy and 64D obstacles for Actor and Critic."""
from isaaclab.managers import ObservationGroupCfg as ObsGroup, ObservationTermCfg as ObsTerm, SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import NoiseCfg
from ...assets.robot import ARM_JOINT_NAMES, HAND_JOINT_NAMES
from ..observation_functions.contact_sweep_clutter import PureWristWrench
from ..observation_functions.final import hand_configuration, previous_action_command, wrist_wrench_observation_noise
from ..observation_functions.revision import arm_joint_positions, arm_joint_velocities
from ..observation_functions.sweep import eef_pose_in_root
from ..observation_functions.wrench_refactor import (InitialObjectPositionInEEF, RelativeRightHandTactile,
    goal_position_in_eef, paired_tactile_observation_noise, initial_obstacle_positions_in_eef)


@configclass
class SweepSimpleObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(func=arm_joint_positions, params={'asset_cfg': SceneEntityCfg('robot', joint_names=list(ARM_JOINT_NAMES), preserve_order=True)})
        joint_vel = ObsTerm(func=arm_joint_velocities, params={'asset_cfg': SceneEntityCfg('robot', joint_names=list(ARM_JOINT_NAMES), preserve_order=True)})
        hand_configuration = ObsTerm(func=hand_configuration, params={'asset_cfg': SceneEntityCfg('robot', joint_names=list(HAND_JOINT_NAMES), preserve_order=True)})
        ee_pose = ObsTerm(func=eef_pose_in_root)
        target_obs_state = ObsTerm(func=InitialObjectPositionInEEF, params={'command_name': 'target_goal_pos', 'coordinate_noise': 0.01})
        goal_pos = ObsTerm(func=goal_position_in_eef, params={'command_name': 'target_goal_pos'})
        actions = ObsTerm(func=previous_action_command)
        tactile = ObsTerm(func=RelativeRightHandTactile, params={'sensor_name': 'hand_tactile', 'threshold_n': 0.05}, noise=NoiseCfg(func=paired_tactile_observation_noise))
        wrist_wrench = ObsTerm(func=PureWristWrench, noise=NoiseCfg(func=wrist_wrench_observation_noise))

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class ObstaclesCfg(ObsGroup):
        initial_positions = ObsTerm(func=initial_obstacle_positions_in_eef)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    obstacles: ObstaclesCfg = ObstaclesCfg()
    critic = None
    critic_obstacles = None
