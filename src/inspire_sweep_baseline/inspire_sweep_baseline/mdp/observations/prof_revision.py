from ..observation_functions.prof_revision import target_obb_in_eef, obstacle_obbs_in_eef, finger_poses_in_eef
from ..observation_functions.sweep_simple import raw_previous_action, hand_drive_positions
from isaaclab.managers import ObservationGroupCfg as ObsGroup, ObservationTermCfg as ObsTerm, SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import NoiseCfg
from ...assets.robot import ARM_JOINT_NAMES
from ...assets.hand_kinematics import HAND_DRIVE_JOINT_NAMES
from ..observation_functions.contact_sweep_clutter import PureWristWrench
from ..observation_functions.final import wrist_wrench_observation_noise
from ..observation_functions.revision import arm_joint_positions, arm_joint_velocities
from ..observation_functions.sweep import eef_pose_in_root
from ..observation_functions.wrench_refactor import RelativeRightHandTactile, goal_position_in_eef, paired_tactile_observation_noise

@configclass
class ProfRevisionObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        # 108D: arm[0:12], EEF[12:19], hand[19:25], fingers[25:60],
        # OBB xyz/size/xyzw[60:70], goal[70:73], actions[73:85], tactile[85:102], wrench[102:108].
        joint_pos = ObsTerm(func=arm_joint_positions, params={'asset_cfg': SceneEntityCfg('robot', joint_names=list(ARM_JOINT_NAMES), preserve_order=True)})
        joint_vel = ObsTerm(func=arm_joint_velocities, params={'asset_cfg': SceneEntityCfg('robot', joint_names=list(ARM_JOINT_NAMES), preserve_order=True)})
        ee_pose = ObsTerm(func=eef_pose_in_root)
        hand_configuration = ObsTerm(func=hand_drive_positions, params={
            'asset_cfg': SceneEntityCfg('robot', joint_names=list(HAND_DRIVE_JOINT_NAMES), preserve_order=True),
        })
        finger_poses = ObsTerm(func=finger_poses_in_eef)
        target_obs_state = ObsTerm(func=target_obb_in_eef)
        goal_pos = ObsTerm(func=goal_position_in_eef, params={'command_name': 'target_goal_pos'})
        actions = ObsTerm(func=raw_previous_action)
        tactile = ObsTerm(func=RelativeRightHandTactile, params={'sensor_name': 'hand_tactile', 'threshold_n': 0.05}, noise=NoiseCfg(func=paired_tactile_observation_noise))
        wrist_wrench = ObsTerm(func=PureWristWrench, noise=NoiseCfg(func=wrist_wrench_observation_noise))


        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class ObstaclesCfg(ObsGroup):
        initial_positions = ObsTerm(func=obstacle_obbs_in_eef)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    obstacles: ObstaclesCfg = ObstaclesCfg()
    critic = None
    critic_obstacles = None
