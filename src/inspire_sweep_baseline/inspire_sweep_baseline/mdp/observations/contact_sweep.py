"""Explicit 57D policy observation order, shared by Actor and Critic."""

from ..common.contact_sweep import arm_entity
from ..observation_functions.sweep import arm_joint_pos_rel
from ..observation_functions.sweep import arm_joint_vel_rel
from ..observation_functions.sweep import eef_pose_in_root
from ..observation_functions.sweep import goal_position_in_root
from ..observation_functions.sweep import last_action
from ..observation_functions.sweep import object_position_in_root
from ..observation_functions.sweep import object_width
from ..observation_functions.sweep import wrist_wrench_eef
from ..observation_functions.tactile_observations import ProjectedTactileContactBits
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

@configclass
class ContactSweepObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        # Arm 6+6, last action 8, object 3+1, EEF 7, goal 3, tactile 17, F/T 6.
        joint_pos = ObsTerm(func=arm_joint_pos_rel, params={"asset_cfg": arm_entity()})
        joint_vel = ObsTerm(func=arm_joint_vel_rel, params={"asset_cfg": arm_entity()})
        actions = ObsTerm(func=last_action)
        target_obs_state = ObsTerm(
            func=object_position_in_root, noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        target_obj_width = ObsTerm(func=object_width, noise=Unoise(n_min=-0.01, n_max=0.01))
        ee_pose = ObsTerm(func=eef_pose_in_root)
        goal_pos = ObsTerm(func=goal_position_in_root, params={"command_name": "target_goal_pos"})
        tactile = ObsTerm(
            func=ProjectedTactileContactBits,
            params={"sensor_name": "hand_tactile", "threshold_n": 0.01},
        )
        wrist_wrench = ObsTerm(func=wrist_wrench_eef)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
