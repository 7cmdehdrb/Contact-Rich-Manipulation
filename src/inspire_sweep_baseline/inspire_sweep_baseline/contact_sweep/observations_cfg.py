"""Explicit 57D policy observation order, shared by Actor and Critic."""

from isaaclab.managers import ObservationGroupCfg as ObsGroup, ObservationTermCfg as ObsTerm
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from ..mdp import observations as obs
from ..mdp.tactile_observations import ProjectedTactileContactBits
from .common import arm_entity


@configclass
class ContactSweepObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        # Arm 6+6, last action 8, object 3+1, EEF 7, goal 3, tactile 17, F/T 6.
        joint_pos = ObsTerm(func=obs.arm_joint_pos_rel, params={"asset_cfg": arm_entity()})
        joint_vel = ObsTerm(func=obs.arm_joint_vel_rel, params={"asset_cfg": arm_entity()})
        actions = ObsTerm(func=obs.last_action)
        target_obs_state = ObsTerm(
            func=obs.object_position_in_root, noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        target_obj_width = ObsTerm(func=obs.object_width, noise=Unoise(n_min=-0.01, n_max=0.01))
        ee_pose = ObsTerm(func=obs.eef_pose_in_root)
        goal_pos = ObsTerm(func=obs.goal_position_in_root, params={"command_name": "target_goal_pos"})
        tactile = ObsTerm(
            func=ProjectedTactileContactBits,
            params={"sensor_name": "hand_tactile", "threshold_n": 0.01},
        )
        wrist_wrench = ObsTerm(func=obs.wrist_wrench_eef)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
