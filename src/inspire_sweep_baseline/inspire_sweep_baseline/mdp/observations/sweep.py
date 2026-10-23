"""Self-contained baseline configuration, derived from the v2-mass behavior.

Realistic physics with an initial-position Actor and privileged live Critic.

Baseline mechanics with one independent binary observation per Hand pad."""

from ..common.sweep import _arm
from ..observation_functions.asymmetric_observations import live_object_state_in_root
from ..observation_functions.sweep import arm_joint_pos_rel
from ..observation_functions.sweep import arm_joint_vel_rel
from ..observation_functions.sweep import eef_pose_in_root
from ..observation_functions.sweep import goal_position_in_root
from ..observation_functions.sweep import last_action
from ..observation_functions.sweep import object_position_in_root
from ..observation_functions.sweep import object_width
from ..observation_functions.sweep import palm_contact_bit
from ..observation_functions.sweep import wrist_wrench_eef
from ..observation_functions.tactile_observations import ProjectedTactileContactBits
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        # Preserve the baseline observation order: 34 + palm1 + wrench6 = 41.
        joint_pos = ObsTerm(func=arm_joint_pos_rel, params={"asset_cfg": _arm()})
        joint_vel = ObsTerm(func=arm_joint_vel_rel, params={"asset_cfg": _arm()})
        actions = ObsTerm(func=last_action)
        target_obs_state = ObsTerm(
            func=object_position_in_root, noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        target_obj_width = ObsTerm(func=object_width, noise=Unoise(n_min=-0.01, n_max=0.01))
        ee_pose = ObsTerm(func=eef_pose_in_root)
        goal_pos = ObsTerm(func=goal_position_in_root, params={"command_name": "target_goal_pos"})
        tactile = ObsTerm(func=palm_contact_bit)
        wrist_wrench = ObsTerm(func=wrist_wrench_eef)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class InitialPrivilegedObservationsCfg(ObservationsCfg):
    @configclass
    class PrivilegedCfg(ObsGroup):
        target_state = ObsTerm(func=live_object_state_in_root)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    privileged: PrivilegedCfg = PrivilegedCfg()


@configclass
class TactileObservationsCfg(ObservationsCfg):
    @configclass
    class PolicyCfg(ObservationsCfg.PolicyCfg):
        tactile = ObsTerm(
            func=ProjectedTactileContactBits, params={"sensor_name": "hand_tactile", "threshold_n": 0.05},
        )

    policy: PolicyCfg = PolicyCfg()
