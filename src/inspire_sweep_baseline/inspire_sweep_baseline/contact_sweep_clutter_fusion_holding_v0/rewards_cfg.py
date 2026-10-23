"""Keep v1_0 terms except exact-goal progress, settling and impact warmup."""
from isaaclab.utils import configclass
from ..contact_sweep_clutter_fusion_v1_0.rewards_cfg import ContactSweepClutterFusionV1_0RewardsCfg
from .mdp import goal_settling


@configclass
class FusionHoldingV0RewardsCfg(ContactSweepClutterFusionV1_0RewardsCfg):
    goal_distance = ContactSweepClutterFusionV1_0RewardsCfg().goal_distance.replace(params={'position_tolerance':0.})
    goal_hold = ContactSweepClutterFusionV1_0RewardsCfg().goal_hold.replace(
        func=goal_settling,weight=2.,params={'position_scale':.01,'speed_scale':.05,'cutoff_multiple':3.})
    object_acceleration = ContactSweepClutterFusionV1_0RewardsCfg().object_acceleration.replace(
        weight=0.,params={'free_acceleration':50.,'max_cost':100.})
    obstacle_contact = ContactSweepClutterFusionV1_0RewardsCfg().obstacle_contact.replace(
        weight=0.,params={'free_force':80.,'max_cost':100.})
    wrench_change = ContactSweepClutterFusionV1_0RewardsCfg().wrench_change.replace(
        weight=0.,params={**ContactSweepClutterFusionV1_0RewardsCfg().wrench_change.params,
                          'free_force_change':50.,'free_moment_change':5.})
