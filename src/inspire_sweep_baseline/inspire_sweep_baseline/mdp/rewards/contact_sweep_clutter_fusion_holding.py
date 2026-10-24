"""Keep v1_0 terms except exact-goal progress, settling and impact warmup.

Double settling weight and grant contact credit inside its shared radius."""

from ..reward_functions.contact_sweep_clutter_fusion_holding import goal_settling
from ..reward_functions.contact_sweep_clutter_fusion_holding_scheduled import goal_settling as goal_settling_contact_sweep_clutter_fusion_holding_scheduled
from ..reward_functions.contact_sweep_clutter_fusion_holding_scheduled import hand_contact_or_goal
from ..reward_functions.contact_sweep_clutter_fusion_holding_scheduled import CappedGoalSettling
from .contact_sweep_clutter_fusion import ContactSweepClutterFusionV1_0RewardsCfg
from isaaclab.utils import configclass

@configclass
class FusionHoldingV0RewardsCfg(ContactSweepClutterFusionV1_0RewardsCfg):
    hand_contact = ContactSweepClutterFusionV1_0RewardsCfg().hand_contact.replace(weight=1.45)
    object = ContactSweepClutterFusionV1_0RewardsCfg().object.replace(
        params={**ContactSweepClutterFusionV1_0RewardsCfg().object.params,
                'displacement_ratio': 5.0})
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


@configclass
class FusionHoldingV2RewardsCfg(FusionHoldingV0RewardsCfg):
    goal_hold = FusionHoldingV0RewardsCfg().goal_hold.replace(
        func=goal_settling_contact_sweep_clutter_fusion_holding_scheduled, weight=4.0,
        params={"position_tolerance": 0.02, "speed_scale": 0.05},
    )
    hand_contact = FusionHoldingV0RewardsCfg().hand_contact.replace(
        func=hand_contact_or_goal, params={"threshold_n": 0.01},
    )


@configclass
class FusionHoldingV3RewardsCfg(FusionHoldingV2RewardsCfg):
    goal_hold = FusionHoldingV2RewardsCfg().goal_hold.replace(
        func=CappedGoalSettling,
        params={**FusionHoldingV2RewardsCfg().goal_hold.params, "max_hold_time": 3.0},
    )
