"""Inherit v1.0 penalties and increase only the requested positive weights."""

from isaaclab.utils import configclass

from ..contact_sweep_clutter_fusion_v1_0.rewards_cfg import ContactSweepClutterFusionV1_0RewardsCfg
from .mdp import scheduled_goal_hold_xy


@configclass
class ContactSweepClutterFusionV1_4RewardsCfg(ContactSweepClutterFusionV1_0RewardsCfg):
    goal_hold = ContactSweepClutterFusionV1_0RewardsCfg().goal_hold.replace(
        func=scheduled_goal_hold_xy, weight=9.0, params={"position_tolerance": 0.0},
    )
    hand_contact = ContactSweepClutterFusionV1_0RewardsCfg().hand_contact.replace(weight=1.45)
