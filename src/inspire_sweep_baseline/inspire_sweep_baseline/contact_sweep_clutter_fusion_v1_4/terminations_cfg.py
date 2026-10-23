"""Retain failures and add success after the existing failure terms."""

from isaaclab.managers import TerminationTermCfg
from isaaclab.utils import configclass

from ..contact_sweep_clutter_fusion_v1_0.terminations_cfg import ContactSweepClutterFusionV1_0TerminationsCfg
from .mdp import GoalHoldSuccess


@configclass
class ContactSweepClutterFusionV1_4TerminationsCfg(ContactSweepClutterFusionV1_0TerminationsCfg):
    push_fast = ContactSweepClutterFusionV1_0TerminationsCfg().push_fast.replace(
        params={"speed_condition": 1.0},
    )
    success = TerminationTermCfg(
        func=GoalHoldSuccess,
        params={"hold_time": 3.0,
                "failure_terms": ("object_drop", "push_fast", "shelf_collision",
                                  "hand_velocity", "shelf_displacement")},
    )
