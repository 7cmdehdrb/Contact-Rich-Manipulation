"""v1_0 failures plus fixed accurate, stationary success."""
from isaaclab.managers import TerminationTermCfg
from isaaclab.utils import configclass
from ..contact_sweep_clutter_fusion_v1_0.terminations_cfg import ContactSweepClutterFusionV1_0TerminationsCfg
from .mdp import HoldingSuccess


@configclass
class FusionHoldingV0TerminationsCfg(ContactSweepClutterFusionV1_0TerminationsCfg):
    push_fast = ContactSweepClutterFusionV1_0TerminationsCfg().push_fast.replace(
        params={'speed_condition': 1.0})
    success = TerminationTermCfg(func=HoldingSuccess,params={
        'position_tolerance':.005,'speed_tolerance':.01,'hold_time':3.,
        'failure_terms':('object_drop','push_fast','shelf_collision','hand_velocity','shelf_displacement')})
