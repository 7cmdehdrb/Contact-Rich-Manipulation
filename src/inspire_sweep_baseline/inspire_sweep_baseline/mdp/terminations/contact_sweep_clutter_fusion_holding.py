"""v1_0 failures plus fixed accurate, stationary success.

Preserve Holding failures and speed/time requirements with a live radius."""

from ..termination_functions.contact_sweep_clutter_fusion_holding import HoldingSuccess
from ..termination_functions.contact_sweep_clutter_fusion_holding import ScheduledHoldingSuccess
from .contact_sweep_clutter_fusion import ContactSweepClutterFusionV1_0TerminationsCfg
from isaaclab.managers import TerminationTermCfg
from isaaclab.utils import configclass

@configclass
class FusionHoldingV0TerminationsCfg(ContactSweepClutterFusionV1_0TerminationsCfg):
    push_fast = ContactSweepClutterFusionV1_0TerminationsCfg().push_fast.replace(
        params={'speed_condition': 1.0})
    success = TerminationTermCfg(func=HoldingSuccess,params={
        'position_tolerance':.005,'speed_tolerance':.01,'hold_time':3.,
        'failure_terms':('object_drop','push_fast','shelf_collision','hand_velocity','shelf_displacement')})


@configclass
class FusionHoldingV2TerminationsCfg(FusionHoldingV0TerminationsCfg):
    success = FusionHoldingV0TerminationsCfg().success.replace(
        func=ScheduledHoldingSuccess,
        params={key: value for key, value in FusionHoldingV0TerminationsCfg().success.params.items()
                if key != "position_tolerance"},
    )


@configclass
class FusionHoldingV4TerminationsCfg(FusionHoldingV2TerminationsCfg):
    # Holding lasts until timeout unless an inherited failure ends the episode.
    success = None
