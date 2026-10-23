"""Preserve Holding failures and speed/time requirements with a live radius."""

from isaaclab.utils import configclass

from ..contact_sweep_clutter_fusion_holding_v0.terminations_cfg import FusionHoldingV0TerminationsCfg
from .mdp import ScheduledHoldingSuccess


@configclass
class FusionHoldingV2TerminationsCfg(FusionHoldingV0TerminationsCfg):
    success = FusionHoldingV0TerminationsCfg().success.replace(
        func=ScheduledHoldingSuccess,
        params={key: value for key, value in FusionHoldingV0TerminationsCfg().success.params.items()
                if key != "position_tolerance"},
    )
