"""Holding v1 with the same robot-base OSC configuration as OSC v0."""

from isaaclab.utils import configclass

from ..contact_sweep_clutter_fusion_holding_v1.env_cfg import FusionHoldingV1EnvCfg
from ..contact_sweep_clutter_fusion_holding_osc_v0.env_cfg import configure_robot_base_osc


@configclass
class FusionHoldingOSCV1EnvCfg(FusionHoldingV1EnvCfg):
    """Preserve Holding v1 managers and replace only the arm controller."""

    def __post_init__(self):
        super().__post_init__()
        configure_robot_base_osc(self)
