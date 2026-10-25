"""Holding v0 with relative OSC commands in the robot base frame.

Holding v1 with the same robot-base OSC configuration as OSC v0.

Change only Holding v2 arm control to the Reach OSC configuration."""

from ..mdp.common.contact_sweep_clutter_fusion_holding_osc_configuration import configure_robot_base_osc
from ..mdp.common.contact_sweep_clutter_fusion_holding_osc_configuration import configure_reach_osc
from .contact_sweep_clutter_fusion_holding_envs import FusionHoldingV0EnvCfg
from .contact_sweep_clutter_fusion_holding_envs import FusionHoldingV1EnvCfg
from .contact_sweep_clutter_fusion_holding_envs import FusionHoldingV2EnvCfg
from .contact_sweep_clutter_fusion_holding_envs import FusionHoldingV3EnvCfg
from .contact_sweep_clutter_fusion_holding_envs import FusionHoldingV4EnvCfg
from isaaclab.utils import configclass

@configclass
class FusionHoldingOSCV0EnvCfg(FusionHoldingV0EnvCfg):
    """Replace only Arm action and the competing joint PD gains."""

    def __post_init__(self):
        super().__post_init__()
        configure_robot_base_osc(self)


@configclass
class FusionHoldingOSCV1EnvCfg(FusionHoldingV1EnvCfg):
    """Preserve Holding v1 managers and replace only the arm controller."""

    def __post_init__(self):
        super().__post_init__()
        configure_robot_base_osc(self)


@configclass
class FusionHoldingOSCV2EnvCfg(FusionHoldingV2EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        configure_reach_osc(self)


@configclass
class FusionHoldingOSCV3EnvCfg(FusionHoldingV3EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        configure_reach_osc(self)


@configclass
class FusionHoldingOSCV4EnvCfg(FusionHoldingV4EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        configure_reach_osc(self)
