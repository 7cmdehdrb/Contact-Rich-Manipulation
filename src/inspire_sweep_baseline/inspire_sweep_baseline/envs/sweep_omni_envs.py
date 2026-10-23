"""Baseline mechanics with realistic physics and goal-based omnidirectional rewards.

OmniV2 mechanics with DexTouch-style distance-record reward shaping."""

from ..assets.scenes.realistic_physics import apply_realistic_physics
from ..mdp.events.sweep_omni import OmniEventsCfg
from ..mdp.reward_functions.goal_rewards import HeightGatedPlanarReachProgress
from ..mdp.rewards.sweep_omni import DexOmniRewardsCfg
from ..mdp.rewards.sweep_omni import OmniRewardsCfg
from ..mdp.rewards.sweep_omni import OmniV2RewardsCfg
from ..mdp.rewards.sweep_omni import OmniV4RewardsCfg
from ..mdp.terminations.sweep_omni import OmniTerminationsCfg
from .sweep_envs import InspireSweepEnvCfg
from isaaclab.utils import configclass

@configclass
class InspireSweepOmniEnvCfg(InspireSweepEnvCfg):
    events: OmniEventsCfg = OmniEventsCfg()
    rewards: OmniRewardsCfg = OmniRewardsCfg()
    terminations: OmniTerminationsCfg = OmniTerminationsCfg()

    def __post_init__(self):
        super().__post_init__()
        apply_realistic_physics(self)
        # Baseline policy remains the live-object 41D observation; actions,
        # scene assets, initial joints, commands, timing and PPO are inherited.
        self.scene.object_collection.rigid_objects["mass_cylinder"].init_state.pos = (
            -0.7,
            0.0,
            1.05,
        )


@configclass
class InspireSweepOmniV1EnvCfg(InspireSweepOmniEnvCfg):
    """Omni variant with a 3 rad/s arm termination threshold."""

    def __post_init__(self):
        super().__post_init__()
        self.terminations.hand_velocity.params["threshold"] = 1.5


@configclass
class InspireSweepOmniV2EnvCfg(InspireSweepOmniV1EnvCfg):
    """Live-object approach and sweep progress without success termination."""

    rewards: OmniV2RewardsCfg = OmniV2RewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.terminations.success = None


@configclass
class InspireDexOmniEnvCfg(InspireSweepOmniV2EnvCfg):
    """Change only task shaping; inherit penalties, termination and spawning."""

    rewards: DexOmniRewardsCfg = DexOmniRewardsCfg()


@configclass
class InspireSweepOmniV3EnvCfg(InspireSweepOmniV2EnvCfg):
    """V2 with height-gated XY reaching; all other settings are inherited."""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.reaching.func = HeightGatedPlanarReachProgress
        self.rewards.reaching.params.update(z_offset=0.09, z_tolerance=0.02)


@configclass
class InspireSweepOmniV4EnvCfg(InspireSweepOmniV3EnvCfg):
    """V3 with relaxed failure limits and object/Arm excess-speed costs."""

    rewards: OmniV4RewardsCfg = OmniV4RewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        # Match the saved V3 run 2026-10-15_16-53-58_height_gated_xy_reach.
        self.terminations.hand_velocity.params["threshold"] = 5.0
        self.terminations.object_outside_shelf = None
        self.terminations.success = None
        self.terminations.push_fast.params["speed_condition"] = 0.5
