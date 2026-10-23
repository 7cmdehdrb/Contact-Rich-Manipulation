"""Renewal composition with shared exponential excess-speed penalties.

Renewal mechanics with Dex distance rewards and exponential speed costs.

Renewal task with relative, fixed-impedance OSC at the existing EEF.

Renewal variants with 17 tactile observations and shared exponential costs.

Dex Tactile v1: bounded speed costs, lower wrist cutoff and lateral goals.

Dex Tactile v1 with goal occupancy rewards continuing until timeout.

Dex Tactile v2 with bounded relative OSC; retain all task mechanics.

Dex Tactile v1 with heavier objects and empirically balanced progress rewards."""

from ..assets.robot import ARM_JOINT_NAMES
from ..assets.scenes.realistic_physics import apply_realistic_physics
from ..assets.scenes.sweep_omni_renewal import TactileOmniRenewalSceneCfg
from ..mdp.actions.bounded_osc import BoundedOperationalSpaceControllerActionCfg
from ..mdp.event_functions.realistic_physics import randomize_scaled_episode_physics
from ..mdp.events.sweep_omni_renewal import OmniRenewalEventsCfg
from ..mdp.observations.sweep_omni_renewal import TactileOmniRenewalObservationsCfg
from ..mdp.reward_functions.goal_rewards import exponential_joint_speed_cost_limit
from ..mdp.reward_functions.goal_rewards import exponential_pushing_speed_cost_limit
from ..mdp.rewards.sweep_omni_renewal import DexOmniRenewalRewardsCfg
from ..mdp.rewards.sweep_omni_renewal import DexTactileOmniRenewalV2RewardsCfg
from ..mdp.rewards.sweep_omni_renewal import OmniRenewalRewardsCfg
from ..mdp.terminations.sweep_omni_renewal import DexTactileOmniRenewalV2TerminationsCfg
from ..mdp.terminations.sweep_omni_renewal import OmniRenewalTerminationsCfg
from .sweep_envs import InspireSweepEnvCfg
from isaaclab.controllers import OperationalSpaceControllerCfg
from isaaclab.envs.mdp.actions import OperationalSpaceControllerActionCfg
from isaaclab.utils import configclass
import math as math

    # success = None


@configclass
class InspireSweepOmniRenewalEnvCfg(InspireSweepEnvCfg):
    """Compose Renewal mechanics and exponential speed costs on the baseline."""

    events: OmniRenewalEventsCfg = OmniRenewalEventsCfg()
    rewards: OmniRenewalRewardsCfg = OmniRenewalRewardsCfg()
    terminations: OmniRenewalTerminationsCfg = OmniRenewalTerminationsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.commands.target_goal_pos.debug_vis = True
        apply_realistic_physics(self)
        self.scene.object_collection.rigid_objects["mass_cylinder"].init_state.pos = (
            -0.7,
            0.0,
            1.05,
        )


@configclass
class InspireDexSweepOmniRenewalEnvCfg(InspireSweepOmniRenewalEnvCfg):
    rewards: DexOmniRenewalRewardsCfg = DexOmniRenewalRewardsCfg()


@configclass
class InspireSweepOmniRenewalOSCEnvCfg(InspireSweepOmniRenewalEnvCfg):
    """OSC Arm control and exponential speed costs; inherit Renewal terminations."""

    def __post_init__(self):
        super().__post_init__()
        eef = self.scene.ee_frame.target_frames[0]
        self.actions.arm_action = OperationalSpaceControllerActionCfg(
            asset_name="robot",
            joint_names=list(ARM_JOINT_NAMES),
            body_name=eef.prim_path.rsplit("/", 1)[-1],
            body_offset=OperationalSpaceControllerActionCfg.OffsetCfg(
                pos=eef.offset.pos,
                rot=eef.offset.rot,
            ),
            controller_cfg=OperationalSpaceControllerCfg(
                target_types=["pose_rel"],
                impedance_mode="fixed",
                motion_stiffness_task=200.0,
                motion_damping_ratio_task=1.0,
                inertial_dynamics_decoupling=True,
                gravity_compensation=True,
            ),
            position_scale=1.0,
            orientation_scale=1.0,
        )
        # OSC supplies joint efforts; implicit joint PD must not compete with it.
        self.scene.robot.actuators["arm"].stiffness = 0.0
        self.scene.robot.actuators["arm"].damping = 0.0


@configclass
class InspireTactileSweepOmniRenewalEnvCfg(InspireSweepOmniRenewalEnvCfg):
    """Sweep Renewal with 17 binary tactile channels instead of one (57D)."""

    scene: TactileOmniRenewalSceneCfg = TactileOmniRenewalSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: TactileOmniRenewalObservationsCfg = TactileOmniRenewalObservationsCfg()


@configclass
class InspireDexTactileSweepOmniRenewalEnvCfg(InspireDexSweepOmniRenewalEnvCfg):
    """Dex Sweep Renewal with the same 17 tactile channels and original rewards."""

    scene: TactileOmniRenewalSceneCfg = TactileOmniRenewalSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: TactileOmniRenewalObservationsCfg = TactileOmniRenewalObservationsCfg()


@configclass
class InspireDexTactileSweepOmniRenewalV1EnvCfg(InspireDexTactileSweepOmniRenewalEnvCfg):
    """Inherit v0's 57D/8D contract and specialize only the requested terms."""

    def __post_init__(self):
        super().__post_init__()
        # PhysX joint speed limits in rad/s, below the 5 rad/s failure cutoff.
        self.scene.robot.actuators["arm"].velocity_limit_sim = dict(
            zip(ARM_JOINT_NAMES, (2.0, 2.0, 2.0, 4.0, 4.0, 4.0))
        )
        self.rewards.pushing_speed.func = exponential_pushing_speed_cost_limit
        self.rewards.pushing_speed.params["max_cost"] = 100.0
        self.rewards.arm_speed_excess.func = exponential_joint_speed_cost_limit
        self.rewards.arm_speed_excess.params["max_cost"] = 100.0
        self.terminations.shelf_collision.params["wrist_clearance"] = 0.03
        self.events.object_spawn.params["excluded_x_half_angle"] = math.radians(15.0)


@configclass
class InspireDexTactileSweepOmniRenewalV2EnvCfg(InspireDexTactileSweepOmniRenewalV1EnvCfg):
    """Inherit v1 mechanics and reward stable occupancy until failure/timeout."""

    rewards: DexTactileOmniRenewalV2RewardsCfg = DexTactileOmniRenewalV2RewardsCfg()
    terminations: DexTactileOmniRenewalV2TerminationsCfg = DexTactileOmniRenewalV2TerminationsCfg()


@configclass
class InspireDexTactileSweepOmniRenewalV2OSCEnvCfg(InspireDexTactileSweepOmniRenewalV2EnvCfg):
    """Replace only Arm control, with 100 stiffness and small pose deltas."""

    def __post_init__(self):
        super().__post_init__()
        eef = self.scene.ee_frame.target_frames[0]
        self.actions.arm_action = BoundedOperationalSpaceControllerActionCfg(
            asset_name="robot",
            joint_names=list(ARM_JOINT_NAMES),
            body_name=eef.prim_path.rsplit("/", 1)[-1],
            body_offset=BoundedOperationalSpaceControllerActionCfg.OffsetCfg(
                pos=eef.offset.pos,
                rot=eef.offset.rot,
            ),
            controller_cfg=OperationalSpaceControllerCfg(
                target_types=["pose_rel"],
                impedance_mode="fixed",
                motion_stiffness_task=100.0,
                motion_damping_ratio_task=1.0,
                inertial_dynamics_decoupling=True,
                gravity_compensation=True,
            ),
            position_scale=0.005,
            orientation_scale=0.025,
            position_command_limit=0.005,
            orientation_command_limit=0.025,
        )
        # As in Renewal OSC, joint PD must not compete with OSC efforts.
        self.scene.robot.actuators["arm"].stiffness = 0.0
        self.scene.robot.actuators["arm"].damping = 0.0


@configclass
class InspireDexTactileSweepOmniRenewalV3EnvCfg(InspireDexTactileSweepOmniRenewalV1EnvCfg):
    """Branch directly from v1; calibrate progress weights against its logged returns."""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.pushing_speed.params["free_speed"] = 0.2
        self.rewards.pushing_speed.weight = -2.0
        # v1 run 2026-10-19_12-15-47, iterations 39027–39036:
        # 300 * (4/3) * 0.7783182442 / 3.3145449400 = 93.9276.
        # Rounded to 94: reaching:goal contribution approximately 3:4.
        self.rewards.goal_distance.weight = 94.0
        cylinder = self.scene.object_collection.rigid_objects["mass_cylinder"]
        cylinder.spawn.mass_props.mass *= 1.5
        # Scale the reset samples as well as the spawn mass. The shared event
        # updates inertia consistently from cached defaults, without compounding.
        self.events.physics_randomization.func = randomize_scaled_episode_physics
        self.events.physics_randomization.params["mass_scale"] = 1.5
