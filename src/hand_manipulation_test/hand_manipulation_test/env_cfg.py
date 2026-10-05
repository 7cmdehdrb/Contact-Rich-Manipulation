"""Robot-independent scene and manager configuration for reaching."""

from dataclasses import MISSING
import math

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs.mdp import time_out
from isaaclab.managers import ActionTermCfg, EventTermCfg, ObservationGroupCfg, ObservationTermCfg
from isaaclab.managers import RewardTermCfg, TerminationTermCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim.simulation_cfg import PhysxCfg
from isaaclab.utils import configclass

from .mdp import observations, rewards
from .mdp.commands import EpisodeTargetPositionCommandCfg
from .mdp.events import reset_robot_to_initial_state


@configclass
class ReachTaskCfg:
    """Sensor calibration and reward parameters retained from the source task."""

    arm_velocity_observation_scale: float = 3.14
    rotation_observation_scale_rad: float = math.pi
    wrench_force_observation_scale_n: float = 40.0
    wrench_moment_observation_scale_nm: float = 4.0
    tactile_threshold_n: float = 0.05
    goal_reward_sigma_m: float = 0.05
    success_distance_m: float = 0.01

    def validate(self):
        for name, value in vars(self).items():
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")


@configclass
class ReachSceneCfg(InteractiveSceneCfg):
    """An effort-controlled arm and its tactile hand, without task obstacles."""

    robot: ArticulationCfg = MISSING
    light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=2500.0),
    )


@configclass
class CommandsCfg:
    target_position = EpisodeTargetPositionCommandCfg()


@configclass
class ActionsCfg:
    arm_action: ActionTermCfg = MISSING
    hand_action: ActionTermCfg = MISSING


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObservationGroupCfg):
        arm_joint_position = ObservationTermCfg(func=observations.arm_joint_position)
        arm_joint_velocity = ObservationTermCfg(func=observations.arm_joint_velocity)
        eef_relative_position = ObservationTermCfg(func=observations.eef_relative_position)
        eef_relative_orientation = ObservationTermCfg(func=observations.eef_relative_orientation)
        hand_state = ObservationTermCfg(func=observations.hand_state)
        surface_header_and_tactile = ObservationTermCfg(func=observations.surface_header_and_tactile)
        wrist_wrench_c = ObservationTermCfg(func=observations.wrist_wrench)
        # Fixed target position in base_link axes, in metres without clipping.
        initial_target_relative_position = ObservationTermCfg(func=observations.initial_target_relative_position)
        last_action = ObservationTermCfg(func=observations.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventsCfg:
    reset_robot = EventTermCfg(func=reset_robot_to_initial_state, mode="reset")


@configclass
class RewardsCfg:
    eef_distance = RewardTermCfg(func=rewards.eef_target_distance_reward, weight=4.0)
    action_rate = RewardTermCfg(func=rewards.ExecutedActionRate, weight=0.06)


@configclass
class TerminationsCfg:
    time_out = TerminationTermCfg(func=time_out, time_out=True)


@configclass
class HandManipulationTestEnvCfg(ManagerBasedRLEnvCfg):
    """Shared task configuration; robot-specific assets and actions are inherited."""

    task: ReachTaskCfg = ReachTaskCfg()
    scene: ReachSceneCfg = ReachSceneCfg(
        num_envs=4096, env_spacing=2.5, replicate_physics=True, lazy_sensor_update=False
    )
    commands: CommandsCfg = CommandsCfg()
    actions: ActionsCfg = ActionsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    events: EventsCfg = EventsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    curriculum = None
    sim: SimulationCfg = SimulationCfg(
        dt=0.01,
        render_interval=2,
        gravity=(0.0, 0.0, -9.81),
        physx=PhysxCfg(
            bounce_threshold_velocity=0.20,
            gpu_max_rigid_contact_count=2**22,
            gpu_max_rigid_patch_count=5 * 2**17,
            gpu_found_lost_aggregate_pairs_capacity=1024 * 1024 * 16 * 16,
            gpu_total_aggregate_pairs_capacity=16 * 1024 * 16,
            friction_correlation_distance=0.00625,
        ),
    )

    def __post_init__(self):
        self.task.validate()
        self.decimation = 2
        self.episode_length_s = 10.0
        self.is_finite_horizon = False
        self.num_rerenders_on_reset = 0
        self.sim.render_interval = self.decimation
        self.viewer.eye = (1.50, 2.00, 2.00)
        self.viewer.lookat = (-0.55, 0.0, 1.15)
