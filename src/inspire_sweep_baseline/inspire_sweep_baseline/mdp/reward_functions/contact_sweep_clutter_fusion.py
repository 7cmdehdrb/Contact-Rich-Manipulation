"""Exact reward aggregation using the original local cost functions.

Independent, bounded EEF and compensated-wrench costs for Clutter Fusion.

Reward and failure reductions over the episode's active object slots.

Whole-Hand contact and bounded physical changes for Fusion v1.0.

These are reward rates. RewardManager applies each weight and ``step_dt``
once. Object acceleration is read directly from Isaac Lab / PhysX.

Zero-disabled goal occupancy and consecutive three-second success."""

from __future__ import annotations

from ..common.contact_sweep_clutter import all_object_tilt_angles
from ..common.contact_sweep_clutter_fusion_contacts import hand_obstacle_contact_force_w
from ..common.contact_sweep_clutter_fusion_contacts import whole_hand_contact_force_w
from ..common.contact_sweep_clutter_fusion_object_state import active_object_mask
from .contact_sweep_clutter import all_object_tilt_cost
from .contact_sweep_clutter import goal_hold_xy
from .contact_sweep_clutter import obstacle_displacement_l2
from .goal_rewards import _exponential_speed_excess_limit
from .goal_rewards import action_rate_l2
from .goal_rewards import joint_vel_l2
from isaaclab.managers import ManagerTermBase
from isaaclab.utils.math import quat_apply
import math as math
import torch as torch

def combined_action_rate(env, max_cost=100.0):
    """Preserve each group's output cap before adding raw-action square sums."""
    return (action_rate_l2(env, action_group="arm", max_cost=max_cost)
            + action_rate_l2(env, action_group="hand", max_cost=max_cost))


def combined_joint_velocity(env, arm_cfg, hand_cfg, hand_ratio=1.0 / 3.0, max_cost=100.0):
    """With weight -0.03, retain Arm -0.03 and Hand -0.01 contributions."""
    return (joint_vel_l2(env, asset_cfg=arm_cfg, max_cost=max_cost)
            + hand_ratio * joint_vel_l2(env, asset_cfg=hand_cfg, max_cost=max_cost))


def combined_object_cost(
    env, free_angle=math.radians(5.0), fail_angle=0.9,
    displacement_ratio=2.5, displacement_max_cost=0.04,
):
    """Weight -2 reproduces -2 tilt and -5 displacement; -30 scales both x15."""
    return (
        all_object_tilt_cost(env, free_angle=free_angle, fail_angle=fail_angle)
        + displacement_ratio * obstacle_displacement_l2(env, max_cost=displacement_max_cost)
    )


def eef_linear_speed_cost(
    env, free_speed=1.5, max_cost=100.0,
    asset_name="robot", body_name="axia80_link", frame_name="ee_frame",
):
    """Bounded exponential speed cost at the actual EEF point, in m/s.

    Link-origin velocity and angular velocity are instantaneous simulation
    quantities. Transport the velocity to the FrameTransformer's target point;
    neither COM velocity nor a control-step position difference is substituted.
    RewardManager applies the negative weight and control dt once.
    """
    robot = env.scene[asset_name]
    body_index = robot.body_names.index(body_name)
    data = robot.data
    offset_w = env.scene[frame_name].data.target_pos_w[:, 0] - data.body_link_pos_w[:, body_index]
    velocity_w = data.body_link_lin_vel_w[:, body_index] + torch.cross(
        data.body_link_ang_vel_w[:, body_index], offset_w, dim=-1,
    )
    speed = torch.linalg.vector_norm(velocity_w, dim=-1)
    speed = torch.nan_to_num(speed, nan=math.inf, posinf=math.inf, neginf=math.inf)
    return _exponential_speed_excess_limit(speed, free_speed, max_cost)


def eef_upward_z_orientation_cost(env, onset_angle=math.pi / 3.0, frame_name="ee_frame"):
    """Finite [0, 1] cost when EEF +Z enters the upward cone.

    The EEF +Z direction points away from the palm normal. Upward +Z therefore
    means a downward-facing palm. Only angles smaller than ``onset_angle`` to
    world +Z are penalized; a downward +Z (upward-facing palm) is free.
    A fixed reward weight provides the calibrated scale, with no reward budget.
    """
    if not math.isfinite(onset_angle) or not 0.0 < onset_angle <= math.pi / 2.0:
        raise ValueError("onset_angle must be finite and in (0, pi/2]")
    quaternion = env.scene[frame_name].data.target_quat_w[:, 0]
    local_z = torch.zeros_like(quaternion[:, :3])
    local_z[:, 2] = 1.0
    cosine = quat_apply(quaternion, local_z)[:, 2]
    cosine = torch.nan_to_num(cosine, nan=1.0, posinf=1.0, neginf=-1.0).clamp(-1.0, 1.0)
    threshold = math.cos(onset_angle)
    return ((cosine - threshold) / (1.0 - threshold)).clamp(0.0, 1.0).square()


class WrenchRatePenalty(ManagerTermBase):
    """Bounded exponential cost of unusually large compensated wrench rates.

    Force and moment use separate physical units. World axes avoid a spurious
    delta when only the EEF axes rotate; moments retain the moving EEF origin.
    The maximum normalized force/moment excess defines one combined severity.
    Both the exponent and raw result are bounded before weighting.

    The term must follow all positive reward terms. At weight -0.1 its absolute
    maximum is -0.1/s, while cumulative cost cannot exceed 5% of positive reward
    already earned in the current episode. Reset/invalid history is never used
    for a rate. Repeated calls in the same physical step reuse the raw result.
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._previous_wrench = torch.zeros((env.num_envs, 6), device=env.device)
        self._previous_valid = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self._cached_raw = torch.zeros(env.num_envs, device=env.device)
        self._cached_key = None
        self._term_name = None
        self._positive_names = None

    def reset(self, env_ids=None):
        if env_ids is None:
            env_ids = slice(None)
        self._previous_wrench[env_ids] = 0.0
        self._previous_valid[env_ids] = False
        self._cached_raw[env_ids] = 0.0
        self._cached_key = None

    def _reward_budget(self, env, max_positive_fraction):
        manager = env.reward_manager
        if self._term_name is None:
            names = manager.active_terms
            own_names = [name for name in names if manager.get_term_cfg(name).func is self]
            if len(own_names) != 1:
                raise RuntimeError("WrenchRatePenalty must appear exactly once in RewardManager")
            own_name = own_names[0]
            weight = manager.get_term_cfg(own_name).weight
            if not math.isfinite(weight) or weight >= 0.0:
                raise ValueError("WrenchRatePenalty requires a finite negative reward weight")
            positive_names = [name for name in names if manager.get_term_cfg(name).weight > 0.0]
            if any(names.index(name) >= names.index(own_name) for name in positive_names):
                raise RuntimeError("WrenchRatePenalty must follow every positive reward term")
            self._term_name = own_name
            self._positive_names = positive_names
        weight = manager.get_term_cfg(self._term_name).weight
        if not math.isfinite(weight) or weight >= 0.0:
            raise ValueError("WrenchRatePenalty requires a finite negative reward weight")
        earned = torch.zeros(env.num_envs, device=env.device)
        for name in self._positive_names:
            earned += manager._episode_sums[name].clamp_min(0.0)
        earned = torch.nan_to_num(earned, nan=0.0, posinf=0.0, neginf=0.0)
        spent = (-manager._episode_sums[self._term_name]).clamp_min(0.0)
        spent = torch.nan_to_num(spent, nan=math.inf, posinf=math.inf, neginf=0.0)
        remaining = (max_positive_fraction * earned - spent).clamp_min(0.0)
        return remaining / (abs(weight) * env.step_dt)

    def __call__(
        self, env, free_force_rate=2000.0, free_moment_rate=100.0,
        saturation_force_rate=5000.0, saturation_moment_rate=250.0,
        sharpness=3.0, max_positive_fraction=0.05,
    ):
        for name, free, saturation in (
            ("force", free_force_rate, saturation_force_rate),
            ("moment", free_moment_rate, saturation_moment_rate),
        ):
            if not math.isfinite(free) or free < 0.0:
                raise ValueError(f"free_{name}_rate must be finite and nonnegative")
            if not math.isfinite(saturation) or saturation <= free:
                raise ValueError(f"saturation_{name}_rate must be finite and exceed free rate")
        if not math.isfinite(sharpness) or not 0.0 < sharpness <= 50.0:
            raise ValueError("sharpness must be finite and in (0, 50]")
        if not math.isfinite(env.step_dt) or env.step_dt <= 0.0:
            raise ValueError("step_dt must be finite and positive")
        if not math.isfinite(max_positive_fraction) or not 0.0 <= max_positive_fraction <= 0.05:
            raise ValueError("max_positive_fraction must be in [0, 0.05]")

        key = (env.common_step_counter, env._sim_step_counter)
        if key != self._cached_key:
            reader = env.observation_manager.cfg.policy.wrist_wrench.func
            current = reader.world_wrench(env)
            fresh = env.sensor_data_fresh
            finite = torch.isfinite(current).all(dim=-1)
            delta = current - self._previous_wrench
            force_rate = torch.linalg.vector_norm(delta[:, :3], dim=-1) / env.step_dt
            moment_rate = torch.linalg.vector_norm(delta[:, 3:], dim=-1) / env.step_dt
            force_excess = (force_rate - free_force_rate).clamp_min(0.0) / (saturation_force_rate - free_force_rate)
            moment_excess = (moment_rate - free_moment_rate).clamp_min(0.0) / (saturation_moment_rate - free_moment_rate)
            severity = torch.maximum(force_excess, moment_excess)
            severity = torch.nan_to_num(severity, nan=1.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
            raw = (torch.expm1(sharpness * severity) / math.expm1(sharpness)).clamp(0.0, 1.0)
            self._cached_raw[:] = torch.where(fresh & self._previous_valid, raw, 0.0)
            self._previous_wrench[:] = torch.where((fresh & finite)[:, None], current, 0.0)
            self._previous_valid[:] = fresh & finite
            self._cached_key = key
        return torch.minimum(self._cached_raw, self._reward_budget(env, max_positive_fraction))


def active_object_tilt_cost(env, free_angle=math.radians(5.0), fail_angle=0.9):
    """Sum capped tilt costs for the active target and obstacles."""
    if not (0.0 <= free_angle < fail_angle <= math.pi):
        raise ValueError("Tilt thresholds must satisfy 0 <= free_angle < fail_angle <= pi")
    angles = all_object_tilt_angles(env)
    active = active_object_mask(env, angles)
    per_object_tilt = ((angles - free_angle) / (fail_angle - free_angle)).clamp(0.0, 1.0).square()
    return torch.where(active, per_object_tilt, 0.0).sum(dim=-1)


def active_obstacle_displacement_cost(env, displacement_max_cost=0.04):
    """Sum squared 3D displacement of active non-target objects, then cap it."""
    if not math.isfinite(displacement_max_cost) or displacement_max_cost <= 0.0:
        raise ValueError("displacement_max_cost must be finite and positive")
    position = env.scene["object_collection"].data.object_pos_w
    squared = (position - env._initial_object_positions_w).square().sum(dim=-1)
    active = active_object_mask(env, squared)
    object_ids = torch.arange(position.shape[1], device=position.device).view(1, -1)
    obstacles = active & (object_ids != env.target_id.reshape(env.num_envs, 1))
    displacement = torch.where(obstacles, squared, 0.0).sum(dim=-1)
    return torch.nan_to_num(
        displacement, nan=displacement_max_cost, posinf=displacement_max_cost, neginf=displacement_max_cost,
    ).clamp(max=displacement_max_cost)


def combined_active_object_cost(
    env, free_angle=math.radians(5.0), fail_angle=0.9,
    displacement_ratio=2.5, displacement_max_cost=0.04,
):
    """Retain the parent tilt/displacement scales, excluding inactive slots.

    Active targets still incur tilt cost; only active non-target objects incur
    displacement cost. Apply the displacement cap after summing active slots,
    exactly as in the parent. Parked states (including NaNs) do not contribute.
    """
    return active_object_tilt_cost(env, free_angle, fail_angle) + displacement_ratio * active_obstacle_displacement_cost(
        env, displacement_max_cost
    )


def hand_contact_any(env, threshold_n=0.01, goal_contact_fade_width=0.015):
    """Contact rate faded to zero at the current goal-hold boundary.

    As with the previous tactile ON reward, sustained contact earns the same
    rate on every step, without a duration ramp. Multiply that rate by
    ``clamp((XY_goal_distance - current_hold_tolerance) / fade_width, 0, 1)``.
    Read the live goal-hold term so curriculum updates affect both rewards
    together. A zero fade width disables distance fading (alpha is always 1).
    The 17 tactile channels remain observation-only.
    """
    if not math.isfinite(threshold_n) or threshold_n <= 0.0:
        raise ValueError("threshold_n must be finite and positive")
    if not math.isfinite(goal_contact_fade_width) or goal_contact_fade_width < 0.0:
        raise ValueError("goal_contact_fade_width must be finite and nonnegative")
    force = whole_hand_contact_force_w(env)
    magnitude = torch.linalg.vector_norm(force, dim=-1)
    valid = torch.isfinite(force).all(dim=-1) & env.sensor_data_fresh
    if goal_contact_fade_width == 0.0:
        return (valid & (magnitude >= threshold_n)).to(magnitude.dtype)
    hold_params = env.reward_manager.get_term_cfg("goal_hold").params
    tolerance = hold_params["position_tolerance"]
    goal = env.command_manager.get_command(hold_params.get("command_name", "target_goal_pos"))
    rows = torch.arange(env.num_envs, device=env.device)
    position = env.scene["object_collection"].data.object_pos_w[rows, env.target_id[:, 0].long()]
    distance = torch.linalg.vector_norm(position[:, :2] - goal[:, :2], dim=-1)
    alpha = ((distance - tolerance) / goal_contact_fade_width).clamp(0.0, 1.0)
    return torch.where(valid & (magnitude >= threshold_n), alpha, 0.0)


def obstacle_contact_force_cost(env, free_force=20.0, max_cost=100.0):
    """Exponential excess of net Hand force from non-target obstacle contacts.

    Contact identities are filtered by the sensors before summing their
    world-axis force vectors. Target contacts cannot contribute. The local
    joint-speed cost's ``0.5 * expm1(2 * excess)`` and finite cap are retained.
    """
    force = hand_obstacle_contact_force_w(env)
    magnitude = torch.linalg.vector_norm(force, dim=-1)
    magnitude = torch.nan_to_num(magnitude, nan=math.inf, posinf=math.inf, neginf=math.inf)
    cost = _exponential_speed_excess_limit(magnitude, free_force, max_cost)
    return torch.where(env.sensor_data_fresh, cost, 0.0)


class WrenchMagnitudeChangePenalty(WrenchRatePenalty):
    """Bounded change of scalar force/moment magnitudes at the observed EEF.

    ``abs(norm(current) - norm(previous))`` has units N and Nm, respectively;
    rotating a vector at constant magnitude is free. The observation's exact
    compensated EEF sample supplies both the reference point and axes.

    Preserve Fusion's normalized exponential, maximum of the two severities,
    [0, 1] raw cap, and cumulative 5% earned-positive-reward budget. Saturation
    tracks each current free threshold at ratio 2.5, preserving the parent's
    saturation/free ratio throughout the new threshold curriculum.
    """

    def reset(self, env_ids=None):
        if env_ids is None:
            env_ids = slice(None)
        self._previous_wrench[env_ids] = 0.0
        self._previous_valid[env_ids] = False
        self._cached_raw[env_ids] = 0.0
        # Preserve the physical-step key: a partial reset must not recompute
        # unaffected rows against their already-updated history in this step.

    def __call__(
        self, env, free_force_change=50.0, free_moment_change=5.0,
        saturation_ratio=2.5, sharpness=3.0, max_positive_fraction=0.05,
    ):
        for name, value in (("force", free_force_change), ("moment", free_moment_change)):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"free_{name}_change must be finite and positive")
        if not math.isfinite(saturation_ratio) or saturation_ratio <= 1.0:
            raise ValueError("saturation_ratio must be finite and greater than one")
        if not math.isfinite(sharpness) or not 0.0 < sharpness <= 50.0:
            raise ValueError("sharpness must be finite and in (0, 50]")
        if not math.isfinite(max_positive_fraction) or not 0.0 <= max_positive_fraction <= 0.05:
            raise ValueError("max_positive_fraction must be in [0, 0.05]")

        key = (env.common_step_counter, env._sim_step_counter)
        if key != self._cached_key:
            term = env.observation_manager.cfg.policy.wrist_wrench
            current = term.func(env, **term.params)
            fresh = env.sensor_data_fresh
            finite = torch.isfinite(current).all(dim=-1)
            force_change = (
                torch.linalg.vector_norm(current[:, :3], dim=-1)
                - torch.linalg.vector_norm(self._previous_wrench[:, :3], dim=-1)
            ).abs()
            moment_change = (
                torch.linalg.vector_norm(current[:, 3:], dim=-1)
                - torch.linalg.vector_norm(self._previous_wrench[:, 3:], dim=-1)
            ).abs()
            force_excess = (force_change - free_force_change).clamp_min(0.0) / (
                (saturation_ratio - 1.0) * free_force_change
            )
            moment_excess = (moment_change - free_moment_change).clamp_min(0.0) / (
                (saturation_ratio - 1.0) * free_moment_change
            )
            severity = torch.maximum(force_excess, moment_excess)
            severity = torch.nan_to_num(severity, nan=1.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
            raw = (torch.expm1(sharpness * severity) / math.expm1(sharpness)).clamp(0.0, 1.0)
            self._cached_raw[:] = torch.where(fresh & self._previous_valid, raw, 0.0)
            self._previous_wrench[:] = torch.where((fresh & finite)[:, None], current, 0.0)
            self._previous_valid[:] = fresh & finite
            self._cached_key = key
        return torch.minimum(self._cached_raw, self._reward_budget(env, max_positive_fraction))


class ObjectAccelerationPenalty(ManagerTermBase):
    """Sum independent exponential costs of every object's 3D acceleration.

    Read Isaac Lab's world-frame center-of-mass linear acceleration from
    PhysX (last physics step, not a control-interval velocity difference).
    Its Euclidean norm penalizes acceleration and deceleration symmetrically.
    Each active object has a 100-unit cap, so the sum is bounded by active
    object count times 100. Keep only reset/activation validity, not velocity
    history: the first sample after reset or activation is suppressed.
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        asset_name = cfg.params.get("asset_name", "object_collection")
        acceleration = env.scene[asset_name].data.object_com_lin_acc_w
        self._previous_valid = torch.zeros(acceleration.shape[:-1], dtype=torch.bool, device=acceleration.device)
        self._cached_cost = torch.zeros(env.num_envs, device=env.device)
        self._cached_key = None

    def reset(self, env_ids=None):
        if env_ids is None:
            env_ids = slice(None)
        self._previous_valid[env_ids] = False
        self._cached_cost[env_ids] = 0.0

    def __call__(self, env, free_acceleration=1.0, max_cost=100.0, asset_name="object_collection"):
        if not math.isfinite(free_acceleration) or free_acceleration < 0.0:
            raise ValueError("free_acceleration must be finite and nonnegative")
        if not math.isfinite(max_cost) or max_cost <= 0.0:
            raise ValueError("max_cost must be finite and positive")
        key = (env.common_step_counter, env._sim_step_counter)
        if key != self._cached_key:
            acceleration_w = env.scene[asset_name].data.object_com_lin_acc_w
            active = active_object_mask(env, acceleration_w)
            acceleration = torch.linalg.vector_norm(acceleration_w, dim=-1)
            acceleration = torch.nan_to_num(
                acceleration, nan=math.inf, posinf=math.inf, neginf=math.inf,
            )
            costs = _exponential_speed_excess_limit(acceleration, free_acceleration, max_cost)
            self._cached_cost[:] = torch.where(active & self._previous_valid, costs, 0.0).sum(dim=-1)
            self._previous_valid[:] = active
            self._cached_key = key
        return self._cached_cost.clone()


def scheduled_goal_hold_xy(env, position_tolerance=0.0, command_name="target_goal_pos"):
    """A zero radius disables occupancy; positive radii retain v1.0 semantics."""
    if position_tolerance == 0.0:
        return torch.zeros(env.num_envs, device=env.device)
    return goal_hold_xy(env, position_tolerance=position_tolerance, command_name=command_name)
