"""Pure Torch episode history shared by pushing rewards and terminations."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, fields
import math

import torch

from .action_math import quaternion_conjugate, quaternion_multiply, quaternion_rotate, quaternion_to_matrix


@dataclass(frozen=True)
class PushStateConfig:
    table_size_m: tuple[float, float, float] = (0.36, 1.00, 0.04)
    cube_size_m: float = 0.06
    contact_threshold_n: float = 0.01
    edge_margin_m: float = 0.005
    alignment_cos: float = math.cos(math.pi / 6)
    approach_sigma_m: float = 0.05
    progress_sigma_fraction: float = 0.5
    palm_height_offset_m: float = 0.03
    success_distance_m: float = 0.01
    success_speed_m_s: float = 0.02
    success_hold_time_s: float = 0.20
    push_seen_distance_m: float = 0.005

    def __post_init__(self):
        if len(self.table_size_m) != 3 or any(not math.isfinite(x) or x <= 0 for x in self.table_size_m):
            raise ValueError("Table size must contain three positive finite dimensions")
        positive = ("cube_size_m", "contact_threshold_n", "approach_sigma_m", "progress_sigma_fraction", "success_distance_m",
                    "success_speed_m_s", "success_hold_time_s", "push_seen_distance_m")
        for name in positive:
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("edge_margin_m", "palm_height_offset_m"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if not 0 < self.alignment_cos <= 1:
            raise ValueError("alignment_cos must be in (0, 1]")
        if self.edge_margin_m >= min(self.table_size_m[:2]) / 2:
            raise ValueError("The edge margin must leave a nonempty table surface")


@dataclass(frozen=True)
class PushSnapshot:
    initial_cube_pos_w: torch.Tensor
    goal_pos_w: torch.Tensor
    direction_w: torch.Tensor
    distance_m: torch.Tensor
    cube_pos_w: torch.Tensor
    cube_quat_w: torch.Tensor
    cube_lin_vel_w: torch.Tensor
    palm_pos_w: torch.Tensor
    hand_quat_w: torch.Tensor
    cube_palm_forces_w_history: torch.Tensor
    cube_table_forces_w_history: torch.Tensor
    robot_table_failure: torch.Tensor
    table_pos_w: torch.Tensor
    table_quat_w: torch.Tensor
    live: torch.Tensor | None = None
    additional_failure: torch.Tensor | None = None
    approach_target_pos_w: torch.Tensor | None = None


@dataclass(frozen=True)
class PushStepState:
    approach_delta: torch.Tensor
    progress_delta: torch.Tensor
    backslide_delta: torch.Tensor
    first_contact: torch.Tensor
    maintaining_contact: torch.Tensor
    alignment_penalty: torch.Tensor
    success: torch.Tensor
    failure: torch.Tensor
    grounded: torch.Tensor
    valid_push_contact: torch.Tensor
    contact_seen: torch.Tensor
    push_seen: torch.Tensor
    settled_time_s: torch.Tensor
    cube_goal_distance_m: torch.Tensor
    palm_cube_contact: torch.Tensor
    palm_alignment_cos: torch.Tensor
    force_alignment_cos: torch.Tensor
    footprint_failure: torch.Tensor
    fall_failure: torch.Tensor
    table_failure: torch.Tensor
    invalid_state: torch.Tensor
    additional_failure: torch.Tensor


_BOOLEAN_FIELDS = frozenset({"first_contact", "success", "failure", "grounded", "valid_push_contact",
                           "contact_seen", "push_seen", "palm_cube_contact", "footprint_failure",
                           "fall_failure", "table_failure", "invalid_state", "additional_failure"})


def push_progress_potential(goal_distance_m: torch.Tensor, command_distance_m: torch.Tensor,
                            sigma_fraction: float) -> torch.Tensor:
    """Bounded progress potential, with more sensitivity near the initial Cube.

    For ``p = clamp(1 - distance / command_length, 0, 1)``, use
    ``phi(p) = (1 - exp(-p / sigma)) / (1 - exp(-1 / sigma))``.
    This is zero at the initial goal distance and one at the goal. Smaller
    sigma increases the reward for early forward motion without increasing
    the total potential available in an episode. ``expm1`` retains accuracy
    for small progress and for large sigma, where the curve approaches linear.
    Inputs are the finite distances and positive lengths sanitized by the
    tracker; sigma is validated by :class:`PushStateConfig`.
    """

    progress_fraction = (1.0 - goal_distance_m / command_distance_m).clamp(0.0, 1.0)
    normalizer = -math.expm1(-1.0 / sigma_fraction)
    return (-torch.expm1(-progress_fraction / sigma_fraction) / normalizer).clamp(0.0, 1.0)


class PushStateTracker:
    """Advance each environment exactly once per physical policy interval.

    Record distances update even when contact is invalid. A later touch cannot
    retroactively collect an earlier drift, airborne move, or release/recovery
    loop. Only reset rows lose their record, contact latch, and settling timer.
    """

    def __init__(self, cfg: PushStateConfig, num_envs: int, device, dtype=torch.float32):
        if num_envs <= 0:
            raise ValueError("num_envs must be positive")
        self.cfg, self.num_envs, self.device, self.dtype = cfg, num_envs, torch.device(device), dtype
        self._initialized = torch.zeros(num_envs, device=self.device, dtype=torch.bool)
        self._has_uninitialized = True
        self._cached_counter: int | None = None
        self._last_counter = torch.full((num_envs,), -1, device=self.device, dtype=torch.long)
        self._previous_goal_error = torch.zeros(num_envs, device=self.device, dtype=dtype)
        self._best_goal_error = torch.zeros_like(self._previous_goal_error)
        self._best_approach = torch.zeros_like(self._previous_goal_error)
        self._contact_seen = torch.zeros_like(self._initialized)
        self._push_seen = torch.zeros_like(self._initialized)
        self._settled_time = torch.zeros_like(self._previous_goal_error)
        self._failure_flags = torch.zeros((num_envs, 4), device=self.device, dtype=torch.bool)
        self._additional_failure = torch.zeros_like(self._initialized)
        self._cached = PushStepState(**{
            field.name: torch.zeros(num_envs, device=self.device,
                                   dtype=torch.bool if field.name in _BOOLEAN_FIELDS else dtype)
            for field in fields(PushStepState)
        })

    def _ids(self, env_ids: Sequence[int] | torch.Tensor | slice | None):
        if env_ids is None or isinstance(env_ids, slice):
            return torch.arange(self.num_envs, device=self.device)[slice(None) if env_ids is None else env_ids]
        return torch.as_tensor(env_ids, device=self.device, dtype=torch.long).reshape(-1)

    def _validate(self, sample: PushSnapshot):
        for name in ("initial_cube_pos_w", "goal_pos_w", "direction_w", "cube_pos_w", "cube_lin_vel_w",
                     "palm_pos_w", "table_pos_w"):
            if getattr(sample, name).shape != (self.num_envs, 3):
                raise ValueError(f"{name} must have shape (N, 3)")
        for name in ("cube_quat_w", "hand_quat_w", "table_quat_w"):
            if getattr(sample, name).shape != (self.num_envs, 4):
                raise ValueError(f"{name} must have shape (N, 4), in wxyz order")
        for name in ("distance_m", "robot_table_failure", "live"):
            value = getattr(sample, name)
            if value is not None and value.shape != (self.num_envs,):
                raise ValueError(f"{name} must have shape (N,)")
        if sample.additional_failure is not None:
            if sample.additional_failure.shape != (self.num_envs,):
                raise ValueError("additional_failure must have shape (N,)")
            if sample.additional_failure.dtype != torch.bool:
                raise ValueError("additional_failure must be a boolean tensor")
        if sample.approach_target_pos_w is not None and sample.approach_target_pos_w.shape != (self.num_envs, 3):
            raise ValueError("approach_target_pos_w must have shape (N, 3)")
        palm, table = sample.cube_palm_forces_w_history, sample.cube_table_forces_w_history
        if palm.ndim != 5 or palm.shape[0] != self.num_envs or tuple(palm.shape[2:]) != (1, 17, 3) or palm.shape[1] == 0:
            raise ValueError("Cube-palm history must have shape (N, T, 1, 17, 3), T > 0")
        if table.ndim != 5 or tuple(table.shape) != (self.num_envs, palm.shape[1], 1, 1, 3):
            raise ValueError("Cube-table history must have shape (N, T, 1, 1, 3), matching palm T")

    def _geometry(self, sample: PushSnapshot):
        # Bad numerical states are classified as failures and never feed NaNs
        # into shaping history or returned rewards.
        invalid = ~torch.isfinite(sample.distance_m) | (sample.distance_m <= 0)
        safe = {}
        for name in ("initial_cube_pos_w", "goal_pos_w", "direction_w", "cube_pos_w", "cube_lin_vel_w",
                     "palm_pos_w", "table_pos_w", "cube_quat_w", "hand_quat_w", "table_quat_w"):
            value = getattr(sample, name)
            invalid |= ~torch.isfinite(value).all(-1)
            safe[name] = torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)
        for name in ("cube_quat_w", "hand_quat_w", "table_quat_w"):
            norm = torch.linalg.vector_norm(safe[name], dim=-1, keepdim=True)
            invalid |= norm[:, 0] < 1.e-8
            identity = safe[name].new_tensor((1., 0., 0., 0.)).expand(self.num_envs, -1)
            safe[name] = torch.where(norm > 1.e-8, safe[name] / norm.clamp_min(1.e-8), identity)
        direction_norm = torch.linalg.vector_norm(safe["direction_w"], dim=-1, keepdim=True)
        invalid |= (direction_norm[:, 0] < 1.e-8) | (safe["direction_w"][:, 2].abs() > 1.e-6)
        direction = safe["direction_w"] / direction_norm.clamp_min(1.e-8)
        inverse_table = quaternion_conjugate(safe["table_quat_w"])
        centre = quaternion_rotate(inverse_table, safe["cube_pos_w"] - safe["table_pos_w"])
        relative_rotation = quaternion_to_matrix(quaternion_multiply(inverse_table, safe["cube_quat_w"]))
        half_cube = self.cfg.cube_size_m / 2
        extent = relative_rotation.abs().sum(-1) * half_cube
        safe_extent = centre.new_tensor(self.cfg.table_size_m[:2]) / 2 - self.cfg.edge_margin_m
        footprint = (centre[:, :2].abs() + extent[:, :2] > safe_extent + 1.e-7).any(-1)
        fall = centre[:, 2] < self.cfg.table_size_m[2] / 2 - half_cube
        local_direction = quaternion_rotate(quaternion_conjugate(safe["cube_quat_w"]), direction)
        support_extent = local_direction.abs().sum(-1) * half_cube
        if sample.approach_target_pos_w is None:
            approach_target = safe["cube_pos_w"] - support_extent[:, None] * direction
            approach_target[:, 2] += self.cfg.palm_height_offset_m
        else:
            invalid |= ~torch.isfinite(sample.approach_target_pos_w).all(-1)
            approach_target = torch.nan_to_num(sample.approach_target_pos_w, nan=0., posinf=0., neginf=0.)
        gap = torch.linalg.vector_norm(safe["palm_pos_w"] - approach_target, dim=-1)
        approach = torch.exp(-gap / self.cfg.approach_sigma_m)
        error_xy = torch.linalg.vector_norm(safe["cube_pos_w"][:, :2] - safe["goal_pos_w"][:, :2], dim=-1)
        error_3d = torch.linalg.vector_norm(safe["cube_pos_w"] - safe["goal_pos_w"], dim=-1)
        palm_normal = quaternion_rotate(safe["hand_quat_w"], direction.new_tensor((0., 1., 0.)).expand_as(direction))
        palm_dot = (palm_normal * direction).sum(-1).clamp(-1., 1.)
        forward = ((safe["cube_pos_w"] - safe["initial_cube_pos_w"]) * direction).sum(-1)
        speed = torch.linalg.vector_norm(safe["cube_lin_vel_w"], dim=-1)
        return invalid, footprint, fall, direction, approach, error_xy, error_3d, palm_dot, forward, speed

    def reset(self, env_ids, snapshot: PushSnapshot):
        """Initialize subset history from realized reset poses, ignoring forces."""

        self._validate(snapshot)
        index = self._ids(env_ids)
        if not len(index):
            return
        _, _, _, _, approach, error_xy, *_ = self._geometry(snapshot)
        self._previous_goal_error[index] = error_xy[index]
        self._best_goal_error[index] = error_xy[index]
        self._best_approach[index] = approach[index]
        self._contact_seen[index] = False
        self._push_seen[index] = False
        self._settled_time[index] = 0
        self._failure_flags[index] = False
        self._additional_failure[index] = False
        self._initialized[index] = True
        # Once every row has been initialized, resets cannot make it uninitialized
        # again. Avoid a device-to-host read on every subsequent partial reset.
        if self._has_uninitialized:
            self._has_uninitialized = not bool(self._initialized.all())
        self._last_counter[index] = -1
        self._cached_counter = None
        # Create new tensors so a previously returned transition stays intact.
        cached = {}
        for field in fields(PushStepState):
            value = getattr(self._cached, field.name).clone()
            value[index] = 0
            if field.name == "cube_goal_distance_m":
                value[index] = torch.linalg.vector_norm(snapshot.cube_pos_w[index] - snapshot.goal_pos_w[index], dim=-1)
            cached[field.name] = value
        self._cached = PushStepState(**cached)

    def update(self, snapshot: PushSnapshot, *, physics_counter: int, step_dt: float) -> PushStepState:
        if not math.isfinite(step_dt) or step_dt <= 0:
            raise ValueError("step_dt must be finite and positive")
        self._validate(snapshot)
        # Repeated termination/reward/metric queries share a Python-counter
        # fast path, avoiding a GPU-to-host boolean read for every manager term.
        if self._cached_counter == physics_counter:
            return self._cached
        changed = self._last_counter != physics_counter
        # The Python cache already handles repeated queries. After a partial
        # reset, compute with this per-row mask so other rows retain their
        # previous transition, without synchronizing changed.any() to the host.
        if self._has_uninitialized:
            self.reset(torch.where(~self._initialized)[0], snapshot)
        invalid, footprint, fall, direction, approach, error_xy, error_3d, palm_dot, forward, speed = self._geometry(snapshot)
        live = torch.ones_like(changed) if snapshot.live is None else snapshot.live.bool()
        palm_forces = snapshot.cube_palm_forces_w_history[:, :, 0]
        table_forces = snapshot.cube_table_forces_w_history[:, :, 0, 0]
        invalid |= ~torch.isfinite(palm_forces).flatten(1).all(-1) | ~torch.isfinite(table_forces).flatten(1).all(-1)
        palm_forces = torch.nan_to_num(palm_forces, nan=0., posinf=0., neginf=0.)
        table_forces = torch.nan_to_num(table_forces, nan=0., posinf=0., neginf=0.)
        threshold = self.cfg.contact_threshold_n
        pair_norm = torch.linalg.vector_norm(palm_forces, dim=-1)
        active_pair = pair_norm >= threshold
        palm_substep = active_pair.any(-1)
        support_substep = (torch.linalg.vector_norm(table_forces, dim=-1) >= threshold) & (table_forces[:, :, 2] >= threshold)
        grounded = support_substep[:, 0] & live
        # Each pad's force is ON Cube. A valid forward pair remains valid
        # beside an opposing pair, while many weak pairs cannot combine into
        # a false threshold crossing or a synthetic aligned normal.
        force_dot = (palm_forces * direction[:, None, None]).sum(-1) / pair_norm.clamp_min(1.e-8)
        force_dot = force_dot.clamp(-1., 1.)
        aligned_substep = (active_pair & (force_dot >= self.cfg.alignment_cos)).any(-1)
        aligned_substep &= (palm_dot >= self.cfg.alignment_cos)[:, None]
        flags = torch.stack((snapshot.robot_table_failure.bool(), footprint, fall, invalid), dim=-1) & live[:, None]
        flags = self._failure_flags | flags
        additional = self._additional_failure
        if snapshot.additional_failure is not None:
            additional = additional | (snapshot.additional_failure & live)
        failure = flags.any(-1) | additional
        good = live & ~failure
        valid_push = (aligned_substep & support_substep).any(-1) & good
        first = valid_push & ~self._contact_seen
        contact_seen = self._contact_seen | valid_push
        push_seen = self._push_seen | (valid_push & grounded & (forward >= self.cfg.push_seen_distance_m))
        length = torch.nan_to_num(snapshot.distance_m, nan=1., posinf=1., neginf=1.).clamp_min(1.e-8)
        current_potential = push_progress_potential(error_xy, length, self.cfg.progress_sigma_fraction)
        record_potential = push_progress_potential(self._best_goal_error, length, self.cfg.progress_sigma_fraction)
        progress = (current_potential - record_potential).clamp_min(0.)
        progress *= (valid_push & grounded & good).to(progress.dtype)
        backslide = -(error_xy - self._previous_goal_error).clamp_min(0.) / length
        backslide *= (contact_seen & good).to(backslide.dtype)
        approach_delta = (approach - self._best_approach).clamp_min(0.)
        approach_delta *= (~contact_seen & good).to(approach_delta.dtype)
        maintaining = (valid_push & grounded & good).to(self.dtype)
        palm_contact = palm_substep.any(-1) & live
        alignment = -(1. - palm_dot) * .5 * good.to(self.dtype)
        settled = (error_3d <= self.cfg.success_distance_m) & (speed <= self.cfg.success_speed_m_s)
        settled &= grounded & contact_seen & good
        settled_time = torch.where(settled, self._settled_time + step_dt, torch.zeros_like(self._settled_time))
        success = (settled_time + 1.e-7 >= self.cfg.success_hold_time_s) & settled & ~failure
        result = PushStepState(
            approach_delta=approach_delta, progress_delta=progress, backslide_delta=backslide,
            first_contact=first, maintaining_contact=maintaining, alignment_penalty=alignment,
            success=success, failure=failure, grounded=grounded, valid_push_contact=valid_push,
            contact_seen=contact_seen, push_seen=push_seen, settled_time_s=settled_time,
            cube_goal_distance_m=error_3d, palm_cube_contact=palm_contact, palm_alignment_cos=palm_dot,
            force_alignment_cos=torch.where(active_pair, force_dot, torch.full_like(force_dot, -1.)).flatten(1).amax(-1),
            footprint_failure=flags[:, 1], fall_failure=flags[:, 2], table_failure=flags[:, 0], invalid_state=flags[:, 3],
            additional_failure=additional,
        )
        # Baselines/records always advance, including no-contact and failure
        # intervals. Only initialized reset rows can restart these histories.
        self._previous_goal_error[changed] = error_xy[changed]
        self._best_goal_error[changed] = torch.minimum(self._best_goal_error, error_xy)[changed]
        self._best_approach[changed] = torch.maximum(self._best_approach, approach)[changed]
        self._contact_seen[changed] = contact_seen[changed]
        self._push_seen[changed] = push_seen[changed]
        self._settled_time[changed] = settled_time[changed]
        self._failure_flags[changed] = flags[changed]
        self._additional_failure[changed] = additional[changed]
        self._last_counter[changed] = physics_counter
        self._cached = PushStepState(**{
            field.name: torch.where(changed, getattr(result, field.name), getattr(self._cached, field.name))
            for field in fields(PushStepState)
        })
        self._cached_counter = physics_counter
        return self._cached


__all__ = ["PushStateConfig", "PushSnapshot", "PushStepState", "PushStateTracker", "push_progress_potential"]
