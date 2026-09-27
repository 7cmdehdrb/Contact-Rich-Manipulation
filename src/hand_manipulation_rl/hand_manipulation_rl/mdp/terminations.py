"""Success, failure, and timeout classification for blind sweeping."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import torch


Tensor = torch.Tensor


class TerminationReason(IntEnum):
    """Single primary reason used for episode statistics."""

    NONE = 0
    SUCCESS = 1
    TOPPLE = 2
    BOARD_CONTACT = 3
    HEIGHT_LIMIT = 4
    TIMEOUT = 5
    INVALID_SIMULATION = 6
    OUT_OF_BOUNDS = 7
    INVALID_RESET = 8


@dataclass(frozen=True)
class TerminationDecision:
    """Resolved Gymnasium-style episode outcome."""

    terminated: Tensor
    truncated: Tensor
    failed: Tensor
    succeeded: Tensor
    timed_out: Tensor
    errored: Tensor
    reason: Tensor

    @property
    def done(self) -> Tensor:
        return self.terminated | self.truncated


def threshold_exceeded(value: Tensor, hard_threshold: float | Tensor) -> Tensor:
    """Use the guide's strict 'exceeds hard threshold' comparison."""

    threshold = torch.as_tensor(hard_threshold, dtype=value.dtype, device=value.device)
    return value > threshold


def success_from_positions(
    object_position: Tensor,
    goal_position: Tensor,
    *,
    distance_threshold: float = 0.01,
) -> Tensor:
    """Candidate success using strict 3D distance error below 0.01 m."""

    if object_position.ndim < 1 or object_position.shape[-1] != 3:
        raise ValueError("object_position must have shape (..., 3)")
    if goal_position.shape != object_position.shape:
        raise ValueError("goal_position must have the same shape as object_position")
    if distance_threshold <= 0:
        raise ValueError("distance_threshold must be positive")
    return torch.linalg.vector_norm(object_position - goal_position, dim=-1) < distance_threshold


def latch_threshold(previous_latch: Tensor, value: Tensor, hard_threshold: float | Tensor) -> Tensor:
    """Preserve a physics-substep threshold crossing until episode reset."""

    return previous_latch.to(dtype=torch.bool) | threshold_exceeded(value, hard_threshold)


def _optional_mask(mask: Tensor | None, reference: Tensor) -> Tensor:
    if mask is None:
        return torch.zeros_like(reference, dtype=torch.bool)
    mask = mask.to(device=reference.device, dtype=torch.bool)
    if mask.shape != reference.shape:
        raise ValueError("all termination masks must have identical shapes")
    return mask


def resolve_terminations(
    *,
    success_candidate: Tensor,
    topple: Tensor,
    board_contact: Tensor,
    height_limit: Tensor,
    timeout: Tensor,
    invalid_simulation: Tensor | None = None,
    out_of_bounds: Tensor | None = None,
) -> TerminationDecision:
    """Resolve simultaneous outcomes with failure > success > timeout.

    Timeout is a truncation, never a termination.  Invalid simulation is kept
    separate from task failure while still blocking success.  When multiple
    task failures coincide, the diagnostic primary reason order is topple,
    board contact, height, then optional out-of-bounds; every raw mask remains
    available to the caller before resolution.
    """

    success_candidate = success_candidate.to(dtype=torch.bool)
    masks = (topple, board_contact, height_limit, timeout)
    if any(mask.shape != success_candidate.shape for mask in masks):
        raise ValueError("all termination masks must have identical shapes")
    topple = topple.to(device=success_candidate.device, dtype=torch.bool)
    board_contact = board_contact.to(device=success_candidate.device, dtype=torch.bool)
    height_limit = height_limit.to(device=success_candidate.device, dtype=torch.bool)
    timeout = timeout.to(device=success_candidate.device, dtype=torch.bool)
    invalid = _optional_mask(invalid_simulation, success_candidate)
    outside = _optional_mask(out_of_bounds, success_candidate)

    failed = topple | board_contact | height_limit | outside
    blocking = failed | invalid
    succeeded = success_candidate & ~blocking
    timed_out = timeout & ~blocking & ~succeeded
    terminated = blocking | succeeded
    truncated = timed_out

    reason = torch.full_like(success_candidate, int(TerminationReason.NONE), dtype=torch.int64)
    reason = torch.where(
        timed_out, torch.full_like(reason, int(TerminationReason.TIMEOUT)), reason
    )
    reason = torch.where(
        succeeded, torch.full_like(reason, int(TerminationReason.SUCCESS)), reason
    )
    reason = torch.where(
        outside, torch.full_like(reason, int(TerminationReason.OUT_OF_BOUNDS)), reason
    )
    reason = torch.where(
        height_limit, torch.full_like(reason, int(TerminationReason.HEIGHT_LIMIT)), reason
    )
    reason = torch.where(
        board_contact, torch.full_like(reason, int(TerminationReason.BOARD_CONTACT)), reason
    )
    reason = torch.where(
        topple, torch.full_like(reason, int(TerminationReason.TOPPLE)), reason
    )
    reason = torch.where(
        invalid, torch.full_like(reason, int(TerminationReason.INVALID_SIMULATION)), reason
    )
    return TerminationDecision(
        terminated=terminated,
        truncated=truncated,
        failed=failed,
        succeeded=succeeded,
        timed_out=timed_out,
        errored=invalid,
        reason=reason,
    )


# Isaac Lab manager adapters.  Hard conditions are latched by the custom
# environment at physics-substep frequency before these policy-step queries.


def topple_failure(env) -> Tensor:
    return env.tilt_hard_latched


def board_contact_failure(env) -> Tensor:
    return env.board_hard_latched


def height_limit_failure(env) -> Tensor:
    return env.height_hard_latched


def task_success(env) -> Tensor:
    candidate = success_from_positions(
        env.scene["target_object"].data.root_pos_w,
        env.goal_pos_w,
        distance_threshold=env.cfg.task.success_distance_m,
    )
    failed = (
        env.tilt_hard_latched
        | env.board_hard_latched
        | env.height_hard_latched
        | object_out_of_bounds(env)
        | env.invalid_reset_latched
    )
    return candidate & ~failed


def invalid_reset_contact(env) -> Tensor:
    """Expose a reset-certificate failure if a future backend sets the latch."""

    return env.invalid_reset_latched


def episode_timeout(env) -> Tensor:
    """Return a pure truncation mask with terminal precedence.

    Isaac Lab increments ``episode_length_buf`` before evaluating terms.  The
    official condition is therefore ``>= max_episode_length`` (without a
    ``-1``).  Masking terminal outcomes here, rather than only on the value
    returned by :meth:`BlindSweepEnv.step`, also prevents a simultaneous
    timeout from polluting TerminationManager's per-term episode logs.
    """

    expired = env.episode_length_buf >= env.max_episode_length
    terminal = (
        env.tilt_hard_latched
        | env.board_hard_latched
        | env.height_hard_latched
        | object_out_of_bounds(env)
        | env.invalid_reset_latched
        | task_success(env)
    )
    return expired & ~terminal


def object_out_of_bounds(env) -> Tensor:
    """Optional protective guard; disabled unless configured explicitly."""

    if not env.cfg.task.enable_out_of_bounds_termination:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    position = env.scene["target_object"].data.root_pos_w - env.scene.env_origins
    center = torch.tensor(env.cfg.task.board_center, device=env.device)
    half = 0.5 * torch.tensor(env.cfg.task.board_size, device=env.device)
    margin = env.cfg.task.out_of_bounds_margin_m
    lower = center[:2] - half[:2] - margin
    upper = center[:2] + half[:2] + margin
    return torch.any((position[:, :2] < lower) | (position[:, :2] > upper), dim=-1)
