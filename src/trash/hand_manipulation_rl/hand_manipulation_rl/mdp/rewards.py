"""Pure reward terms for the blind-sweeping task."""

from __future__ import annotations

import torch


Tensor = torch.Tensor


def _require_vector3(value: Tensor, name: str) -> None:
    if value.ndim < 1 or value.shape[-1] != 3:
        raise ValueError(f"{name} must have shape (..., 3)")


def _positive_scalar_like(value: float | Tensor, like: Tensor, name: str) -> Tensor:
    result = torch.as_tensor(value, dtype=like.dtype, device=like.device)
    if torch.any(result <= 0):
        raise ValueError(f"{name} must be positive")
    return result


def goal_reward(
    object_position: Tensor,
    goal_position: Tensor,
    sigma_position: float | Tensor,
) -> Tensor:
    """Return ``exp(-||p_object - p_goal|| / sigma_position)``."""

    _require_vector3(object_position, "object_position")
    _require_vector3(goal_position, "goal_position")
    sigma = _positive_scalar_like(sigma_position, object_position, "sigma_position")
    error = torch.linalg.vector_norm(object_position - goal_position, dim=-1)
    return torch.exp(-error / sigma)


def resultant_contact_normal(
    contact_normals_on_object: Tensor,
    normal_loads: Tensor,
    *,
    contact_mask: Tensor | None = None,
    epsilon: float = 1.0e-8,
) -> tuple[Tensor, Tensor]:
    """Build the load-weighted hand-on-object resultant unit normal.

    Returns the unit direction and a mask indicating that the resultant is
    sufficiently large to define a direction.
    """

    if contact_normals_on_object.ndim < 2 or contact_normals_on_object.shape[-1] != 3:
        raise ValueError("contact_normals_on_object must have shape (..., contacts, 3)")
    if normal_loads.shape != contact_normals_on_object.shape[:-1]:
        raise ValueError("normal_loads must match the contact dimensions")
    if torch.any(normal_loads < 0):
        raise ValueError("normal_loads must be non-negative")

    normal_norm = torch.linalg.vector_norm(contact_normals_on_object, dim=-1, keepdim=True)
    unit_normals = contact_normals_on_object / torch.clamp_min(normal_norm, epsilon)
    loads = normal_loads
    if contact_mask is not None:
        if contact_mask.shape != normal_loads.shape:
            raise ValueError("contact_mask must have the same shape as normal_loads")
        loads = loads * contact_mask.to(dtype=loads.dtype)
    resultant = torch.sum(loads.unsqueeze(-1) * unit_normals, dim=-2)
    magnitude = torch.linalg.vector_norm(resultant, dim=-1, keepdim=True)
    valid = magnitude.squeeze(-1) > epsilon
    return resultant / torch.clamp_min(magnitude, epsilon), valid


def contact_normal_alignment_reward(
    contact_normal_on_object: Tensor,
    command_direction: Tensor,
    *,
    valid_contact: Tensor | None = None,
    epsilon: float = 1.0e-8,
) -> Tensor:
    """Reward actual hand-on-object normal alignment in ``[-1, 0]``.

    Undefined or absent contact has value zero, as required by the task
    contract.
    """

    _require_vector3(contact_normal_on_object, "contact_normal_on_object")
    _require_vector3(command_direction, "command_direction")
    normal_norm = torch.linalg.vector_norm(contact_normal_on_object, dim=-1)
    direction_norm = torch.linalg.vector_norm(command_direction, dim=-1)
    valid = (normal_norm > epsilon) & (direction_norm > epsilon)
    if valid_contact is not None:
        valid &= valid_contact.to(dtype=torch.bool)
    normal = contact_normal_on_object / torch.clamp_min(normal_norm.unsqueeze(-1), epsilon)
    direction = command_direction / torch.clamp_min(direction_norm.unsqueeze(-1), epsilon)
    dot = torch.clamp(torch.sum(normal * direction, dim=-1), -1.0, 1.0)
    value = -(1.0 - dot) * 0.5
    return torch.where(valid, value, torch.zeros_like(value))


def tactile_contact_reward(
    selected_surface_tactile: Tensor,
    *,
    beta: float = 0.5,
    region_weights: Tensor | None = None,
) -> Tensor:
    """Compute contact-presence plus weighted active-region reward.

    Input contains only the selected surface's 17 bits; the mode header is
    deliberately absent and therefore can never be counted as contact.
    """

    if selected_surface_tactile.ndim < 1 or selected_surface_tactile.shape[-1] != 17:
        raise ValueError("selected_surface_tactile must have shape (..., 17)")
    if not 0.0 <= beta <= 1.0:
        raise ValueError("beta must be in [0, 1]")
    active = selected_surface_tactile > 0
    reward_dtype = (
        selected_surface_tactile.dtype
        if selected_surface_tactile.dtype.is_floating_point
        else torch.get_default_dtype()
    )
    if region_weights is None:
        weights = torch.ones(
            17,
            dtype=reward_dtype,
            device=selected_surface_tactile.device,
        )
    else:
        weights = torch.as_tensor(
            region_weights,
            dtype=reward_dtype,
            device=selected_surface_tactile.device,
        )
        if weights.shape[-1:] != (17,):
            raise ValueError("region_weights must end in dimension 17")
        if torch.any(weights < 0) or torch.any(weights.sum(dim=-1) <= 0):
            raise ValueError("region_weights must be non-negative with positive sum")
    active_fraction = torch.sum(active.to(reward_dtype) * weights, dim=-1) / weights.sum(
        dim=-1
    )
    contact_present = torch.any(active, dim=-1).to(dtype=active_fraction.dtype)
    return beta * contact_present + (1.0 - beta) * active_fraction - 1.0


def target_gated_tactile_contact_reward(
    selected_surface_tactile: Tensor,
    target_contact: Tensor,
    *,
    beta: float = 0.5,
) -> Tensor:
    """Reward selected tactile activity only during verified object contact.

    Unlike :func:`tactile_contact_reward`, no contact is neutral rather than a
    penalty.  A shelf-only tactile event is also exactly zero, preventing the
    board collision shortcut while leaving the actor's raw tactile observation
    unchanged.
    """

    expected = selected_surface_tactile.shape[:-1]
    if target_contact.shape != expected:
        raise ValueError("target_contact must match the tactile batch shape")
    shifted = tactile_contact_reward(selected_surface_tactile, beta=beta) + 1.0
    return torch.where(target_contact.to(dtype=torch.bool), shifted, torch.zeros_like(shifted))


def action_change_reward(
    normalized_action: Tensor,
    previous_normalized_action: Tensor,
    *,
    previous_action_valid: Tensor | None = None,
    arm_coefficient: float = 1.0,
    hand_coefficient: float = 1.0,
) -> Tensor:
    """Penalize changes in executed normalized arm and hand actions."""

    if normalized_action.shape != previous_normalized_action.shape:
        raise ValueError("current and previous action tensors must have identical shapes")
    if normalized_action.ndim < 1 or normalized_action.shape[-1] != 8:
        raise ValueError("actions must have shape (..., 8)")
    if arm_coefficient < 0 or hand_coefficient < 0:
        raise ValueError("action coefficients must be non-negative")
    difference = normalized_action - previous_normalized_action
    arm = torch.sum(difference[..., :6] ** 2, dim=-1)
    hand = torch.sum(difference[..., 6:] ** 2, dim=-1)
    value = -arm_coefficient * arm - hand_coefficient * hand
    if previous_action_valid is not None:
        value = torch.where(
            previous_action_valid.to(dtype=torch.bool), value, torch.zeros_like(value)
        )
    return value


def time_penalty_rate(reference_time: float | Tensor) -> float | Tensor:
    """Return ``-1 / T_ref`` for a manager that multiplies rewards by dt."""

    if isinstance(reference_time, Tensor):
        if torch.any(reference_time <= 0):
            raise ValueError("reference_time must be positive")
        return -torch.ones_like(reference_time) / reference_time
    if reference_time <= 0:
        raise ValueError("reference_time must be positive")
    return -1.0 / reference_time


def quadratic_risk_ramp(
    value: Tensor,
    soft_threshold: float | Tensor,
    hard_threshold: float | Tensor,
) -> Tensor:
    """Return the positive quadratic ramp psi in ``[0, 1]``."""

    soft = torch.as_tensor(soft_threshold, dtype=value.dtype, device=value.device)
    hard = torch.as_tensor(hard_threshold, dtype=value.dtype, device=value.device)
    if torch.any(hard <= soft):
        raise ValueError("hard_threshold must be greater than soft_threshold")
    normalized = torch.clamp((value - soft) / (hard - soft), 0.0, 1.0)
    return normalized * normalized


def risk_penalty(
    value: Tensor,
    soft_threshold: float | Tensor,
    hard_threshold: float | Tensor,
) -> Tensor:
    """Return the signed risk term ``-psi``."""

    return -quadratic_risk_ramp(value, soft_threshold, hard_threshold)


def upright_tilt_angle(current_upright: Tensor, initial_upright: Tensor) -> Tensor:
    """Angle between current and initial upright axes, independent of yaw."""

    _require_vector3(current_upright, "current_upright")
    _require_vector3(initial_upright, "initial_upright")
    current = current_upright / torch.clamp_min(
        torch.linalg.vector_norm(current_upright, dim=-1, keepdim=True), 1.0e-8
    )
    initial = initial_upright / torch.clamp_min(
        torch.linalg.vector_norm(initial_upright, dim=-1, keepdim=True), 1.0e-8
    )
    cosine = torch.clamp(torch.sum(current * initial, dim=-1), -1.0, 1.0)
    return torch.acos(cosine)


def height_increase(current_height_s: Tensor, initial_height_s: Tensor) -> Tensor:
    """Cube-center height increase relative to episode start."""

    return current_height_s - initial_height_s


def sum_link_normal_force(link_normal_force_vectors: Tensor) -> Tensor:
    """Sum per-link normal-force magnitudes without vector cancellation."""

    if link_normal_force_vectors.ndim < 2 or link_normal_force_vectors.shape[-1] != 3:
        raise ValueError("link_normal_force_vectors must have shape (..., links, 3)")
    return torch.linalg.vector_norm(link_normal_force_vectors, dim=-1).sum(dim=-1)


# ---------------------------------------------------------------------------
# Isaac Lab manager adapters.  They intentionally use duck typing so the pure
# numerical functions above remain importable and unit-testable without Kit.
# ---------------------------------------------------------------------------


def object_goal_reward(env) -> Tensor:
    return goal_reward(
        env.scene["target_object"].data.root_pos_w,
        env.goal_pos_w,
        env.cfg.task.goal_reward_sigma_m,
    )


def actual_contact_normal_alignment(env) -> Tensor:
    sample = env.hand_object_normal()
    return contact_normal_alignment_reward(
        sample.normal_w,
        env.command_direction_w,
        valid_contact=sample.valid,
    )


def selected_tactile_contact(env) -> Tensor:
    bits = env.selected_tactile_bits()
    sensor_live = env.sensor_valid & env.sensor_data_fresh
    bits = torch.where(sensor_live.unsqueeze(-1), bits, torch.zeros_like(bits))
    return tactile_contact_reward(bits, beta=env.cfg.task.tactile_contact_beta)


def normalized_action_change(env) -> Tensor:
    return action_change_reward(
        env.current_policy_action,
        env.previous_policy_action,
        previous_action_valid=env.has_previous_policy_action,
        arm_coefficient=env.cfg.task.arm_action_change_coefficient,
        hand_coefficient=env.cfg.task.hand_action_change_coefficient,
    )


def time_cost_rate(env) -> Tensor:
    return torch.full(
        (env.num_envs,),
        float(time_penalty_rate(env.cfg.task.time_reference_s)),
        dtype=torch.float32,
        device=env.device,
    )


def tilt_risk_penalty(env) -> Tensor:
    return risk_penalty(
        env.current_tilt,
        env.cfg.task.tilt_soft_rad,
        env.cfg.task.tilt_hard_rad,
    )


def board_contact_risk_penalty(env) -> Tensor:
    return risk_penalty(
        env.board_force,
        env.cfg.task.board_force_soft_n,
        env.cfg.task.board_force_hard_n,
    )


def object_height_risk_penalty(env) -> Tensor:
    return risk_penalty(
        env.current_height_increase,
        env.cfg.task.height_soft_m,
        env.cfg.task.height_hard_m,
    )


def success_terminal_bonus_rate(env) -> Tensor:
    """Optional one-shot bonus divided by dt to cancel RewardManager's dt."""

    if env.cfg.task.success_terminal_bonus == 0.0:
        return torch.zeros(env.num_envs, device=env.device)
    distance = torch.linalg.vector_norm(
        env.scene["target_object"].data.root_pos_w - env.goal_pos_w, dim=-1
    )
    failed = (
        env.tilt_hard_latched
        | env.board_hard_latched
        | env.height_hard_latched
        | env.invalid_reset_latched
        | env.termination_manager.get_term_cfg("out_of_bounds").func(env)
    )
    success = (distance < env.cfg.task.success_distance_m) & ~failed
    return success.float() * (env.cfg.task.success_terminal_bonus / env.step_dt)


def failure_terminal_penalty_rate(env) -> Tensor:
    """Optional one-shot penalty divided by dt to cancel RewardManager's dt."""

    if env.cfg.task.failure_terminal_penalty == 0.0:
        return torch.zeros(env.num_envs, device=env.device)
    # Invalid reset contact is an environment sampling rejection, not an agent
    # failure, and therefore receives no policy penalty.
    failed = (
        env.tilt_hard_latched
        | env.board_hard_latched
        | env.height_hard_latched
        | env.termination_manager.get_term_cfg("out_of_bounds").func(env)
    )
    return failed.float() * (env.cfg.task.failure_terminal_penalty / env.step_dt)
