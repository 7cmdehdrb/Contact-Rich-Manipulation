"""Revision hold, contact-height gate and inclusive timeout bonus."""

import math

import torch

from ..common.contact_sweep_clutter_fusion import SHELF_SURFACE_HEIGHT
from ..common.contact_sweep_clutter_fusion_contacts import target_hand_contact_force_w
from ..common.revision_goal import GOAL_SUCCESS_RADIUS, HOLD_OUTER_RADIUS, target_measurements


def goal_hold(env, position_tolerance=GOAL_SUCCESS_RADIUS, outer_radius=HOLD_OUTER_RADIUS,
              speed_scale=0.05, command_name="target_goal_pos"):
    """The v0_2 continuous settling curve with bounds-aware bottom validation."""
    if not all(math.isfinite(v) and v > 0 for v in
               (position_tolerance, outer_radius, speed_scale)) or outer_radius <= position_tolerance:
        raise ValueError("Expected 0 < goal radius < outer radius and positive speed scale")
    if position_tolerance != env.cfg.goal_success_radius:
        raise ValueError("Revision Hold and success radius must remain synchronized")
    distance, speed, valid = target_measurements(env, command_name)
    inner = torch.exp(-math.log(2) * (distance / position_tolerance).square())
    inner *= 1.0 + (1.0 - distance / position_tolerance).clamp(0.0, 1.0).square()
    outer = 0.5 * ((outer_radius - distance) / (outer_radius - position_tolerance)).clamp(0.0, 1.0).square()
    rate = torch.where(distance <= position_tolerance, inner, outer)
    rate *= torch.exp(-math.log(2) * (speed / speed_scale).square())
    return torch.where(valid, rate, 0.0)


def target_contact_below_height(env, threshold_n=0.01, frame_name="ee_frame"):
    """Require actual fresh target/hand contact AND EEF below one fixed height.

    The work-surface reference is captured before physics starts. Neither the
    target shape/scale/pose nor later shelf motion changes the height threshold.
    Equality and non-finite EEF heights fail the strict height predicate.
    """
    height_limit = float(env.cfg.contact_height_limit)
    if not math.isfinite(threshold_n) or threshold_n <= 0:
        raise ValueError("Contact threshold must be finite and positive")
    if not math.isfinite(height_limit) or height_limit <= 0:
        raise ValueError("Contact height limit must be finite and positive")
    force = target_hand_contact_force_w(env)
    eef_z = env.scene[frame_name].data.target_pos_w[:, 0, 2]
    ceiling = env._revision_initial_shelf_pos[:, 2] + SHELF_SURFACE_HEIGHT + height_limit
    contact = (
        env.sensor_data_fresh
        & torch.isfinite(force).all(-1)
        & (torch.linalg.vector_norm(force, dim=-1) >= threshold_n)
        & torch.isfinite(eef_z)
        & (eef_z < ceiling)
    )
    return contact.to(force.dtype)


def successful_timeout_bonus(env):
    """Reward the same inclusive success partition exactly once, before reset."""
    return env.termination_manager.get_term("target_goal_success_at_timeout").float() / env.step_dt
