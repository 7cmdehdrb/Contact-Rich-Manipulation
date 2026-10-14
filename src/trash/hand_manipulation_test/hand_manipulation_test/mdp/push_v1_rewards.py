"""Full palm-normal alignment and optional reset-preferred roll for Push-v1."""

from __future__ import annotations

import torch


def push_v1_alignment_reward(env):
    """Keep the full 3-D palm normal aligned before and after contact."""

    return env.command_manager.get_term("target_position").state().alignment_penalty


def push_v1_roll_reward(env):
    """Penalize roll away from the collision-safe pose selected at reset.

    This is a nonpositive rate, so manager dt integration preserves its cost
    per second. It gives no stationary positive bonus and is zero on failure
    or reset rows, exactly as the normal-alignment term.
    """

    command = env.command_manager.get_term("target_position")
    command.state()
    return command.roll_penalty


def push_v1_action_excess_reward(env):
    """Penalize Gaussian input magnitude beyond the executed +/-1 range.

    The physical action-rate term measures changes after clipping. This
    separate nonpositive rate measures excess caller input before clipping,
    so growing an ineffective Gaussian scale has a cost while the requested
    PPO distribution and hyperparameters remain unchanged. At eight inputs
    of magnitude 2, this term returns -1; weight 0.02 costs 0.02 per second.
    The manager applies dt once. Invalid inputs are sanitized like the action
    terms, and the finite magnitude cap prevents squared overflow.
    """

    actions = env.action_manager.action.to(dtype=torch.float32)
    safe = torch.nan_to_num(actions, nan=0., posinf=0., neginf=0.)
    excess = (safe.abs().clamp_max(1.e6) - 1.).clamp_min(0.)
    return -excess.square().mean(-1)


def push_v1_contact_distance_penalty(env):
    """Charge actual distance from the current palmar contact target every step.

    The existing approach impulse only pays a new record. This nonpositive
    rate also discourages leaving the object after setting a record or losing
    contact; hovering at the object cannot collect a positive proximity bonus.
    """
    command = env.command_manager.get_term("target_position")
    command.state()
    return command.contact_distance_penalty


def push_v1_eef_height_penalty(env):
    """Linearly penalize actual C height above the configurable soft limit."""
    command = env.command_manager.get_term("target_position")
    command.state()
    return command.eef_height_penalty


__all__ = ["push_v1_alignment_reward", "push_v1_roll_reward", "push_v1_action_excess_reward",
           "push_v1_contact_distance_penalty", "push_v1_eef_height_penalty"]
