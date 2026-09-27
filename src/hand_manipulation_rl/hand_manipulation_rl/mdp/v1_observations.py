"""Policy observation extension for the corrective Approach-v1 task."""

from __future__ import annotations

import torch

from .observations import policy_observation


V1_OBSERVATION_DIM = 60


def approach_v1_policy_observation(env) -> torch.Tensor:
    """Append selected-surface target error in task-action axis order.

    The target is the episode-fixed, calibrated pre-contact frontier, so this
    adds no online object tracking to the blind policy.  The components are
    ``[world-up, toward-object, shelf-depth(+X)]``, exactly matching the v1
    translation action axes and remaining invariant across palm/dorsal and
    sweep direction.
    """

    base = policy_observation(env)
    error_w = (
        env.selected_surface_approach_target_w()
        - env.selected_surface_control_point_w()
    )
    error_task = torch.stack(
        (
            error_w[:, 2],
            torch.sum(error_w * env.command_direction_w, dim=-1),
            error_w[:, 0],
        ),
        dim=-1,
    )
    scaled_error = torch.clamp(
        error_task / env.cfg.task.position_observation_scale_m,
        -1.0,
        1.0,
    )
    observation = torch.cat((base, scaled_error), dim=-1)
    if observation.shape != (env.num_envs, V1_OBSERVATION_DIM):
        raise RuntimeError(f"Invalid v1 policy observation shape: {tuple(observation.shape)}")
    if not torch.isfinite(observation).all():
        raise RuntimeError("v1 policy observation contains non-finite values")
    return observation


__all__ = ["V1_OBSERVATION_DIM", "approach_v1_policy_observation"]
