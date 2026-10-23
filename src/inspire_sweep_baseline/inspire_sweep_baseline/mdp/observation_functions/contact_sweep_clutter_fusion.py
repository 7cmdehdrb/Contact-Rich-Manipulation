"""Initial target-to-goal command in the original robot-root coordinate axes."""

from isaaclab.utils.math import quat_apply_inverse
import math as math
import torch as torch

def initial_goal_polar_in_root(env, command_name="target_goal_pos"):
    """Return [distance metres, clockwise azimuth radians] from the initial target.

    Root +X is straight into the shelf, so +X, -Y, -X, +Y correspond to
    0, pi/2, pi, 3*pi/2. Both positions come from the episode-fixed command;
    the actor does not learn current target displacement through this input.
    The command itself remains the same absolute world-frame goal.
    """
    command = env.command_manager.get_term(command_name)
    displacement_w = command.command[:, :3] - command.target_init_state_w[:, :3]
    displacement_root = quat_apply_inverse(env.scene["robot"].data.root_quat_w, displacement_w)
    distance = displacement_root[:, :2].norm(dim=-1)
    theta = torch.remainder(torch.atan2(-displacement_root[:, 1], displacement_root[:, 0]), 2.0 * math.pi)
    # A zero-length command has no direction; make its representation explicit.
    theta = torch.where(distance > 0.0, theta, torch.zeros_like(theta))
    return torch.stack((distance, theta), dim=-1)
