"""Dense settling reward and independent strict, sustained success.

One live goal radius for settling, success and contact credit."""

from isaaclab.utils.math import quat_apply
from isaaclab.utils.math import quat_apply_inverse
import math as math
import torch as torch

def target_measurements(env, command_name='target_goal_pos'):
    data = env.scene['object_collection'].data
    rows = torch.arange(env.num_envs, device=env.device)
    ids = env.target_id[:, 0].long()
    pos, vel = data.object_pos_w[rows,ids], data.object_lin_vel_w[rows,ids]
    goal = env.command_manager.get_command(command_name)
    shelf = env.scene['shelf'].data
    local = quat_apply_inverse(shelf.root_quat_w, pos-shelf.root_pos_w)
    up = torch.zeros_like(pos); up[:,2] = 1.
    target_up = quat_apply(data.object_quat_w[rows,ids], up)
    alignment = (target_up*quat_apply(shelf.root_quat_w,up)).sum(-1)
    valid = (torch.isfinite(pos).all(-1) & torch.isfinite(vel).all(-1)
             & torch.isfinite(goal[:,:2]).all(-1) & torch.isfinite(alignment)
             & (local[:,0].abs() <= .16) & (local[:,1].abs() <= .46)
             & (local[:,2] >= 1.04) & (local[:,2] <= 1.07)
             & (alignment >= math.cos(.9)))
    distance = (pos[:,:2]-goal[:,:2]).norm(dim=-1)
    speed = vel.norm(dim=-1)
    return distance, speed, valid


def goal_radius(env):
    """Read the reward manager so curriculum load/rewind reaches every consumer."""
    return env.reward_manager.get_term_cfg("goal_hold").params["position_tolerance"]
