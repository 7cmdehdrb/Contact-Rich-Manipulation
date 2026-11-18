"""Episode-fixed visible points transformed by the current EEF pose."""
import torch
from isaaclab.utils.math import quat_apply_inverse


def target_surface_in_eef(env):
    frame=env.scene['ee_frame'].data
    return quat_apply_inverse(frame.target_quat_w[:,0],env._surface_target_w-frame.target_pos_w[:,0])


def obstacle_surfaces_in_eef(env):
    frame=env.scene['ee_frame'].data
    points=quat_apply_inverse(frame.target_quat_w[:,0,None].expand(-1,16,-1),
                              env._surface_obstacles_w-frame.target_pos_w[:,0,None])
    valid=env._surface_valid
    return torch.cat((valid[...,None].to(points.dtype),torch.where(valid[...,None],points,0.)),-1).flatten(1)


def phase(env):
    return env._sweep_phase[:,None].float()
