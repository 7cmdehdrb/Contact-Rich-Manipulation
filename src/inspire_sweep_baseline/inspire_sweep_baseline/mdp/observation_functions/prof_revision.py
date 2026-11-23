"""Frozen initial OBB observations, expressed in the current EEF frame."""
import torch
from ..common.prof_revision_geometry import obb_in_eef


def target_obb_in_eef(env):
    return obb_in_eef(env, env._obb_target_pos_w, env._obb_target_quat_w, env._obb_target_size)


def obstacle_obbs_in_eef(env):
    boxes = obb_in_eef(env, env._obb_obstacle_pos_w, env._obb_obstacle_quat_w, env._obb_obstacle_size)
    mask = env._obb_valid[..., None]
    return torch.cat((mask.to(boxes.dtype), torch.where(mask, boxes, 0.)), -1).flatten(1)


def finger_poses_in_eef(env):
    return env._finger_pose_e.flatten(1)
