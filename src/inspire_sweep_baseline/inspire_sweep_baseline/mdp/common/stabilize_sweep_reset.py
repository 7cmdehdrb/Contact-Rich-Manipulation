"""Reset pose calculations in the shelf frame; no simulation writes."""
import math
import torch
from isaaclab.utils.math import quat_apply, quat_mul


def hand_pose(center, direction, target_radius, default_pos, default_quat,
              hand_vertices_eef, draws):
    """Face EEF -Z toward the goal, separating the facing surfaces by 5cm.

    draws supplies three uniform [0,1] variates (shelf X/Y and yaw). Keeping
    random draws outside this calculation makes batch reset reproducible.
    """
    normal = quat_apply(default_quat, default_pos.new_tensor([0., 0., -1.]).expand_as(default_pos))
    yaw = torch.atan2(direction[:, 1], direction[:, 0]) - torch.atan2(normal[:, 1], normal[:, 0])
    # Surface gap belongs to the nominal pose, before the agreed XY/yaw jitter.
    rotation = default_quat.new_zeros(default_quat.shape)
    rotation[:, 0], rotation[:, 3] = (yaw / 2).cos(), (yaw / 2).sin()
    nominal = quat_mul(rotation, default_quat)
    vertices = hand_vertices_eef.expand(len(center), -1, -1)
    transformed = quat_apply(nominal[:, None].expand(-1, vertices.shape[1], -1), vertices)
    support = (transformed * direction[:, None]).sum(-1).amax(-1)
    pos = default_pos.clone()
    pos[:, :2] = center[:, :2] - direction[:, :2] * (target_radius + .05 + support)[:, None]
    pos[:, :2] += (2 * draws[:, :2] - 1) * .01
    yaw += (2 * draws[:, 2] - 1) * math.radians(10)
    rotation[:, 0], rotation[:, 3] = (yaw / 2).cos(), (yaw / 2).sin()
    return pos, quat_mul(rotation, default_quat)
